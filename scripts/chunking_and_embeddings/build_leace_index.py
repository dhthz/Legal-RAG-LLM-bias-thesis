import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime

import faiss
import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.chunking_and_embeddings.index_variants import file_sha256, record_index_variant
from src.llm.config import PipelineConfig
from src.mitigation.finetuned_embedder import load_encoder
from src.mitigation.leace import LEACE_ERASER_PATH, LEACE_INDEX_PATH, LEACE_QUERY_ERASER_PATH, erase

# Builds arm 6 (LEACE, Belrose et al. 2023): fits the eraser on the frozen chunk vectors (reconstructed exactly from the
# IndexFlatL2) with each chunk's case gender as the concept, writes the erased vectors to a new flat
# index (same chunk order, so the frozen metadata applies), and checks the erasure with linear probes on held-out chunks.
# The frozen index is only read. Output: faiss_indices/leace_*, the manifest's index_variants.leace entry, and
# logs/mitigation/leace/build.json. CPU only.
# --concept query_pronoun builds the revised arm (leace_query): the concept is the pronoun gender of the query, fit on
# the he/she twins of the dev-split queries built for arm 4 (logs/mitigation/cda/pairs.jsonl, train split), embedded as
# the retriever embeds queries; held-out probe on the dev validation twins. Output: faiss_indices/leace_query_eraser.npz,
# index_variants.leace_query, logs/mitigation/leace_query/build.json.

TRAIN_METADATA_PATH = "dataset/train_with_metadata.jsonl"
REPORT_PATH = "logs/mitigation/leace/build.json"
CLASSES = ["Male", "Female", "Multiple Applicants", "Unknown"]
SEED = 42
PROBE_TRAIN, PROBE_TEST = 50000, 20000
TWINS_PATH = "logs/mitigation/cda/pairs.jsonl"
QUERY_REPORT_PATH = "logs/mitigation/leace_query/build.json"


def chunk_labels(metadata_path):
    case_gender = {}
    with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            case_gender[rec["case_id"]] = rec.get("classification", {}).get("gender", "Unknown")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    labels = [case_gender.get(m["case_id"], "Unknown") for m in metadata]
    unexpected = set(labels) - set(CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected gender labels: {unexpected}")
    return np.array([CLASSES.index(g) for g in labels])


# Held-out linear probe: can gender still be read off the vectors? Chance = 0.25 balanced accuracy (4 classes)
# and AUC 0.5 for Male vs Female
def probe(vectors, labels, train_idx, test_idx):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(vectors[train_idx], labels[train_idx])
    pred = clf.predict(vectors[test_idx])
    proba = clf.predict_proba(vectors[test_idx])
    mf = np.isin(labels[test_idx], [0, 1])
    male_score = proba[mf, 0] - proba[mf, 1]
    return {"balanced_accuracy_4class": float(balanced_accuracy_score(labels[test_idx], pred)),
            "auc_male_vs_female": float(roc_auc_score(labels[test_idx][mf] == 0, male_score))}


def fit_eraser(x, z):
    eraser = LeaceEraser.fit(torch.from_numpy(x).double(), torch.from_numpy(z).double())
    params = {"bias": eraser.bias.numpy(), "proj_left": eraser.proj_left.numpy(), "proj_right": eraser.proj_right.numpy()}
    # The numpy transform used at query time must match the library's eraser
    ref = eraser(torch.from_numpy(x[:1000]).double()).numpy()
    assert np.allclose(erase(x[:1000], params), ref, atol=1e-5), "numpy erase() differs from LeaceEraser"
    return params


def he_she_vectors(model, rows):
    he = [r["query"] if r["gender"] == "Male" else r["twin"] for r in rows]
    she = [r["twin"] if r["gender"] == "Male" else r["query"] for r in rows]
    encode = lambda qs: np.vstack([model.encode([q], convert_to_numpy=True, normalize_embeddings=True) for q in qs])
    x = np.vstack([encode(he), encode(she)])
    return x, np.r_[np.zeros(len(rows)), np.ones(len(rows))]


def build_query_pronoun(device):
    start = time.time()
    with open(TWINS_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    train, val = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    model = load_encoder("nomic-ai/nomic-embed-text-v1", device=device)
    print(f"Embedding {len(train)} + {len(val)} he/she twins on {device} ...")
    x, y = he_she_vectors(model, train)
    vx, vy = he_she_vectors(model, val)

    params = fit_eraser(x, y[:, None])
    np.savez(LEACE_QUERY_ERASER_PATH, **params)
    probes = {}
    for name, f in (("original", lambda v: v), ("erased", lambda v: erase(v, params))):
        clf = LogisticRegression(max_iter=5000).fit(f(x), y)
        probes[name] = {"auc_he_vs_she": float(roc_auc_score(vy, clf.predict_proba(f(vx))[:, 1]))}
        print(f"Probe he vs she on held-out dev twins, {name}: AUC {probes[name]['auc_he_vs_she']:.3f}")

    report = {"created": datetime.now().isoformat(timespec="seconds"),
              "method": "LEACE (Belrose et al. 2023), concept-erasure library defaults; concept = pronoun gender of the "
                        "query (he = 0, she = 1); applied to the query vector only, the frozen index is searched as is",
              "fit_data": {"path": TWINS_PATH, "sha256": file_sha256(TWINS_PATH), "train_twins": len(train),
                           "embedding": f"frozen model, warm-up and one query at a time as ChunkRetriever, {device}"},
              "eraser_rank": int(np.linalg.matrix_rank(params["proj_left"] @ params["proj_right"])),
              "probe": {"design": f"logistic regression fit on the {len(train)} train twins, tested on the {len(val)} "
                                  f"held-out dev validation twins", **probes},
              "minutes": round((time.time() - start) / 60, 1)}
    report["index_variant"] = record_index_variant("leace_query", eraser_file=LEACE_QUERY_ERASER_PATH)
    os.makedirs(os.path.dirname(QUERY_REPORT_PATH), exist_ok=True)
    with open(QUERY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {LEACE_QUERY_ERASER_PATH}, manifest index_variants.leace_query, report {QUERY_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--concept", choices=("case_gender", "query_pronoun"), default="case_gender")
    parser.add_argument("--device", default="cpu", help="for --concept query_pronoun (query embedding)")
    args = parser.parse_args()
    if args.concept == "query_pronoun":
        build_query_pronoun(args.device)
        return

    config = PipelineConfig.load_from_manifest()
    start = time.time()
    print(f"Reading the frozen index {config.index_path} ...")
    frozen = faiss.read_index(config.index_path)
    vectors = frozen.reconstruct_n(0, frozen.ntotal)
    labels = chunk_labels(config.metadata_path)
    assert len(labels) == len(vectors), f"{len(labels)} labels for {len(vectors)} vectors"
    counts = Counter(CLASSES[i] for i in labels)
    print(f"{len(vectors)} vectors x {vectors.shape[1]}; concept classes: {dict(counts)}")

    print("Fitting the LEACE eraser (float64, library defaults) ...")
    params = fit_eraser(vectors, np.eye(len(CLASSES))[labels])

    erased = erase(vectors, params)
    os.makedirs(os.path.dirname(LEACE_INDEX_PATH), exist_ok=True)
    np.savez(LEACE_ERASER_PATH, **params)
    index = faiss.IndexFlatL2(erased.shape[1])
    index.add(erased)
    faiss.write_index(index, LEACE_INDEX_PATH)
    print(f"Wrote {LEACE_INDEX_PATH} and {LEACE_ERASER_PATH} ({time.time() - start:.0f}s)")

    report = {"created": datetime.now().isoformat(timespec="seconds"),
              "method": "LEACE (Belrose et al. 2023), concept-erasure library defaults (method='leace', affine, "
                        "shrinkage); concept = one-hot case gender of each chunk; erased vectors used as they are (not re-normalised)",
              "frozen_index": {"path": config.index_path, "sha256": file_sha256(config.index_path)},
              "concept_classes": dict(counts),
              "eraser_rank": int(np.linalg.matrix_rank(params["proj_left"] @ params["proj_right"])),
              "erased_norm": {"mean": float(np.linalg.norm(erased, axis=1).mean()),
                              "sd": float(np.linalg.norm(erased, axis=1).std())},
              "mean_cosine_original_vs_erased": float(np.mean(np.sum(vectors * erased, axis=1)
                                                              / np.linalg.norm(erased, axis=1)))}

    if not args.skip_probe:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(vectors))
        train_idx, test_idx = perm[:PROBE_TRAIN], perm[PROBE_TRAIN:PROBE_TRAIN + PROBE_TEST]
        report["probe"] = {"design": f"logistic regression, {PROBE_TRAIN} train / {PROBE_TEST} held-out chunks, seed {SEED}",
                           "majority_class_share": float(np.mean(labels[test_idx] == 0))}
        for name, vecs in (("original", vectors), ("erased", erased)):
            print(f"Probing gender on the {name} vectors ...")
            report["probe"][name] = probe(vecs, labels, train_idx, test_idx)
            print(f"  {report['probe'][name]}")

    report["index_variant"] = record_index_variant("leace", file=LEACE_INDEX_PATH, eraser_file=LEACE_ERASER_PATH)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Manifest index_variants.leace updated; report at {REPORT_PATH} ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
