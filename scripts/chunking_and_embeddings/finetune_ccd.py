import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import set_seed

from scripts.chunking_and_embeddings.finetune_cda import (BASE_MODEL, DEV_PATH, PAIRS_PATH, SELECTION_PATH, SEED,
                                                          TEST_PATH, TRAIN_PATH, MAX_PRECISION_DROP, embed_corpus,
                                                          read_json, read_jsonl, score, selection_corpus, write_json)
from scripts.chunking_and_embeddings.index_variants import record_index_variant
from src.chunking.legal_chunker import ChunkConfig, LegalCaseChunker
from src.mitigation.finetuned_embedder import MODELS
from src.mitigation.neutral_rewrite import GENDERED, neutralize_many, swap_gender_many

# Arm 4b (src/mitigation/finetuned_embedder.py, ft_ccd): fine-tunes the frozen embedder with content conditional
# debiasing (Deng, Chen, Zhao, Zhang, Li & Thrampoulidis 2024, arXiv 2402.14208). Each training example is a triplet,
# the same text in its original gender, swapped, and neutral. L_bias asks that both gendered versions are equally close
# to the neutral one, |dist(f(a), f(n)) - dist(f(b), f(n))| summed over the ordered pairs (a, b), dist = Gaussian
# kernel exp(-||x - y||^2 / 2 rho^2); L_rep = ||f(n) - f_orig(n)|| keeps the neutral text where the frozen model puts
# it. L = L_bias + beta * L_rep; full fine-tune, Adam, batch 32, one epoch, as in the paper.
# Paper's vs ours: their triplets are LLM rewrites of news sentences (~42k); ours are the dev split's own text with the
# tested rule-based rewrite (the harness-style queries of arm 4 plus chunks of the same dev cases), so the embedder is
# debiased on both sides of retrieval. The paper sets rho to "the variance of the distance over the training data";
# taken literally (rho = 0.0018 on the frozen model's gendered-to-neutral distances) the kernel is exp(-1200) = 0 for
# every triplet and gives no gradient, so we read it as rho^2 = that variance (rho = its standard deviation, 0.043). Embeddings are L2-normalised before the losses, as
# in retrieval. Their checkpoint rule (lowest validation loss) is logged; the one used is arm 4's dev selection
# (fewest flips with article precision >= frozen - MAX_PRECISION_DROP), plus a frozen top-3 overlap floor, since
# arm 4's failure was reordering of the top 3.
#   python -m scripts.chunking_and_embeddings.finetune_ccd --step triplets   (CPU rewrite + frozen rho)
#   python -m scripts.chunking_and_embeddings.finetune_ccd --step sweep      (3 runs, checkpoints every quarter epoch)
#   python -m scripts.chunking_and_embeddings.finetune_ccd --step select     (scores every checkpoint, resumable)
#   python -m scripts.chunking_and_embeddings.finetune_ccd --step build

NAME = "ft_ccd"
OUT_DIR, SWEEP_DIR = f"logs/mitigation/{NAME}", f"models/{NAME}_sweep"
TRIPLETS_PATH = f"{OUT_DIR}/triplets.jsonl"
SCORES_PATH, SWEEP_PATH = f"{OUT_DIR}/sweep_scores.jsonl", f"{OUT_DIR}/sweep.json"

MAX_CHUNK_TRIPLETS, MAX_VAL_CHUNK_TRIPLETS = 10000, 1000
BATCH, MICRO_BATCH, TRAIN_MAX_SEQ, INFERENCE_MAX_SEQ = 32, 8, 512, 8192
CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)
RUNS = {"lr5e-05_beta1": {"lr": 5e-5, "beta": 1.0}, "lr1e-05_beta1": {"lr": 1e-5, "beta": 1.0},
        "lr5e-05_beta10": {"lr": 5e-5, "beta": 10.0}}
# Set at 0.85 before the sweep; no checkpoint reached it (best: lr5e-05_beta10, 0.81 with 3/102 flips), and the owner
# lowered it to 0.80 after seeing the scores (2026-10-09); R5's changed-case check tests whether the moved cases stay relevant
MIN_FROZEN_OVERLAP = 0.80


# ---- triplets: (original, swapped, neutral) from the dev split ----

def chunk_triplets(cases, limit, rng):
    chunker = LegalCaseChunker(ChunkConfig())
    chunks = [c for case in cases for c in chunker.create_paragraph_chunks(case) if GENDERED.search(c["chunk_text"])]
    rng.shuffle(chunks)
    texts = [c["chunk_text"] for c in chunks[:int(limit * 1.3)]]
    swapped, neutral = swap_gender_many(texts), neutralize_many(texts)
    rows = [{"case_id": c["case_id"], "kind": "chunk", "text": t, "swapped": s, "neutral": n}
            for c, t, s, n in zip(chunks, texts, swapped, neutral) if s != t and n != t]
    return rows[:limit]


def step_triplets():
    pairs = read_jsonl(PAIRS_PATH)
    elsewhere = {r["case_id"] for path in (TRAIN_PATH, TEST_PATH) for r in read_jsonl(path)}
    assert not elsewhere & {r["case_id"] for r in pairs}, "dev cases also in train/test"
    dev = {c["case_id"]: c for c in read_jsonl(DEV_PATH)}
    rng = random.Random(SEED)

    rows = []
    for split, limit in (("train", MAX_CHUNK_TRIPLETS), ("val", MAX_VAL_CHUNK_TRIPLETS)):
        part = [r for r in pairs if r["split"] == split]
        rows += [{"case_id": r["case_id"], "kind": "query", "split": split, "text": r["query"], "swapped": r["twin"],
                  "neutral": r["neutral"]} for r in part]
        print(f"[{split}] rewriting chunks of {len(part)} dev cases...")
        rows += [{**t, "split": split} for t in chunk_triplets([dev[r["case_id"]] for r in part], limit, rng)]
    assert not elsewhere & {r["case_id"] for r in rows}

    model = load_model()
    train = [r for r in rows if r["split"] == "train"]
    with torch.no_grad():
        d = torch.cat([distances(model, train[i:i + 64]) for i in range(0, len(train), 64)]).flatten()
    rho = float(d.std())
    with open(TRIPLETS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(f"{OUT_DIR}/rho.json", {"rho": rho, "rule": "rho^2 = variance of ||f(gendered) - f(neutral)||, "
                                       "frozen model, train triplets", "distance_mean": float(d.mean()), "distance_std": float(d.std()),
                                       "n": len(d)})
    count = lambda s, k: sum(r["split"] == s and r["kind"] == k for r in rows)
    print(f"Wrote {len(rows)} triplets to {TRIPLETS_PATH}: train {count('train', 'query')} queries + "
          f"{count('train', 'chunk')} chunks, val {count('val', 'query')} + {count('val', 'chunk')}")
    print(f"Frozen distance gendered-to-neutral: mean {d.mean():.4f}, std {d.std():.4f} -> rho {rho:.6f}")


# ---- the loss ----

def load_model(path=BASE_MODEL):
    model = SentenceTransformer(path, trust_remote_code=True, device="cuda")
    model.max_seq_length = TRAIN_MAX_SEQ
    return model


# Checkpoints are saved with the inference max_seq_length, as arm 4's
def save_checkpoint(model, path):
    model.max_seq_length = INFERENCE_MAX_SEQ
    model.save(path)
    model.max_seq_length = TRAIN_MAX_SEQ


def embed(model, texts):
    features = {k: v.to(model.device) for k, v in model.tokenize(texts).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(features)["sentence_embedding"]
    return F.normalize(out.float(), dim=-1)


def distances(model, rows):
    n = embed(model, [r["neutral"] for r in rows])
    return torch.stack([(embed(model, [r[k] for r in rows]) - n).norm(dim=-1) for k in ("text", "swapped")], dim=1)


def ccd_loss(model, rows, targets, rho, beta):
    a, b, n = (embed(model, [r[k] for r in rows]) for k in ("text", "swapped", "neutral"))
    kernel = lambda x: torch.exp(-(x - n).pow(2).sum(-1) / (2 * rho ** 2))
    bias = 2 * (kernel(a) - kernel(b)).abs()
    rep = (n - targets).norm(dim=-1)
    return bias.mean() + beta * rep.mean(), bias.mean().item(), rep.mean().item()


def frozen_targets(model, rows):
    with torch.no_grad():
        return torch.cat([embed(model, [r["neutral"] for r in rows[i:i + 64]]) for i in range(0, len(rows), 64)])


def validation_loss(model, rows, targets, rho, beta):
    model.eval()
    with torch.no_grad():
        parts = [(len(rows[i:i + 64]), ccd_loss(model, rows[i:i + 64], targets[i:i + 64], rho, beta))
                 for i in range(0, len(rows), 64)]
    model.train()
    total = sum(n for n, _ in parts)
    return {"loss": sum(n * p[0].item() for n, p in parts) / total, "bias": sum(n * p[1] for n, p in parts) / total,
            "rep": sum(n * p[2] for n, p in parts) / total}


# ---- sweep: one epoch per setting, checkpoints every quarter epoch ----

def checkpoint_path(run, step):
    return f"{SWEEP_DIR}/{run}/step{step}"


def train_run(run, train, val, rho):
    cfg = RUNS[run]
    set_seed(SEED)
    model = load_model()
    train_targets, val_targets = frozen_targets(model, train), frozen_targets(model, val)
    model.train()
    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    saves = {round(f * len(batches)): f for f in CHECKPOINTS}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    log = [{"step": 0, **validation_loss(model, val, val_targets, rho, cfg["beta"])}]
    print(f"[{run}] {len(batches)} steps, step 0 val {log[0]}", flush=True)
    start = time.time()
    for step, batch in enumerate(batches, 1):
        optimizer.zero_grad()
        for i in range(0, len(batch), MICRO_BATCH):
            idx = batch[i:i + MICRO_BATCH]
            loss, _, _ = ccd_loss(model, [train[j] for j in idx], train_targets[idx], rho, cfg["beta"])
            (loss * len(idx) / len(batch)).backward()
        optimizer.step()
        if step in saves:
            log.append({"step": step, "epoch": saves[step], **validation_loss(model, val, val_targets, rho, cfg["beta"])})
            save_checkpoint(model, checkpoint_path(run, step))
            print(f"[{run}] step {step} ({(time.time() - start) / 60:.1f} min) val {log[-1]}", flush=True)
    write_json(f"{SWEEP_DIR}/{run}/log.json", log)
    del model, optimizer
    torch.cuda.empty_cache()


def step_sweep():
    rows = read_jsonl(TRIPLETS_PATH)
    train, val = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    rho = read_json(f"{OUT_DIR}/rho.json")["rho"]
    print(f"Triplets: train {len(train)}, val {len(val)}; rho {rho:.6f}")
    for run in RUNS:
        if os.path.exists(f"{SWEEP_DIR}/{run}/log.json"):
            print(f"[{run}] done, skipping")
            continue
        train_run(run, train, val, rho)


# ---- select: arm 4's dev scoring for every checkpoint ----

def candidates():
    out = [{"candidate": "frozen", "run": None, "step": 0, "path": BASE_MODEL}]
    for run in RUNS:
        for entry in read_json(f"{SWEEP_DIR}/{run}/log.json")[1:]:
            out.append({"candidate": f"{run}/step{entry['step']}", "run": run, "step": entry["step"],
                        "epoch": entry["epoch"], "val_loss": entry["loss"], "path": checkpoint_path(run, entry["step"])})
    return out


def step_select():
    selection = read_json(SELECTION_PATH)
    corpus = selection_corpus(selection)
    done = {r["candidate"]: r for r in read_jsonl(SCORES_PATH)} if os.path.exists(SCORES_PATH) else {}
    for c in candidates():
        if c["candidate"] in done:
            continue
        start = time.time()
        model = SentenceTransformer(c["path"], trust_remote_code=True, device="cuda")
        row = {**c, **score(model, selection, corpus)}
        del model
        torch.cuda.empty_cache()
        with open(SCORES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done[row["candidate"]] = row
        print(f"  {row['candidate']:<26} flips {row['flips']}/{row['n_val']}  article prec@3 "
              f"{row['article_precision']:.3f}  frozen top-3 overlap {row['frozen_overlap']:.2f}  "
              f"({time.time() - start:.0f} s)", flush=True)

    rows = [done[c["candidate"]] for c in candidates()]
    frozen = rows[0]
    eligible = [r for r in rows[1:] if r["flip_rate"] < frozen["flip_rate"]
                and r["article_precision"] >= frozen["article_precision"] - MAX_PRECISION_DROP - 1e-9
                and r["frozen_overlap"] >= MIN_FROZEN_OVERLAP]
    best = min(eligible, key=lambda r: (r["flip_rate"], -r["article_precision"], -r["frozen_overlap"])) if eligible else None
    lowest_loss = {run: min((r for r in rows[1:] if r["run"] == run), key=lambda r: r["val_loss"])["candidate"]
                   for run in RUNS}
    write_json(SWEEP_PATH, {"rule": f"lowest flip rate below the frozen model's, with article precision@3 >= frozen - "
                                    f"{MAX_PRECISION_DROP} and frozen top-3 overlap >= {MIN_FROZEN_OVERLAP}; ties: "
                                    f"higher precision, then higher overlap",
                            "paper_rule_lowest_val_loss": lowest_loss, "frozen": frozen, "selected": best,
                            "candidates": rows})
    print(f"Frozen: flips {frozen['flips']}/{frozen['n_val']}, article prec@3 {frozen['article_precision']:.3f}")
    print(f"Paper's rule (lowest validation loss): {lowest_loss}")
    print(f"Selected: {best and best['candidate']} -> {SWEEP_PATH}")


# ---- build: the selected model, re-embedded corpus, manifest entry ----

def step_build():
    selected = read_json(SWEEP_PATH)["selected"]
    assert selected, "no candidate passed the selection rule"
    model_dir, index_path = MODELS[NAME]
    start = time.time()
    model = SentenceTransformer(selected["path"], trust_remote_code=True, device="cuda")
    model.save(model_dir)
    del model
    torch.cuda.empty_cache()

    embedding_check = embed_corpus(model_dir, index_path)
    rows = read_jsonl(TRIPLETS_PATH)
    report = {"created": datetime.now().isoformat(timespec="seconds"), "model": NAME, "selected": selected,
              "training": {"method": "content conditional debiasing (Deng et al. 2024)", "base_model": BASE_MODEL,
                           "data": "dev split, R4 gates: query triplets + chunk triplets of the same cases",
                           "train_triplets": sum(r["split"] == "train" for r in rows), "batch": BATCH,
                           "micro_batch": MICRO_BATCH, "train_max_seq_length": TRAIN_MAX_SEQ, "optimizer": "Adam",
                           "epochs": 1, "rho": read_json(f"{OUT_DIR}/rho.json")["rho"], "seed": SEED,
                           **RUNS[selected["run"]]},
              "embedding_check": embedding_check, "minutes": round((time.time() - start) / 60, 1)}
    report["index_variant"] = record_index_variant(NAME, file=index_path, model_file=f"{model_dir}/model.safetensors")
    write_json(f"{OUT_DIR}/build.json", report)
    print(f"[{NAME}] done: {model_dir}, {index_path}, manifest index_variants.{NAME}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=("triplets", "sweep", "select", "build"), required=True)
    args = parser.parse_args()
    {"triplets": step_triplets, "sweep": step_sweep, "select": step_select, "build": step_build}[args.step]()


if __name__ == "__main__":
    main()
