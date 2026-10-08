import argparse
import json
import os
import random
import time
from datetime import datetime

import faiss
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback, set_seed

from scripts.chunking_and_embeddings.index_variants import record_index_variant
from scripts.dataset.generate_eval_queries import MIN_FACTS_WORDS, build_query
from scripts.dataset.variant_set_construction.select_variant_bases import INTRINSIC, SENSITIVE
from scripts.eval.run_swap_set_test import MAX_WORDS, dominant_gender
from src.chunking.legal_chunker import ChunkConfig, LegalCaseChunker
from src.embeddings.chunk_embedder import ChunkEmbedder, ChunkRetriever
from src.llm.config import PipelineConfig
from src.mitigation.finetuned_embedder import MODELS
from src.mitigation.neutral_rewrite import neutralize_many, swap_gender_many

# Arm 4 (src/mitigation/finetuned_embedder.py): fine-tunes the frozen embedder with label-preserving counterfactual
# data augmentation, following the training and selection recipe of Kim et al. (ACL Findings 2025): in-batch
# contrastive loss, AdamW, full fine-tuning merged with the base model (WiSE-FT) or only the last layers (PEFT),
# a sweep of embedders, and the least biased one whose utility drops at most MAX_RECALL_DROP is kept.
# Data: the dev split only (never indexed; the audit, R4 and harness cases are all train/test cases), with the R4 gates
# (one dominant applicant gender, not gender-intrinsic or sensitive, <= 3000 words). Query = harness-style query of the
# case; twin = swap_gender() of it; both share one positive, the case's chunk the frozen model ranks highest for the
# neutral rewrite of the query, read at the training length (TRAIN_MAX_SEQ tokens) so the positive matches what the
# model sees in training.
# Selection uses dev validation twins (top-1 flips) and 100 harness-style queries from train cases outside the B.1
# harness (recall@1), on a small index; the audit, R4 and harness sets are only used after selection.
#   python -m scripts.chunking_and_embeddings.finetune_cda --step pairs     (CPU rewrite + frozen GPU retrieval)
#   python -m scripts.chunking_and_embeddings.finetune_cda --step sweep     (5 training runs, resumable)
#   python -m scripts.chunking_and_embeddings.finetune_cda --step select    (scores every candidate, resumable)
#   python -m scripts.chunking_and_embeddings.finetune_cda --step build --model cda|ft_control

BASE_MODEL = "nomic-ai/nomic-embed-text-v1"
DEV_PATH, TRAIN_PATH, TEST_PATH = "dataset/dev.jsonl", "dataset/train.jsonl", "dataset/test.jsonl"
HARNESS_PATH = "dataset/eval/retrieval_harness_queries.jsonl"
OUT_DIR, SWEEP_DIR = "logs/mitigation/cda", "models/cda_sweep"
PAIRS_PATH, SELECTION_PATH = f"{OUT_DIR}/pairs.jsonl", f"{OUT_DIR}/selection_set.json"
SCORES_PATH, SWEEP_PATH = f"{OUT_DIR}/sweep_scores.jsonl", f"{OUT_DIR}/sweep.json"

SEED, VAL_SHARE, UTILITY_QUERIES = 42, 0.2, 100
SELECTION_TOP_CASES, SELECTION_CHUNKS = 5, 15000
# Cached (GradCache) form of the in-batch contrastive loss: the same loss and gradients in mini-batches of
# MINI_BATCH, so a batch of 16 at 1024 tokens fits the 12 GB GPU (uncached: 12 GB, spills, 4x slower)
BATCH, MINI_BATCH, TRAIN_MAX_SEQ, SCALE, WEIGHT_DECAY = 16, 4, 1024, 50.0, 0.01
EPOCHS, WISE = (5, 10, 15), (0.3, 0.5, 0.7, 1.0)
RUNS = {"full_lr1e-05": {"lr": 1e-5, "layers": None}, "full_lr3e-05": {"lr": 3e-5, "layers": None},
        "peft_l1": {"lr": 3e-5, "layers": 1}, "peft_l2": {"lr": 3e-5, "layers": 2}, "peft_l4": {"lr": 3e-5, "layers": 4}}
MAX_RECALL_DROP = 0.02


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# As ChunkRetriever: pins nomic's stateful rotary scaling before queries are embedded. Corpus chunks are embedded
# before it, as in the frozen build; training (TRAIN_MAX_SEQ tokens) never reaches the scaling.
def warm_up(model):
    model.encode(["legal " * (model.max_seq_length + 1000)], normalize_embeddings=True)


def encode_queries(model, queries):
    return np.vstack([model.encode([q], convert_to_numpy=True, normalize_embeddings=True) for q in queries])


# ---- pairs: dev twins with their positive, the validation split, the selection index ----

def step_pairs():
    dev = read_jsonl(DEV_PATH)
    elsewhere = {r["case_id"] for path in (TRAIN_PATH, TEST_PATH) for r in read_jsonl(path)}
    assert not elsewhere & {c["case_id"] for c in dev}, "dev cases also in train/test"

    cases = []
    for case in dev:
        gender, _ = dominant_gender(case["facts"])
        text = " ".join(case["facts"])
        query, _ = build_query(case)
        if gender and not (INTRINSIC.search(text) or SENSITIVE.search(text)) and len(query.split()) <= MAX_WORDS:
            cases.append({"case": case, "gender": gender, "query": query})
    print(f"Eligible dev cases: {len(cases)} (Male {sum(c['gender'] == 'Male' for c in cases)}, "
          f"Female {sum(c['gender'] == 'Female' for c in cases)})")

    print("Swapping and neutralising the queries...")
    queries = [c["query"] for c in cases]
    twins, neutral = swap_gender_many(queries), neutralize_many(queries)

    chunker = LegalCaseChunker(ChunkConfig())
    chunks = [chunker.create_paragraph_chunks(c["case"]) for c in cases]
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device="cuda")
    model.max_seq_length = TRAIN_MAX_SEQ
    flat = [ch["chunk_text"] for case_chunks in chunks for ch in case_chunks]
    vectors = model.encode(flat, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    neutral_vectors = encode_queries(model, neutral)

    rng = random.Random(SEED)
    val_ids = set()
    for gender in ("Male", "Female"):
        ids = sorted(c["case"]["case_id"] for c in cases if c["gender"] == gender)
        val_ids |= set(rng.sample(ids, round(VAL_SHARE * len(ids))))

    rows, start = [], 0
    for c, case_chunks, twin, neu, nv in zip(cases, chunks, twins, neutral, neutral_vectors):
        block = vectors[start:start + len(case_chunks)]
        start += len(case_chunks)
        best = int(np.argmax(block @ nv))
        rows.append({"case_id": c["case"]["case_id"], "gender": c["gender"],
                     "split": "val" if c["case"]["case_id"] in val_ids else "train",
                     "query": c["query"], "twin": twin, "neutral": neu,
                     "positive_chunk_id": case_chunks[best]["chunk_id"], "positive": case_chunks[best]["chunk_text"]})
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()
    print(f"Wrote {len(rows)} twins to {PAIRS_PATH} "
          f"(train {sum(r['split'] == 'train' for r in rows)}, val {sum(r['split'] == 'val' for r in rows)})")
    build_selection_set(rows)


def build_selection_set(rows):
    config = PipelineConfig.load_from_manifest()
    harness = {q["source_case_id"] for q in read_jsonl(HARNESS_PATH)}
    train = read_jsonl(TRAIN_PATH)
    pool = sorted((c for c in train if c["case_id"] not in harness
                   and sum(len(p.split()) for p in c["facts"]) >= MIN_FACTS_WORDS), key=lambda c: c["case_id"])
    utility = [{"case_id": c["case_id"], "query": build_query(c)[0]} for c in random.Random(SEED).sample(pool, UTILITY_QUERIES)]

    retriever = ChunkRetriever(config.index_path, config.metadata_path)
    keep = {u["case_id"] for u in utility}
    val = [r for r in rows if r["split"] == "val"]
    for r in val:
        for q in (r["query"], r["twin"]):
            ranked = list(dict.fromkeys(ch["case_id"] for ch in retriever.retrieve_chunks(q, top_k=200)))
            keep |= set(ranked[:SELECTION_TOP_CASES])

    by_case = {}
    for meta in retriever.metadata:
        by_case.setdefault(meta["case_id"], []).append(meta["chunk_id"])
    rest = sorted(set(by_case) - keep)
    random.Random(SEED).shuffle(rest)
    n_chunks = sum(len(by_case[c]) for c in keep)
    for case_id in rest:
        if n_chunks >= SELECTION_CHUNKS:
            break
        keep.add(case_id)
        n_chunks += len(by_case[case_id])
    chunk_ids = [cid for case_id in sorted(keep) for cid in by_case[case_id]]
    write_json(SELECTION_PATH, {"val": [{k: r[k] for k in ("case_id", "gender", "query", "twin")} for r in val],
                                "utility": utility, "cases": len(keep), "chunk_ids": chunk_ids})
    print(f"Selection set: {len(val)} validation twins, {len(utility)} utility queries, "
          f"{len(keep)} cases / {len(chunk_ids)} chunks -> {SELECTION_PATH}")


# ---- sweep: training runs, checkpoints at EPOCHS ----

def layer_numbers(model):
    return sorted({int(n.split("encoder.layers.")[1].split(".")[0])
                   for n, _ in model.named_parameters() if "encoder.layers." in n})


# PEFT as in Kim et al.: only the last `layers` transformer layers are trained; None = full fine-tuning
def set_trainable(model, layers):
    last = layer_numbers(model)[-layers:] if layers else None
    for name, p in model.named_parameters():
        p.requires_grad = last is None or any(f"encoder.layers.{i}." in name for i in last)


# With a constant learning rate, the epoch-5 and epoch-10 checkpoints of one 15-epoch run equal separate 5- and
# 10-epoch runs, so one run per configuration covers Kim et al.'s epoch grid
class SaveAtEpochs(TrainerCallback):
    def __init__(self, model, out_dir, epochs, layers, max_seq_length):
        self.model, self.out_dir, self.epochs = model, out_dir, epochs
        self.layers, self.max_seq_length = layers, max_seq_length

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch in self.epochs:
            save_checkpoint(self.model, f"{self.out_dir}/epoch{epoch}", self.layers, self.max_seq_length)


# PEFT checkpoints keep only the trained layers; full ones are saved with the inference max_seq_length
def save_checkpoint(model, path, layers, max_seq_length):
    os.makedirs(path, exist_ok=True)
    if layers:
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, f"{path}/layers.pt")
        return
    train_len, model.max_seq_length = model.max_seq_length, max_seq_length
    model.save(path)
    model.max_seq_length = train_len


def checkpoint_path(run, epoch):
    return f"{SWEEP_DIR}/{run}/epoch{epoch}"


def train_run(run, pairs, out_dir, epochs):
    cfg = RUNS[run]
    set_seed(SEED)
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device="cuda")
    base_len, model.max_seq_length = model.max_seq_length, TRAIN_MAX_SEQ
    set_trainable(model, cfg["layers"])
    data = Dataset.from_dict({"anchor": [a for a, _ in pairs], "positive": [p for _, p in pairs]})
    loss = CachedMultipleNegativesRankingLoss(model, scale=SCALE, mini_batch_size=MINI_BATCH)
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{out_dir}/trainer", num_train_epochs=max(epochs), per_device_train_batch_size=BATCH,
        learning_rate=cfg["lr"], weight_decay=WEIGHT_DECAY, lr_scheduler_type="constant", warmup_ratio=0.0,
        bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, seed=SEED, save_strategy="no", logging_steps=20,
        report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=loss,
                                         callbacks=[SaveAtEpochs(model, out_dir, set(epochs), cfg["layers"], base_len)])
    start = time.time()
    trainer.train()
    print(f"[{run}] trained in {(time.time() - start) / 60:.1f} min -> {out_dir}")
    del trainer, model
    torch.cuda.empty_cache()


def training_pairs(twins=True):
    rows = [r for r in read_jsonl(PAIRS_PATH) if r["split"] == "train"]
    pairs = [(r["query"], r["positive"]) for r in rows]
    return pairs + [(r["twin"], r["positive"]) for r in rows] if twins else pairs


def step_sweep():
    pairs = training_pairs()
    print(f"Training pairs: {len(pairs)} (queries + twins)")
    for run in RUNS:
        if all(os.path.exists(checkpoint_path(run, e)) for e in EPOCHS):
            print(f"[{run}] done, skipping")
            continue
        train_run(run, pairs, f"{SWEEP_DIR}/{run}", EPOCHS)


# ---- select: score every candidate on the dev selection set ----

def candidates():
    out = [{"run": None, "epoch": 0, "wise": 0.0}]
    for run, cfg in RUNS.items():
        for epoch in EPOCHS:
            for lam in (WISE if cfg["layers"] is None else (1.0,)):
                out.append({"run": run, "epoch": epoch, "wise": lam})
    return out


def key(c):
    return "frozen" if c["run"] is None else f"{c['run']}/epoch{c['epoch']}/wise{c['wise']}"


# A candidate = base weights moved toward a checkpoint by WiSE-FT, (1 - wise) * base + wise * tuned (wise 1 = the checkpoint)
def load_candidate(c, path=None):
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device="cuda")
    if c["run"] is None:
        return model
    path = path or checkpoint_path(c["run"], c["epoch"])
    if RUNS[c["run"]]["layers"]:
        tuned = torch.load(f"{path}/layers.pt")
    else:
        tuned = SentenceTransformer(path, trust_remote_code=True, device="cpu").state_dict()
    state = model.state_dict()
    for name, value in tuned.items():
        if torch.is_floating_point(state[name]):
            state[name] = (1 - c["wise"]) * state[name] + c["wise"] * value.to(state[name].device, state[name].dtype)
    model.load_state_dict(state)
    return model


def score(model, selection, texts, case_of):
    # Chunks first on the fresh model (as the corpus build), then the warm-up and the queries (as the retriever)
    index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
    index.add(model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True).astype("float32"))
    warm_up(model)

    def top_case(queries):
        _, idx = index.search(encode_queries(model, queries).astype("float32"), 1)
        return [case_of[i[0]] for i in idx]

    val = selection["val"]
    original, twin = top_case([v["query"] for v in val]), top_case([v["twin"] for v in val])
    utility = top_case([u["query"] for u in selection["utility"]])
    flips = sum(a != b for a, b in zip(original, twin))
    hits = sum(t == u["case_id"] for t, u in zip(utility, selection["utility"]))
    return {"flips": flips, "n_val": len(val), "flip_rate": flips / len(val),
            "recall_at_1": hits / len(selection["utility"])}


def selection_corpus(selection):
    config = PipelineConfig.load_from_manifest()
    wanted = set(selection["chunk_ids"])
    chunks = [c for c in read_jsonl(config.chunk_text_path) if c["chunk_id"] in wanted]
    return [c["chunk_text"] for c in chunks], [c["case_id"] for c in chunks]


def step_select():
    selection = read_json(SELECTION_PATH)
    texts, case_of = selection_corpus(selection)
    done = {r["candidate"]: r for r in read_jsonl(SCORES_PATH)} if os.path.exists(SCORES_PATH) else {}
    for c in candidates():
        if key(c) in done:
            continue
        start = time.time()
        model = load_candidate(c)
        row = {"candidate": key(c), **c, **score(model, selection, texts, case_of)}
        del model
        torch.cuda.empty_cache()
        with open(SCORES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done[row["candidate"]] = row
        print(f"  {row['candidate']:<32} flips {row['flips']}/{row['n_val']}  recall@1 {row['recall_at_1']:.2f}  "
              f"({time.time() - start:.0f} s)", flush=True)

    rows = [done[key(c)] for c in candidates()]
    frozen = rows[0]
    eligible = [r for r in rows[1:] if r["recall_at_1"] >= frozen["recall_at_1"] - MAX_RECALL_DROP - 1e-9
                and r["flip_rate"] < frozen["flip_rate"]]
    best = (min(eligible, key=lambda r: (r["flip_rate"], -r["recall_at_1"], RUNS[r["run"]]["layers"] or float("inf")))
            if eligible else None)
    write_json(SWEEP_PATH, {"rule": f"lowest flip rate below the frozen model's, with recall@1 >= frozen - "
                                    f"{MAX_RECALL_DROP}; ties: higher recall, then fewer trained layers",
                            "frozen": frozen, "selected": best, "candidates": rows})
    print(f"Frozen: flips {frozen['flips']}/{frozen['n_val']}, recall@1 {frozen['recall_at_1']:.2f}")
    print(f"Selected: {best and best['candidate']} -> {SWEEP_PATH}")


# ---- build: the selected CDA model / the control, re-embedded corpus, manifest entry ----

def step_build(name):
    selected = read_json(SWEEP_PATH)["selected"]
    assert selected, "no candidate passed the selection rule"
    model_dir, index_path = MODELS[name]
    start = time.time()
    if name == "cda":
        model = load_candidate(selected)
    else:
        control = f"{SWEEP_DIR}/control_{selected['run']}"
        if not os.path.exists(f"{control}/epoch{selected['epoch']}"):
            train_run(selected["run"], training_pairs(twins=False), control, (selected["epoch"],))
        model = load_candidate(selected, path=f"{control}/epoch{selected['epoch']}")
    model.save(model_dir)
    del model
    torch.cuda.empty_cache()

    config = PipelineConfig.load_from_manifest()
    texts = [c["chunk_text"] for c in read_jsonl(config.chunk_text_path)]
    embedder = ChunkEmbedder(model_name=model_dir, device="cuda", batch_size=32)
    vectors = embedder.create_embeddings(texts)
    faiss.write_index(embedder.build_faiss_index(vectors), index_path)
    check = ChunkEmbedder(model_name=model_dir, device="cuda", batch_size=32).create_embeddings(texts[:2000])
    cosine = np.sum(check * vectors[:2000], axis=1)

    report = {"created": datetime.now().isoformat(timespec="seconds"), "model": name, "selected": selected,
              "training": {"base_model": BASE_MODEL, "data": "dev split, R4 gates", "twins": name == "cda",
                           "pairs": len(training_pairs(twins=name == "cda")), "batch": BATCH,
                           "train_max_seq_length": TRAIN_MAX_SEQ, "loss": "CachedMultipleNegativesRankingLoss",
                           "mini_batch": MINI_BATCH,
                           "scale": SCALE, "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
                           "lr_schedule": "constant", "seed": SEED, **RUNS[selected["run"]]},
              "embedding_check": {"sample": 2000, "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean())},
              "minutes": round((time.time() - start) / 60, 1)}
    report["index_variant"] = record_index_variant(name, file=index_path, model_file=f"{model_dir}/model.safetensors")
    write_json(f"logs/mitigation/{name}/build.json", report)
    print(f"[{name}] done: {model_dir}, {index_path}, manifest index_variants.{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=("pairs", "sweep", "select", "build"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), default="cda", help="for --step build")
    args = parser.parse_args()
    if args.step == "pairs":
        step_pairs()
    elif args.step == "sweep":
        step_sweep()
    elif args.step == "select":
        step_select()
    else:
        step_build(args.model)


if __name__ == "__main__":
    main()
