import argparse
import json
import os
import time
from datetime import datetime

import faiss
import numpy as np

from scripts.chunking_and_embeddings.index_variants import record_index_variant
from src.chunking.legal_chunker import ChunkConfig, LegalCaseChunker
from src.embeddings.chunk_embedder import ChunkEmbedder
from src.llm.config import PipelineConfig
from src.mitigation.blind_index import (BLIND_CHUNKS_PATH, BLIND_INDEX_PATH, FACTS_ONLY_CHUNKS_PATH,
                                        FACTS_ONLY_INDEX_PATH, blind_chunks, facts_only_chunks)
from src.mitigation.neutral_rewrite import GENDERED

# Builds the arm 2 corpora (src/mitigation/blind_index.py): the chunk file, then the index, embedded exactly as the
# frozen one (ChunkEmbedder, same model, batch size and file order). The rewrite is resumable in batches.
# A sample of frozen chunks is re-embedded the same way and compared with the stored vectors, so any drift between
# this embedding run and the frozen build is measured. The frozen index and chunk file are only read.
#   python -m scripts.chunking_and_embeddings.build_blind_index --variant blind_index   (~1-1.5 h CPU + ~30 min GPU)
#   python -m scripts.chunking_and_embeddings.build_blind_index --variant facts_only    (~minutes CPU + ~30 min GPU)

TRAIN_PATH = "dataset/train.jsonl"
REWRITE_BATCH = 5000
CHECK_SAMPLE = 2000
VARIANTS = {"blind_index": (BLIND_CHUNKS_PATH, BLIND_INDEX_PATH),
            "facts_only": (FACTS_ONLY_CHUNKS_PATH, FACTS_ONLY_INDEX_PATH)}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_blind_chunks(frozen_path, out_path, n_process):
    chunks = read_jsonl(frozen_path)
    done = len(read_jsonl(out_path)) if os.path.exists(out_path) else 0
    start = time.time()
    for i in range(done, len(chunks), REWRITE_BATCH):
        append_jsonl(out_path, blind_chunks(chunks[i:i + REWRITE_BATCH], n_process=n_process))
        n = min(i + REWRITE_BATCH, len(chunks))
        rate = (n - done) / (time.time() - start)
        print(f"  rewritten {n}/{len(chunks)} ({rate:.0f}/s, ~{(len(chunks) - n) / rate / 60:.0f} min left)", flush=True)


def write_facts_only_chunks(out_path):
    chunks = facts_only_chunks(read_jsonl(TRAIN_PATH), LegalCaseChunker(ChunkConfig()))
    if os.path.exists(out_path):
        os.remove(out_path)
    append_jsonl(out_path, chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--n-process", type=int, default=1, help="spaCy parser processes for the rewrite")
    parser.add_argument("--rewrite-only", action="store_true", help="Write the chunk file and stop before the GPU embedding")
    args = parser.parse_args()

    config = PipelineConfig.load_from_manifest()
    chunks_path, index_path = VARIANTS[args.variant]
    report_path = f"logs/mitigation/{args.variant}/build.json"
    start = time.time()

    print(f"[{args.variant}] writing {chunks_path} ...")
    if args.variant == "blind_index":
        write_blind_chunks(config.chunk_text_path, chunks_path, args.n_process)
    else:
        write_facts_only_chunks(chunks_path)
    if args.rewrite_only:
        print(f"[{args.variant}] chunk file written; embed later without --rewrite-only")
        return
    chunks = read_jsonl(chunks_path)
    texts = [c["chunk_text"] for c in chunks]

    frozen_chunks = read_jsonl(config.chunk_text_path)
    embedder = ChunkEmbedder(model_name="nomic-ai/nomic-embed-text-v1", device="cuda", batch_size=32)
    frozen = faiss.read_index(config.index_path)
    sample = frozen_chunks[:CHECK_SAMPLE]
    check = np.sum(embedder.create_embeddings([c["chunk_text"] for c in sample]) * frozen.reconstruct_n(0, CHECK_SAMPLE), axis=1)
    print(f"Re-embedding check on {CHECK_SAMPLE} frozen chunks: cosine to stored vectors min {check.min():.6f}, mean {check.mean():.6f}")

    index = embedder.build_faiss_index(embedder.create_embeddings(texts))
    faiss.write_index(index, index_path)

    report = {"created": datetime.now().isoformat(timespec="seconds"), "variant": args.variant,
              "chunks": len(chunks), "cases": len({c["case_id"] for c in chunks}),
              "embedding_check": {"sample": CHECK_SAMPLE, "cosine_min": float(check.min()), "cosine_mean": float(check.mean())},
              "minutes": round((time.time() - start) / 60, 1)}
    if args.variant == "blind_index":
        report["chunks_rewritten"] = sum(t != f["chunk_text"] for t, f in zip(texts, frozen_chunks))
        report["gendered_words_left"] = sum(len(GENDERED.findall(t)) for t in texts)
    report["index_variant"] = record_index_variant(args.variant, file=index_path, chunks_file=chunks_path)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{args.variant}] done: {index_path}, manifest index_variants.{args.variant}, report {report_path}")


if __name__ == "__main__":
    main()
