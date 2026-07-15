import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.embeddings.chunk_embedder import ChunkRetriever
from src.llm.config import PipelineConfig

QUERY_FILE = "dataset/eval/retrieval_eval_queries.jsonl"
OUTPUT_DIR = "logs/retrieval_eval"
EVAL_TOP_K_CHUNKS = 100
RECALL_KS = (1, 5, 10)
MRR_CUTOFF = 10
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


# Compute Wilson score confidence interval for binary metrics (recall@k); better than normal approximation for small N or extreme proportions
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


# Bootstrap confidence interval for MRR: resample queries with replacement, compute mean each time, take percentiles
def bootstrap_ci(values, resamples, seed, z_low=2.5, z_high=97.5):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(resamples)
    for i in range(resamples):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return (float(np.percentile(means, z_low)), float(np.percentile(means, z_high)))


# Compute SHA256 hash of query file for traceability: same queries = same hash, ensures reproducibility across runs
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# Aggregate chunks to cases (by mean similarity), then re-sort deterministically by avg_similarity descending + case_id as tie-break
def rank_cases(retriever, chunks):
    cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=len(chunks))
    cases.sort(key=lambda c: (-c["avg_similarity"], c["case_id"]))
    return cases


# Find the rank (1-indexed) of the ground-truth case in a ranked list, or None if not found
def find_rank(case_list, gt_case_id):
    for i, case in enumerate(case_list, start=1):
        if case["case_id"] == gt_case_id:
            return i
    return None


# Evaluate a single query: retrieve to depth 100, compute eval-depth rank, also compute prod-config rank (top_k_chunks → top_k_cases); collect diagnostics
def evaluate_query(retriever, query, config):
    chunks = retriever.retrieve_chunks(query["query_text"], top_k=EVAL_TOP_K_CHUNKS)

    eval_cases = rank_cases(retriever, chunks)
    eval_rank = find_rank(eval_cases, query["source_case_id"])

    prod_chunks = chunks[:config.top_k_chunks]
    prod_cases = rank_cases(retriever, prod_chunks)[:config.top_k_cases]
    prod_rank = find_rank(prod_cases, query["source_case_id"])

    top5_sim = float(np.mean([c["similarity_score"] for c in chunks[:5]]))
    gt_chunk_sims = [c["similarity_score"] for c in chunks if c["case_id"] == query["source_case_id"]]
    gt_top_chunk_sim = float(max(gt_chunk_sims)) if gt_chunk_sims else None

    return {
        "query_id": query["query_id"],
        "source_case_id": query["source_case_id"],
        "eval_rank": eval_rank,
        "prod_rank": prod_rank,
        "mean_top5_chunk_similarity": top5_sim,
        "gt_top_chunk_similarity": gt_top_chunk_sim,
        "gt_chunks_in_pool": len(gt_chunk_sims),
        "top_10_cases": [
            {"case_id": c["case_id"], "avg_similarity": float(c["avg_similarity"]),
             "num_chunks": c["num_chunks"]}
            for c in eval_cases[:10]
        ],
    }


# Main harness: load config and queries, evaluate all 100 queries, compute metrics (recall@k, MRR, CIs), print table, write traceable JSON output
def main():
    config = PipelineConfig.load_from_manifest()
    print(f"Index: {config.index_path}")
    print(f"Metadata: {config.metadata_path}")
    print(f"Production config: top_k_chunks={config.top_k_chunks}, top_k_cases={config.top_k_cases}")
    print(f"Eval retrieval depth: {EVAL_TOP_K_CHUNKS} chunks\n")

    retriever = ChunkRetriever(config.index_path, config.metadata_path)

    queries = []
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    n = len(queries)
    print(f"Loaded {n} queries from {QUERY_FILE}\n")

    start = time.time()
    per_query = []
    for i, query in enumerate(queries, start=1):
        result = evaluate_query(retriever, query, config)
        per_query.append(result)
        if i % 20 == 0:
            print(f"  {i}/{n} queries evaluated ({time.time() - start:.1f}s)")
    elapsed = time.time() - start
    print(f"Done: {n} queries in {elapsed:.1f}s\n")

    eval_ranks = [r["eval_rank"] for r in per_query]
    metrics = {}
    for k in RECALL_KS:
        hits = sum(1 for rank in eval_ranks if rank is not None and rank <= k)
        lo, hi = wilson_ci(hits, n)
        metrics[f"recall@{k}"] = {"value": hits / n, "hits": hits, "n": n,
                                  "wilson_95ci": [lo, hi]}

    rr_values = [1.0 / rank if rank is not None and rank <= MRR_CUTOFF else 0.0
                 for rank in eval_ranks]
    mrr = float(np.mean(rr_values))
    mrr_ci = bootstrap_ci(rr_values, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    metrics[f"mrr@{MRR_CUTOFF}"] = {"value": mrr, "bootstrap_95ci": list(mrr_ci)}

    prod_hits_1 = sum(1 for r in per_query if r["prod_rank"] == 1)
    prod_hits_3 = sum(1 for r in per_query if r["prod_rank"] is not None)
    for label, hits in (("prod_recall@1", prod_hits_1), (f"prod_recall@{config.top_k_cases}", prod_hits_3)):
        lo, hi = wilson_ci(hits, n)
        metrics[label] = {"value": hits / n, "hits": hits, "n": n, "wilson_95ci": [lo, hi]}

    metrics["mean_top5_chunk_similarity"] = float(np.mean(
        [r["mean_top5_chunk_similarity"] for r in per_query]))
    gt_sims = [r["gt_top_chunk_similarity"] for r in per_query
               if r["gt_top_chunk_similarity"] is not None]
    metrics["mean_gt_top_chunk_similarity"] = float(np.mean(gt_sims)) if gt_sims else None
    metrics["queries_with_gt_in_chunk_pool"] = len(gt_sims)

    print("=" * 64)
    print("RETRIEVAL EVALUATION RESULTS (case-level, mean-sim aggregation)")
    print("=" * 64)
    print(f"{'Metric':<28}{'Value':>8}   95% CI")
    print("-" * 64)
    for k in RECALL_KS:
        m = metrics[f"recall@{k}"]
        print(f"{'recall@' + str(k):<28}{m['value']:>8.3f}   "
              f"[{m['wilson_95ci'][0]:.3f}, {m['wilson_95ci'][1]:.3f}]  ({m['hits']}/{m['n']})")
    m = metrics[f"mrr@{MRR_CUTOFF}"]
    print(f"{'MRR@' + str(MRR_CUTOFF):<28}{m['value']:>8.3f}   "
          f"[{m['bootstrap_95ci'][0]:.3f}, {m['bootstrap_95ci'][1]:.3f}]")
    print("-" * 64)
    for label in ("prod_recall@1", f"prod_recall@{config.top_k_cases}"):
        m = metrics[label]
        print(f"{label:<28}{m['value']:>8.3f}   "
              f"[{m['wilson_95ci'][0]:.3f}, {m['wilson_95ci'][1]:.3f}]  ({m['hits']}/{m['n']})")
    print("-" * 64)
    print(f"{'mean top-5 chunk sim':<28}{metrics['mean_top5_chunk_similarity']:>8.3f}")
    if metrics["mean_gt_top_chunk_similarity"] is not None:
        print(f"{'mean gt top-chunk sim':<28}{metrics['mean_gt_top_chunk_similarity']:>8.3f}   "
              f"(gt in {EVAL_TOP_K_CHUNKS}-chunk pool: {metrics['queries_with_gt_in_chunk_pool']}/{n})")

    rank_dist = {}
    for rank in eval_ranks:
        key = str(rank) if rank is not None and rank <= 10 else ">10/miss"
        rank_dist[key] = rank_dist.get(key, 0) + 1
    print("\nRank distribution (eval pass): "
          + ", ".join(f"rank {k}: {v}" for k, v in sorted(
            rank_dist.items(), key=lambda x: (x[0] == '>10/miss', x[0].zfill(3)))))

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "index_path": config.index_path,
            "metadata_path": config.metadata_path,
            "prod_top_k_chunks": config.top_k_chunks,
            "prod_top_k_cases": config.top_k_cases,
            "eval_top_k_chunks": EVAL_TOP_K_CHUNKS,
            "recall_ks": list(RECALL_KS),
            "mrr_cutoff": MRR_CUTOFF,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "aggregation": "mean_similarity, tie-break case_id",
            "query_file": QUERY_FILE,
            "query_file_sha256": file_sha256(QUERY_FILE),
            "n_queries": n,
            "eval_runtime_seconds": round(elapsed, 1),
        },
        "metrics": metrics,
        "per_query": per_query,
    }

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
