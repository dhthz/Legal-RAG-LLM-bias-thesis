import json
import sys

from src.mitigation import MitigationArm
from src.rag.pipeline import RAGPipeline

# Tripwire for Phase F: with no arms (and with a no-op arm, which exercises every hook), retrieval must
# reproduce the frozen audit run exactly
REFERENCE_LOG = "logs/bias_audit/stateless/run40_A.jsonl"


def retrieval_signature(cases):
    return [(c["case_id"], [(ch["chunk_id"], round(float(ch["similarity_score"]), 6)) for ch in c["chunks"]])
            for c in cases]


def check(pipeline, reference, label):
    mismatches = [row["query_id"] for row in reference
                  if retrieval_signature(pipeline.retrieve(row["query"])[1]) != retrieval_signature(row["retrieved_cases"])]
    print(f"{label}: {len(reference) - len(mismatches)}/{len(reference)} identical")
    if mismatches:
        print(f"  first mismatches: {mismatches[:10]}")
    return not mismatches


def main():
    with open(REFERENCE_LOG, "r", encoding="utf-8") as f:
        reference = [json.loads(line) for line in f]

    pipeline = RAGPipeline()
    ok = check(pipeline, reference, "no arms")
    pipeline.arms = [MitigationArm()]
    ok &= check(pipeline, reference, "no-op arm")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
