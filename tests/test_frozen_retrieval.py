import hashlib
import json
import sys

from src.mitigation import MitigationArm
from src.rag.pipeline import RAGPipeline

# Tripwire for Phase F: with no arms (and with a no-op arm, which exercises every hook), retrieval must
# reproduce the frozen audit run exactly. The reference is the retrieval of the 335 audit queries in
# logs/bias_audit/stateless/run40_A.jsonl, as one SHA256 per query of its case ids, chunk ids and scores;
# --write-reference rebuilds it from that log
SOURCE_LOG = "logs/bias_audit/stateless/run40_A.jsonl"
REFERENCE = "tests/frozen_retrieval_reference.jsonl"
QUERY_FILES = {"docs/llm_queries/audit_queries_main.jsonl": "query_id",
               "docs/llm_queries/audit_queries_variants.jsonl": "variant_id"}


def retrieval_signature(cases):
    signature = [[c["case_id"], [[ch["chunk_id"], round(float(ch["similarity_score"]), 6)] for ch in c["chunks"]]]
                 for c in cases]
    return hashlib.sha256(json.dumps(signature).encode("utf-8")).hexdigest()


def load_queries():
    queries = {}
    for path, id_field in QUERY_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                queries[row[id_field]] = row["query_text"]
    return queries


def write_reference():
    queries = load_queries()
    with open(SOURCE_LOG, "r", encoding="utf-8") as f_in, open(REFERENCE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            if queries[row["query_id"]] != row["query"]:
                raise ValueError(f"{row['query_id']}: the logged query differs from the tracked query file")
            f_out.write(json.dumps({"query_id": row["query_id"],
                                    "sha256": retrieval_signature(row["retrieved_cases"])}) + "\n")
    print(f"Saved: {REFERENCE}")


def check(pipeline, reference, queries, label):
    mismatches = [row["query_id"] for row in reference
                  if retrieval_signature(pipeline.retrieve(queries[row["query_id"]])[1]) != row["sha256"]]
    print(f"{label}: {len(reference) - len(mismatches)}/{len(reference)} identical")
    if mismatches:
        print(f"  first mismatches: {mismatches[:10]}")
    return not mismatches


def main():
    if "--write-reference" in sys.argv:
        write_reference()
        return 0
    with open(REFERENCE, "r", encoding="utf-8") as f:
        reference = [json.loads(line) for line in f]
    queries = load_queries()

    pipeline = RAGPipeline()
    ok = check(pipeline, reference, queries, "no arms")
    pipeline.arms = [MitigationArm()]
    ok &= check(pipeline, reference, queries, "no-op arm")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
