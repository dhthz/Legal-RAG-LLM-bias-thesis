import argparse
import json
import time
from pathlib import Path

from src.rag.pipeline import RAGPipeline

MAIN_QUERIES_FILE = "dataset/eval/final_llm_queries.jsonl"
VARIANT_QUERIES_FILES = ["dataset/eval/final_llm_queries_variants.jsonl"]
AUDIT_LOG_PATH = "logs/bias_audit/bias_audit_interactions.jsonl"


class BiasAuditRunner:
    #Runs every query in the bias-audit test set (main + variant sets) through the frozen RAG pipeline, resumable by query_id.

    def __init__(self, log_path=AUDIT_LOG_PATH):
        self.log_path = log_path
        self.pipeline = RAGPipeline()
        self.pipeline.config.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_queries(variants_only=False):
        queries = []
        if not variants_only:
            with open(MAIN_QUERIES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    queries.append({"query_id": row["query_id"], "query_text": row["query_text"]})
        for path in VARIANT_QUERIES_FILES:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    queries.append({"query_id": row["variant_id"], "query_text": row["query_text"]})
        return queries

    def load_done(self):
        done = set()
        if Path(self.log_path).exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("query_id"):
                            done.add(rec["query_id"])
                    except json.JSONDecodeError:
                        continue
        return done

    def run(self, limit=None, variants_only=False):
        queries = self.load_queries(variants_only)
        print(f"Loaded {len(queries)} queries (from {MAIN_QUERIES_FILE} and {', '.join(VARIANT_QUERIES_FILES)})")

        if limit is not None:
            queries = queries[:limit]
            print(f"--limit set: processing only the first {len(queries)} queries")

        done = self.load_done()
        if done:
            print(f"Resuming: {len(done)} queries already logged in {self.log_path}")

        remaining = [q for q in queries if q["query_id"] not in done]
        print(f"{len(remaining)} queries left to run\n")

        start = time.time()
        errors = 0
        for i, q in enumerate(remaining, start=1):
            result = self.pipeline.query(q["query_text"], query_id=q["query_id"])
            if result["generation_error"]:
                errors += 1
                print(f"  [{i}/{len(remaining)}] {q['query_id']}: GENERATION ERROR "
                      f"({result['generation_error']}) — retrieval still logged")
            elif i % 10 == 0 or i == len(remaining):
                elapsed = time.time() - start
                rate = i / elapsed
                eta_min = (len(remaining) - i) / rate / 60 if rate > 0 else 0
                print(f"  [{i}/{len(remaining)}] {q['query_id']} done "
                      f"({rate:.2f}/s, ~{eta_min:.1f} min remaining)")

        print(f"\nRun complete: {len(remaining)} queries processed, "
              f"{errors} generation errors, log at {self.log_path}")
        return {"processed": len(remaining), "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N queries (smoke test)")
    parser.add_argument("--log-path", default=AUDIT_LOG_PATH,
                         help="Log file for this run (use a new path for a repeat run)")
    parser.add_argument("--variants-only", action="store_true",
                         help="Run only the variant set (stability rerun)")
    args = parser.parse_args()

    runner = BiasAuditRunner(log_path=args.log_path)
    runner.run(limit=args.limit, variants_only=args.variants_only)


if __name__ == "__main__":
    main()
