import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from scripts.eval.run_bias_audit import BiasAuditRunner, VARIANT_QUERIES_FILES

# Does the LLM's answer depend on the applicant's pronoun when the retrieved cases are identical?
# For each of the 40 bases the frozen system retrieves once, with the hand-written neutral variant (gender-free
# search), and the same cases in the same order are given to the LLM with the neutral, male and female wording.
# Only the query text differs. 4 runs give the repeat-noise floor. predict_articles prompt, as the whole-system runs.
# Output: logs/robustness/fixed_cases/run_{A..D}.jsonl (audit-log format, resumable by query_id);
# analysed by `python -m scripts.eval.robustness_checks --checks r3b`.

OUT_DIR = "logs/robustness/fixed_cases"
RUNS = "ABCD"
TYPES = ("neutral", "male", "female")


def load_bases():
    bases = defaultdict(dict)
    with open(VARIANT_QUERIES_FILES[0], "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["variant_type"] in TYPES:
                bases[row["base_query_id"]][row["variant_type"]] = row
    return dict(sorted(bases.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=RUNS)
    args = parser.parse_args()

    bases = load_bases()
    runner = BiasAuditRunner(log_path=f"{OUT_DIR}/run_{args.runs[0]}.jsonl", predict_articles=True)
    pipeline = runner.pipeline

    print(f"Retrieving the fixed cases for {len(bases)} bases (neutral wording, frozen system)...")
    fixed = {b: pipeline.retrieve(v["neutral"]["query_text"])[1] for b, v in bases.items()}

    for run in args.runs:
        runner.log_path = pipeline.config.log_path = f"{OUT_DIR}/run_{run}.jsonl"
        done = runner.load_done()
        todo = [(b, t) for b in bases for t in TYPES if bases[b][t]["variant_id"] not in done]
        print(f"\nRun {run}: {len(todo)} generations left ({len(done)} already logged)")
        start, errors = time.time(), 0
        for i, (b, t) in enumerate(todo, start=1):
            v = bases[b][t]
            result = pipeline.answer(v["query_text"], fixed[b], query_id=v["variant_id"])
            errors += bool(result["generation_error"])
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                print(f"  [{i}/{len(todo)}] {v['variant_id']} ({rate:.2f}/s, ~{(len(todo) - i) / rate / 60:.1f} min left)")
        print(f"Run {run} complete: {errors} generation errors, log at {runner.log_path}")


if __name__ == "__main__":
    main()
