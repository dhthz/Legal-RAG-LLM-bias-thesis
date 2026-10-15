import argparse
import sys

from validate import load_jsonl


def retrieval(path):
    return {(r["run_id"], r["query_id"]): [d["doc_id"] for d in sorted(r["retrieved"], key=lambda d: d["rank"])]
            for r in load_jsonl(path) if not r["arms"]}


# Baseline tripwire: with every fix switched off, the changed code must retrieve exactly what it retrieved before the
# change (same documents in the same order for every query). Compares the baseline rows of two run logs.
def main():
    parser = argparse.ArgumentParser(description="Check that the baseline retrieval did not change.")
    parser.add_argument("--reference", required=True, help="Run log written before the code change")
    parser.add_argument("--check", required=True, help="Run log written after it, with every fix off")
    args = parser.parse_args()

    ref, new = retrieval(args.reference), retrieval(args.check)
    common = {q for _, q in ref} & {q for _, q in new}
    ref = {q: docs for (_, q), docs in sorted(ref.items()) if q in common}
    new = {q: docs for (_, q), docs in sorted(new.items()) if q in common}
    if not common:
        sys.exit("FAILED: the two logs share no baseline query (rows with arms empty)")
    differ = [q for q in sorted(common) if ref[q] != new[q]]
    if differ:
        print(f"FAILED: {len(differ)}/{len(common)} queries retrieve different documents, e.g.")
        for q in differ[:3]:
            print(f"  {q}: before {ref[q]}, after {new[q]}")
        sys.exit(1)
    print(f"PASSED: {len(common)}/{len(common)} queries retrieve the same documents in the same order")


if __name__ == "__main__":
    main()
