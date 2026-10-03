import argparse
import csv
import json

MAIN_QUERIES = "dataset/eval/audit_queries_main.jsonl"
SELECTION = "dataset/eval/variant_batch2_selection.csv"

# Same row order and gender labels as dataset/eval/llm_queries_gender_and_emotional_variants.jsonl
VARIANTS = (("neutral", "Unknown"), ("male", "Male"), ("female", "Female"), ("emotional", "Unknown"))


def load_main(path):
    with open(path, "r", encoding="utf-8") as f:
        return {r["query_id"]: r for r in map(json.loads, f)}


def write_template(path, base_ids, main):
    # query_text is pre-filled with the base text for every variant: the author rewrites each row by hand
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for base in base_ids:
            q = main[base]
            for vtype, gender in VARIANTS:
                f.write(json.dumps({
                    "variant_id": f"{base}_{vtype}",
                    "base_query_id": base,
                    "source_case_id": q["source_case_id"],
                    "variant_type": vtype,
                    "gender": gender,
                    "query_text": q["query_text"],
                    "article": q["article"],
                    "country": q["country"],
                    "year": q["year"],
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} rows ({len(base_ids)} bases) to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=SELECTION)
    parser.add_argument("--primary-output", default="dataset/eval/llm_queries_gender_and_emotional_variants_batch2.jsonl")
    parser.add_argument("--reserve-output", default="dataset/eval/llm_queries_gender_and_emotional_variants_batch2_reserves.jsonl")
    args = parser.parse_args()

    main_q = load_main(MAIN_QUERIES)
    with open(args.selection, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    primary = [r["query_id"] for r in sorted((r for r in rows if r["role"] == "primary"), key=lambda r: int(r["order"]))]
    reserve = [r["query_id"] for r in sorted((r for r in rows if r["role"] == "reserve"), key=lambda r: int(r["order"]))]

    write_template(args.primary_output, primary, main_q)
    write_template(args.reserve_output, reserve, main_q)


if __name__ == "__main__":
    main()
