import argparse
import json
import os
import random
import re
import sys

# The rewrite functions are in the plugin's fixes/ folder, two levels above this skill
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from fixes.neutral_rewrite import PERSON_SWAP, neutralize_many, swap_gender_many  # noqa: E402
from fixes.pcf import query_attribute  # noqa: E402

SEED = 42
# Gendered words the swap does not change (it cannot know "actor" for "actress"); a twin with one is left out
UNSWAPPED = ("actress", "waitress", "hostess", "stewardess", "businessman", "businesswoman", "chairman", "chairwoman",
             "spokesman", "spokeswoman", "salesman", "saleswoman", "policeman", "policewoman", "fireman", "king", "queen",
             "prince", "princess", "sir", "madam", "ma'am")
CUES = re.compile(r"\b(" + "|".join(sorted(set(PERSON_SWAP) | {w + s for w in UNSWAPPED for s in ("", "s", "es")},
                                          key=len, reverse=True)) + r")\b", re.IGNORECASE)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Gender words still in the same place in both twins (not swapped), or left in the neutral version
def leftover_cues(versions):
    he, she = ([w.lower() for w in CUES.findall(versions[v])] for v in ("male", "female"))
    left = {a for a, b in zip(he, she) if a == b} | (set(he) ^ set(she) if len(he) != len(she) else set())
    if "neutral" in versions:
        left |= {w.lower() for w in CUES.findall(versions["neutral"])}
    return sorted(left)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Every question is kept as it is (with its own attribute value, from the input or from its pronouns). Questions whose
# pronouns are clearly he or she also get twins: the he version, the she version and, with --neutral, a version with
# singular they. Questions without pronouns or with both he and she equally often cannot be twinned.
def main():
    parser = argparse.ArgumentParser(description="Build he/she twin questions for the bias audit.")
    parser.add_argument("--questions", required=True, help="JSONL with query_id and query (optional attribute_value, topics)")
    parser.add_argument("--groups", nargs=2, default=["Male", "Female"], metavar=("HE_VALUE", "SHE_VALUE"),
                        help="Label values of the he and she groups, as in the labels file")
    parser.add_argument("--max-twins", type=int, default=None, help="Twin only this many questions (random, fixed seed)")
    parser.add_argument("--neutral", action="store_true", help="Also write a they version of each twinned question")
    parser.add_argument("--out-queries", default="audit_queries.jsonl", help="Every question to run through the system")
    parser.add_argument("--out-pairs", default="query_pairs.jsonl")
    parser.add_argument("--out-neutral", help="The they version of each twinned question, for the LLM test with fixed "
                                              "documents (default: neutral_versions.jsonl next to --out-pairs)")
    parser.add_argument("--out-left-out", help="Questions not twinned because a gender word stayed the same "
                                               "(default: twins_left_out.jsonl next to --out-pairs)")
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    value = {"male": args.groups[0], "female": args.groups[1]}
    pairs, queries, twinnable = [], [], []
    for q in questions:
        a = query_attribute(q["query"])
        row = {"query_id": q["query_id"]}
        if q.get("attribute_value") or a in value:
            row["attribute_value"] = q.get("attribute_value") or value[a]
        if q.get("topics"):
            row["topics"] = q["topics"]
        pairs.append(row)
        queries.append({"query_id": q["query_id"], "query": q["query"]})
        if a in value:
            twinnable.append((q, a))

    swapped = swap_gender_many(q["query"] for q, _ in twinnable)
    neutral = neutralize_many(q["query"] for q, _ in twinnable)
    clean, left_out = [], []
    for (q, a), other, they in zip(twinnable, swapped, neutral):
        versions = {"male": q["query"] if a == "male" else other, "female": q["query"] if a == "female" else other}
        left = leftover_cues({**versions, "neutral": they})
        if args.neutral:
            versions["neutral"] = they
        if left:
            left_out.append({"query_id": q["query_id"], "query": q["query"], "words": left, "versions": {**versions, "neutral": they}})
        else:
            clean.append((q, versions, they))

    chosen = clean
    if args.max_twins is not None and args.max_twins < len(clean):
        chosen = sorted(random.Random(SEED).sample(clean, args.max_twins), key=lambda t: t[0]["query_id"])
    for q, versions, _ in chosen:
        for variant, text in versions.items():
            qid = f"{q['query_id']}_{variant}"
            queries.append({"query_id": qid, "query": text})
            pairs.append({"query_id": qid, "pair_id": q["query_id"], "variant": variant})

    ids = [q["query_id"] for q in queries]
    if len(set(ids)) != len(ids):
        sys.exit("Some twin ids clash with existing query ids (a question id already ends in _male/_female/_neutral)")
    write_jsonl(args.out_queries, queries)
    write_jsonl(args.out_pairs, pairs)
    left_out_path = args.out_left_out or os.path.join(os.path.dirname(args.out_pairs), "twins_left_out.jsonl")
    write_jsonl(left_out_path, left_out)
    neutral_path = args.out_neutral or os.path.join(os.path.dirname(args.out_pairs), "neutral_versions.jsonl")
    write_jsonl(neutral_path, [{"pair_id": q["query_id"], "query": they} for q, _, they in chosen])
    print(f"{len(questions)} questions, {len(twinnable)} can be twinned (clear he or she), {len(left_out)} left out "
          f"(a gender word the swap does not change, listed in {left_out_path}), {len(chosen)} twinned")
    print(f"Wrote {args.out_queries} ({len(queries)} questions to run), {args.out_pairs} and {neutral_path}")
    for q in left_out[:3]:
        print(f"  left out {q['query_id']}: {', '.join(q['words'])}")
    for q, _, _ in chosen[:3]:
        print(f"\n{q['query_id']}")
        for variant in ("male", "female", "neutral"):
            hit = next((x["query"] for x in queries if x["query_id"] == f"{q['query_id']}_{variant}"), None)
            if hit:
                print(f"  {variant:8s} {hit[:150]}")


if __name__ == "__main__":
    main()
