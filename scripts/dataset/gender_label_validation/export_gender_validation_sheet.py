import argparse
import csv
import json
import random
import re

CLASSES = ("Male", "Female", "Multiple Applicants")
GENDERED = re.compile(r"\b(he|him|his|himself|she|her|hers|herself|mr|mrs|ms|miss)\b", re.I)
APPLICANT = re.compile(r"\bapplicants?\b", re.I)
EXCEL_CELL_LIMIT = 32000


def evidence_snippets(facts, max_snippets):
    # Prefer sentences that mention the applicant and carry gendered words; fall back to the opening facts
    picked = []
    for i, fact in enumerate(facts):
        for sentence in re.split(r"(?<=[.!?])\s+", fact):
            if APPLICANT.search(sentence) and GENDERED.search(sentence):
                picked.append(f"[{i}] {sentence.strip()}")
                if len(picked) >= max_snippets:
                    return picked
    if len(picked) < max_snippets:
        for i, fact in enumerate(facts[:3]):
            picked.append(f"[{i}] {fact.strip()[:300]}")
    return picked[:max_snippets]


def load_ai_labelled(path):
    # Human-labelled cases (confidence == "human") are ground truth already, so they are never part of the sample
    by_class = {c: [] for c in CLASSES}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cls = rec["classification"]
            if cls["gender"] in CLASSES and cls.get("confidence") != "human":
                by_class[cls["gender"]].append(rec)
    return by_class


def draw_sample(records, n, rng):
    # Male/Female split the draw across High and Medium confidence so the confidence flag can be validated too
    confidences = sorted({r["classification"].get("confidence") for r in records})
    if len(confidences) < 2:
        return [(r, "all") for r in rng.sample(records, min(n, len(records)))]
    sample = []
    per_conf = n // len(confidences)
    for conf in confidences:
        pool = [r for r in records if r["classification"].get("confidence") == conf]
        sample += [(r, conf) for r in rng.sample(pool, min(per_conf, len(pool)))]
    taken = {r["case_id"] for r, _ in sample}
    shortfall = n - len(sample)
    if shortfall > 0:
        rest = [r for r in records if r["case_id"] not in taken]
        sample += [(r, r["classification"].get("confidence")) for r in rng.sample(rest, min(shortfall, len(rest)))]
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/train_with_metadata.jsonl")
    parser.add_argument("--output", default="dataset/gender_validation_to_tag.csv")
    parser.add_argument("--key-output", default="dataset/gender_validation_key.csv")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_class = load_ai_labelled(args.input)

    drawn = []
    for cls in CLASSES:
        part = draw_sample(by_class[cls], args.per_class, rng)
        print(f"{cls:<22} pool {len(by_class[cls]):>5}  sampled {len(part)}")
        drawn += [(r, c) for r, c in part]
    rng.shuffle(drawn)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f_tag, \
         open(args.key_output, "w", encoding="utf-8-sig", newline="") as f_key:
        tag = csv.DictWriter(f_tag, fieldnames=["case_id", "title", "applicants", "human_gender", "evidence", "facts"])
        key = csv.DictWriter(f_key, fieldnames=["case_id", "ai_gender", "ai_confidence", "stratum_pool_size"])
        tag.writeheader()
        key.writeheader()
        for rec, conf in drawn:
            tag.writerow({
                "case_id": rec["case_id"],
                "title": rec["title"],
                "applicants": "; ".join(rec.get("applicants") or []),
                "human_gender": "",
                "evidence": "\n".join(evidence_snippets(rec["facts"], 6)),
                "facts": "\n".join(rec["facts"])[:EXCEL_CELL_LIMIT],
            })
            key.writerow({
                "case_id": rec["case_id"],
                "ai_gender": rec["classification"]["gender"],
                "ai_confidence": rec["classification"].get("confidence"),
                "stratum_pool_size": len(by_class[rec["classification"]["gender"]]),
            })

    print(f"\nWrote {len(drawn)} cases (seed {args.seed}) to {args.output}; hidden AI labels in {args.key_output}")
    print("Fill human_gender with one of: Male, Female, Multiple Applicants, Unknown")


if __name__ == "__main__":
    main()
