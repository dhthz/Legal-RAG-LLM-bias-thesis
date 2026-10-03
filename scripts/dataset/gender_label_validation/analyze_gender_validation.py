import argparse
import csv
import json
from collections import Counter, defaultdict

from scripts.dataset.gender_label_validation.export_gender_validation_sheet import CLASSES
from scripts.eval.bias_statistics import wilson_ci

HUMAN_VALUES = ("Male", "Female", "Multiple Applicants", "Unknown")
HUMAN_ALIASES = {"multiple applicants": "Multiple Applicants", "multliple applicants": "Multiple Applicants",
                 "male": "Male", "female": "Female", "unknown": "Unknown"}


def load_pairs(tagged_path, key_path):
    with open(key_path, "r", encoding="utf-8-sig", newline="") as f:
        key = {r["case_id"]: r for r in csv.DictReader(f)}
    pairs = []
    with open(tagged_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            human = row["human_gender"].strip()
            if not human:
                continue
            human = HUMAN_ALIASES.get(human.lower(), human)
            if human not in HUMAN_VALUES:
                raise ValueError(f"{row['case_id']}: '{human}' is not one of {HUMAN_VALUES}")
            k = key[row["case_id"]]
            pairs.append({"case_id": row["case_id"], "ai": k["ai_gender"], "conf": k["ai_confidence"],
                          "human": human, "pool": int(k["stratum_pool_size"])})
    return pairs


def accuracy_block(rows):
    n = len(rows)
    correct = sum(r["ai"] == r["human"] for r in rows)
    lo, hi = wilson_ci(correct, n)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None, "wilson_95ci": [lo, hi]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged", default="dataset/gender_validation_to_tag.csv")
    parser.add_argument("--key", default="dataset/gender_validation_key.csv")
    parser.add_argument("--output", default="logs/bias_audit/gender_validation_results.json")
    args = parser.parse_args()

    pairs = load_pairs(args.tagged, args.key)
    print(f"{len(pairs)} tagged cases")

    by_class = defaultdict(list)
    for p in pairs:
        by_class[p["ai"]].append(p)

    per_class = {c: accuracy_block(by_class[c]) for c in CLASSES if by_class[c]}
    # Unknown = the human could not decide from the facts, so also report accuracy over decidable cases only
    decidable = {c: accuracy_block([p for p in by_class[c] if p["human"] != "Unknown"]) for c in per_class}

    # Overall accuracy reweighted by each class's share of the AI-labelled corpus; stratified-sampling standard error
    total_pool = sum(by_class[c][0]["pool"] for c in per_class)
    weights = {c: by_class[c][0]["pool"] / total_pool for c in per_class}
    overall = sum(weights[c] * per_class[c]["accuracy"] for c in per_class)
    se = sum((weights[c] ** 2) * per_class[c]["accuracy"] * (1 - per_class[c]["accuracy"]) / per_class[c]["n"] for c in per_class) ** 0.5
    overall_block = {"accuracy": overall, "approx_95ci": [overall - 1.96 * se, overall + 1.96 * se], "weights": weights}

    by_conf = {}
    for c in ("Male", "Female"):
        for conf in sorted({p["conf"] for p in by_class[c]}):
            by_conf[f"{c}/{conf}"] = accuracy_block([p for p in by_class[c] if p["conf"] == conf])

    confusion = {c: dict(Counter(p["human"] for p in by_class[c])) for c in per_class}

    male, female = by_class["Male"], by_class["Female"]
    swaps = {
        "ai_male_but_human_female": sum(p["human"] == "Female" for p in male),
        "ai_female_but_human_male": sum(p["human"] == "Male" for p in female),
    }

    print(f"\n{'Class':<22}{'n':>5}{'correct':>9}{'accuracy':>10}   95% Wilson CI")
    for c, b in per_class.items():
        print(f"{c:<22}{b['n']:>5}{b['correct']:>9}{b['accuracy']:>10.1%}   [{b['wilson_95ci'][0]:.1%}, {b['wilson_95ci'][1]:.1%}]")
    print(f"\nOverall (reweighted to corpus shares): {overall:.1%}  95% CI [{overall_block['approx_95ci'][0]:.1%}, {overall_block['approx_95ci'][1]:.1%}]")
    print("\nExcluding cases the human tagged Unknown:")
    for c, b in decidable.items():
        print(f"  {c:<22} n={b['n']:>3}  accuracy {b['accuracy']:.1%}  [{b['wilson_95ci'][0]:.1%}, {b['wilson_95ci'][1]:.1%}]")
    print("\nBy confidence:")
    for k, b in by_conf.items():
        print(f"  {k:<18} n={b['n']:>3}  accuracy {b['accuracy']:.1%}  [{b['wilson_95ci'][0]:.1%}, {b['wilson_95ci'][1]:.1%}]")
    print("\nConfusion (rows = AI label, columns = human label):")
    for c, row in confusion.items():
        print(f"  {c:<22}{row}")
    print(f"\nMale/Female swaps: {swaps}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"n_tagged": len(pairs), "per_class": per_class, "excluding_unknown": decidable, "overall_reweighted": overall_block,
                   "by_confidence": by_conf, "confusion": confusion, "swaps": swaps}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
