import argparse
import difflib
import html
import json
import os
import re

from src.mitigation.neutral_rewrite import GENDERED, neutralize_many, swap_gender_many

# How faithful the automatic gender rewrites are, measured on the 40 hand-written audit bases:
#   swap fidelity: swap_gender(male) vs female, the counterfactual generator G used by PCF / CDA. The hand-written
#                  variants changed the applicant's words only, while CDA swaps everyone's, so words are split into
#                  applicant words (changed by hand) and third-party gendered words (kept by hand)
#   invariance:    neutralize(male) == neutralize(female), the property that makes the blind query pronoun-proof
#   residue:       gendered words left after neutralize() (target 0)
#   reference:     closeness of neutralize(male) to the hand-written neutral variant (which rewrote the applicant only)

VARIANT_QUERIES_PATH = "dataset/eval/audit_queries_variants.jsonl"
MAIN_QUERIES_PATH = "dataset/eval/audit_queries_main.jsonl"
OUT_PATH = "logs/mitigation/rewrite_fidelity.json"
SIDE_BY_SIDE_PATH = "logs/mitigation/rewrite_side_by_side.html"
WORD = re.compile(r"\S+")
TOKEN = re.compile(r"\w+|[^\w\s]")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def token_diff(a, b):
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    ops = [(" ".join(ta[i1:i2]), " ".join(tb[j1:j2]))
           for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes() if op != "equal"]
    return ops


def similarity(a, b):
    return difflib.SequenceMatcher(None, TOKEN.findall(a), TOKEN.findall(b), autojunk=False).ratio()


# Per male-variant token: was it changed by hand (applicant) or a kept gendered word (third party)?
def swap_word_accuracy(male, female, swapped):
    tm, tf, ts = TOKEN.findall(male), TOKEN.findall(female), TOKEN.findall(swapped)
    if len(ts) != len(tm):
        return None
    applicant_ok = applicant = third_party = third_party_swapped = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, tm, tf, autojunk=False).get_opcodes():
        if op == "equal":
            for i in range(i1, i2):
                if GENDERED.fullmatch(tm[i]):
                    third_party += 1
                    third_party_swapped += ts[i] != tm[i]
        elif op == "replace" and i2 - i1 == j2 - j1:
            for i, j in zip(range(i1, i2), range(j1, j2)):
                applicant += 1
                applicant_ok += ts[i] == tf[j]
    return applicant_ok, applicant, third_party, third_party_swapped


def exact_rate(pairs):
    hits = sum(a == b for a, b in pairs)
    return {"identical": hits, "of": len(pairs), "rate": hits / len(pairs)}


def swap_summary(bases, male, female, swapped):
    counts = {b: swap_word_accuracy(m, f, s) for b, m, f, s in zip(bases, male, female, swapped)}
    scored = [c for c in counts.values() if c is not None]
    ok, applicant, third, third_swapped = (sum(col) for col in zip(*scored))
    return {
        "applicant_words_correct": ok, "applicant_words": applicant, "applicant_word_accuracy": ok / applicant,
        "third_party_gendered_words": third, "third_party_words_swapped": third_swapped,
        "bases_scored": len(scored),
        "applicant_errors": {b: c[1] - c[0] for b, c in counts.items() if c and c[0] < c[1]},
    }


# Words of `text` that differ from `reference` are highlighted, so every rewrite can be checked by eye
def highlight(text, reference):
    tt, tr = WORD.findall(text), WORD.findall(reference)
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, tr, tt, autojunk=False).get_opcodes():
        words = html.escape(" ".join(tt[i1:i2]))
        if words:
            out.append(words if op == "equal" else f"<mark>{words}</mark>")
    return " ".join(out)


def write_side_by_side(path, bases, by_base, neutral_auto, swap_auto, invariance_mismatch):
    columns = ["Male variant (input)", "Automatic neutral", "Hand-written neutral", "Automatic swap", "Hand-written female"]
    rows = []
    for b, n_auto, s_auto in zip(bases, neutral_auto, swap_auto):
        male = by_base[b]["male"]
        flag = " (neutral rewrite of male and female differ)" if b in invariance_mismatch else ""
        cells = [html.escape(male), highlight(n_auto, male), highlight(by_base[b]["neutral"], male),
                 highlight(s_auto, male), highlight(by_base[b]["female"], male)]
        rows.append(f"<tr><th colspan='5'>{b}{flag}</th></tr><tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>Rewrite side by side</title><style>"
            "body{font:13px/1.5 system-ui,sans-serif;margin:16px;background:#fafaf7;color:#222}"
            "table{border-collapse:collapse;width:100%;table-layout:fixed}td{vertical-align:top;padding:8px;"
            "border:1px solid #ddd;background:#fff}th{text-align:left;padding:10px 8px 4px;font-size:14px}"
            "thead th{position:sticky;top:0;background:#eef;border:1px solid #ccd}mark{background:#ffe08a;padding:0 1px}"
            "</style><h1>Gender rewrites on the 40 audit bases</h1><p>Highlighted words differ from the male variant. "
            "Columns 2 and 4 are produced by src/mitigation/neutral_rewrite.py; columns 3 and 5 are the hand-written "
            "audit variants.</p><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead>"
            + "".join(rows) + "</table>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    by_base = {}
    for v in load_jsonl(VARIANT_QUERIES_PATH):
        by_base.setdefault(v["base_query_id"], {})[v["variant_type"]] = v["query_text"]
    bases = sorted(by_base)
    male = [by_base[b]["male"] for b in bases]
    female = [by_base[b]["female"] for b in bases]
    neutral = [by_base[b]["neutral"] for b in bases]

    swapped_male, swapped_female = swap_gender_many(male), swap_gender_many(female)
    neutral_male, neutral_female = neutralize_many(male), neutralize_many(female)

    all_queries = [q["query_text"] for q in load_jsonl(MAIN_QUERIES_PATH)] + [t for b in bases for t in by_base[b].values()]
    residue = [GENDERED.findall(t) for t in neutralize_many(all_queries)]

    res = {
        "n_bases": len(bases),
        "swap_fidelity": {
            "male_to_female_exact": exact_rate(list(zip(swapped_male, female))),
            "female_to_male_exact": exact_rate(list(zip(swapped_female, male))),
            **swap_summary(bases, male, female, swapped_male),
            "mismatches": {b: token_diff(s, f) for b, s, f in zip(bases, swapped_male, female) if s != f},
        },
        "neutral_invariance": {
            **exact_rate(list(zip(neutral_male, neutral_female))),
            "mismatches": {b: token_diff(m, f) for b, m, f in zip(bases, neutral_male, neutral_female) if m != f},
        },
        "neutral_residue": {
            "queries": len(all_queries),
            "queries_with_gendered_words_left": sum(bool(r) for r in residue),
            "words_left": sum(len(r) for r in residue),
        },
        "vs_hand_neutral": {
            "mean_token_similarity": sum(similarity(a, h) for a, h in zip(neutral_male, neutral)) / len(bases),
            "hand_neutral_similarity_to_male": sum(similarity(m, h) for m, h in zip(male, neutral)) / len(bases),
            "examples": {b: token_diff(a, h)[:8] for b, a, h in list(zip(bases, neutral_male, neutral))[:5]},
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_side_by_side(SIDE_BY_SIDE_PATH, bases, by_base, neutral_male, swapped_male,
                       res["neutral_invariance"]["mismatches"])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    sf, ni, nr = res["swap_fidelity"], res["neutral_invariance"], res["neutral_residue"]
    print(f"Swap fidelity: applicant words {sf['applicant_words_correct']}/{sf['applicant_words']} correct "
          f"({sf['applicant_word_accuracy']:.1%}) over {sf['bases_scored']} bases; third-party gendered words also swapped "
          f"{sf['third_party_words_swapped']}/{sf['third_party_gendered_words']}; exact text match "
          f"{sf['male_to_female_exact']['identical']}/{len(bases)}")
    print(f"Neutral invariance (male and female rewrite to the same text): {ni['identical']}/{len(bases)}")
    print(f"Residue after neutralize: {nr['queries_with_gendered_words_left']}/{nr['queries']} queries, {nr['words_left']} words")
    vh = res["vs_hand_neutral"]
    print(f"Token similarity to hand-written neutral: auto {vh['mean_token_similarity']:.3f} "
          f"(original male variant {vh['hand_neutral_similarity_to_male']:.3f})")
    print(f"Saved: {args.out} and {SIDE_BY_SIDE_PATH}")


if __name__ == "__main__":
    main()
