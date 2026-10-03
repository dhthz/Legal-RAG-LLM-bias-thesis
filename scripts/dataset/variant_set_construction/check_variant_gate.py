import argparse
import difflib
import json
import math
import re
from collections import defaultdict

from nrclex import NRCLex

MAIN_QUERIES = "dataset/eval/final_llm_queries.jsonl"
DEFAULT_FILE = "dataset/eval/llm_queries_gender_and_emotional_variants_batch2.jsonl"

EMOTION_TAGS = ("fear", "anger", "sadness", "disgust")
MIN_RELATIVE_LIFT = 0.06
MAX_WORDCOUNT_DRIFT = 0.03

# HE and SHE must match once these gender-marking tokens are masked (relative nouns like brother/sister are NOT masked on purpose)
GENDER_TOKENS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "mr", "mrs", "ms", "miss",
                 "man", "woman", "men", "women", "male", "female", "gentleman", "lady"}
TOKEN = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")


def score(text):
    # Same formula as sentiment_extractor.py: share of NRC tags that are fear/anger/sadness/disgust, capped at 1.0
    n = NRCLex(text)
    f = n.affect_frequencies
    return min(sum(f[e] for e in EMOTION_TAGS), 1.0), n.raw_emotion_scores, n.affect_dict


def tag_counts(raw):
    emo = sum(raw.get(e, 0) for e in EMOTION_TAGS)
    return emo, sum(raw.values())


def masked_tokens(text):
    return ["<G>" if t.lower() in GENDER_TOKENS else t for t in TOKEN.findall(text)]


def he_she_diff(male, female):
    a, b = masked_tokens(male), masked_tokens(female)
    if a == b:
        return []
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            out.append(f"male '{' '.join(a[i1:i2])}' vs female '{' '.join(b[j1:j2])}' (token {i1})")
    return out


def swaps_needed(emo, total, target, emo_per_word=4, other_per_word=1):
    # Swapping a non-lexicon word for a strongly emotional one (e.g. 4 emotion tags + 1 other tag) moves the share up by this much per swap
    denom = emo_per_word - target * (emo_per_word + other_per_word)
    if denom <= 0:
        return None
    return max(0, math.ceil((target * total - emo) / denom))


def highlight(text, show_all=False):
    n = NRCLex(text)
    out = []
    for w in n.words:
        tags = n.lexicon.get(w)
        if not tags:
            continue
        emo = [t for t in tags if t in EMOTION_TAGS]
        if emo or show_all:
            out.append((w, emo, [t for t in tags if t not in EMOTION_TAGS]))
    return out


def load_rows(path):
    by_base = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_base[r["base_query_id"]][r["variant_type"]] = r
    return by_base


def load_main(path):
    with open(path, "r", encoding="utf-8") as f:
        return {r["query_id"]: r for r in map(json.loads, f)}


def check_base(base, rows, original_text, source_gender):
    report = {"base": base, "status": [], "notes": []}
    texts = {v: rows[v]["query_text"] for v in rows}
    words = {v: len(t.split()) for v, t in texts.items()}
    # The variant matching the source applicant's gender may legitimately equal the original text
    same_as_source = "female" if source_gender == "Female" else "male"
    unedited = [v for v, t in texts.items() if t.strip() == original_text.strip() and v != same_as_source]
    report["unedited"] = unedited
    if texts.get(same_as_source, "").strip() == original_text.strip():
        report["notes"].append(f"{same_as_source} row equals the original text (expected only if the original already names no applicant and needs no change)")
    report["words"] = words

    neu = texts.get("neutral")
    emo = texts.get("emotional")
    if "neutral" in unedited or "emotional" in unedited:
        report["status"].append("neutral/emotional not written yet" if "emotional" in unedited else "neutral not written yet")
    if neu and emo and emo.strip() == neu.strip():
        report["status"].append("emotional row is identical to the neutral row (not rewritten yet)")
    elif neu and emo and "emotional" not in unedited:
        s_n, raw_n, _ = score(neu)
        s_e, raw_e, _ = score(emo)
        lift = (s_e - s_n) / s_n if s_n else float("inf")
        report.update({"neutral_score": s_n, "emotional_score": s_e, "lift": lift})
        drift = abs(words["emotional"] - words["neutral"]) / words["neutral"]
        report["wc_drift"] = drift
        report["status"].append(f"emotional lift {lift:+.1%} ({'PASS' if lift >= MIN_RELATIVE_LIFT else 'FAIL, need >= +6%'})")
        report["status"].append(f"word count {words['neutral']} -> {words['emotional']} ({drift:.1%}, {'ok' if drift <= MAX_WORDCOUNT_DRIFT else 'CHECK, over 3%'})")
        if lift < MIN_RELATIVE_LIFT:
            emo_t, tot = tag_counts(raw_n)
            k = swaps_needed(emo_t, tot, s_n * (1 + MIN_RELATIVE_LIFT))
            report["notes"].append(f"neutral has {emo_t} emotion tags of {tot} total; about {k} swap(s) of a calm word for a strongly emotional one "
                                   f"(4 emotion tags) would reach +6%")
    elif neu:
        s_n, raw_n, _ = score(neu)
        emo_t, tot = tag_counts(raw_n)
        report["neutral_score"] = s_n
        target = s_n * (1 + MIN_RELATIVE_LIFT)
        k = swaps_needed(emo_t, tot, target)
        report["notes"].append(f"neutral score {s_n:.3f} ({emo_t} emotion tags of {tot}); the emotional version needs >= {target:.3f}, "
                               f"about {k} swap(s) to strongly emotional words")

    if "male" in texts and "female" in texts and "male" not in unedited and "female" not in unedited:
        diffs = he_she_diff(texts["male"], texts["female"])
        report["he_she"] = diffs
        report["status"].append("HE/SHE identical except gender tokens: PASS" if not diffs else f"HE/SHE differ beyond gender tokens ({len(diffs)}): FAIL")

    if neu and "neutral" not in unedited:
        toks = TOKEN.findall(neu)
        spots = [i for i, t in enumerate(toks) if t.lower() in GENDER_TOKENS]
        if spots:
            report["notes"].append(f"neutral still contains {len(spots)} gender token(s); check each refers to a third party, not the applicant:")
            for i in spots[:8]:
                report["notes"].append(f"    ...{' '.join(toks[max(0, i - 6):i + 4])}...")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--base", default=None, help="Check a single base (e.g. d002); default all bases that have any edited row")
    parser.add_argument("--highlight", default=None, help="variant_id (e.g. d002_neutral): list the NRC emotion words driving its score")
    parser.add_argument("--all-tags", action="store_true", help="With --highlight, also list words that carry only positive/trust/other tags")
    args = parser.parse_args()

    by_base = load_rows(args.file)
    main_q = load_main(MAIN_QUERIES)

    if args.highlight:
        base, vtype = args.highlight.rsplit("_", 1)
        text = by_base[base][vtype]["query_text"]
        s, raw, _ = score(text)
        emo_t, tot = tag_counts(raw)
        print(f"{args.highlight}: {len(text.split())} words, intensity {s:.3f} = {emo_t} emotion tags / {tot} total tags")
        print("tag totals:", dict(sorted(raw.items(), key=lambda kv: -kv[1])))
        print("\nwords carrying fear/anger/sadness/disgust tags (case-sensitive match, as in the scoring):")
        for w, emo, other in sorted(highlight(text, args.all_tags), key=lambda x: -len(x[1])):
            print(f"  {w:<16} emotion: {','.join(emo) or '-':<28} other: {','.join(other)}")
        return

    bases = [args.base] if args.base else sorted(by_base)
    shown = 0
    for base in bases:
        original = main_q[base]["query_text"].strip()
        rep = check_base(base, by_base[base], main_q[base]["query_text"], main_q[base]["gender"])
        written = [v for v in by_base[base] if by_base[base][v]["query_text"].strip() != original]
        if not args.base and not written:
            continue
        shown += 1
        print(f"\n{base}  words {rep['words']}")
        print(f"  written rows: {written or 'none'} | still the original text: {rep['unedited'] or 'none'}")
        for line in rep["status"]:
            print("  -", line)
        for line in rep["notes"]:
            print("  *", line)
        for d in rep.get("he_she", [])[:6]:
            print("    diff:", d)
    if not shown:
        print("No edited rows found yet.")


if __name__ == "__main__":
    main()
