import argparse
import json
import os
import random
import re
import time

from scripts.dataset.generate_eval_queries import build_query
from scripts.dataset.variant_set_construction.select_variant_bases import (INTRINSIC, PRONOUNS, SENSITIVE,
                                                                            applicant_mentions)
from src.mitigation import build_arms
from src.mitigation.neutral_rewrite import swap_gender_many
from src.rag.pipeline import RAGPipeline

# R4: the pronoun-flip test on a larger, automatically built set of he/she pairs (search only, no LLM).
# Pairs: test cases (outside the index) whose facts refer to a single applicant by one gender, with the same gates as
# the hand-written 40 where possible (see MIN_MENTIONS below; not gender-intrinsic, not a sensitive topic).
# The query is built like the B.1 harness queries (facts, names stripped); its twin is swap_gender() of it, the
# two-sided CDA swap, so third parties are swapped too (the hand-written variants kept them).
# Output: logs/robustness/swap_set/pairs.jsonl and <arm>.jsonl (audit-log format, one row per query, resumable per arm);
# analysed by `python -m scripts.eval.robustness_checks --checks r4`.

OUT_DIR = "logs/robustness/swap_set"
PAIRS_PATH = f"{OUT_DIR}/pairs.jsonl"
TEST_PATH = "dataset/test.jsonl"
VARIANTS_PATH = "dataset/eval/audit_queries_variants.jsonl"
MAIN_PATH = "dataset/eval/audit_queries_main.jsonl"
ARMS = ("baseline", "blind_query", "pcf", "leace", "blind_index", "cda", "ft_control")
N_PAIRS, SEED = 300, 42
# Looser than the hand-written set (there: other gender absent, <= 1500 words, a limit set by hand-writing effort):
# with those gates only 34 female-applicant test cases exist. Here the applicant's gender must dominate 3:1 and the
# query stays within the length range of the 175 main audit queries.
MIN_MENTIONS, DOMINANCE, MAX_WORDS = 3, 3, 3000


def dominant_gender(facts):
    counts = {g: applicant_mentions(facts, g) for g in PRONOUNS}
    male, female = counts["Male"], counts["Female"]
    if male >= MIN_MENTIONS and male >= DOMINANCE * female:
        return "Male", counts
    if female >= MIN_MENTIONS and female >= DOMINANCE * male:
        return "Female", counts
    return None, counts


def hand_set_sources():
    main = {}
    with open(MAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            main[q["query_id"]] = q["source_case_id"]
    with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
        return {main[json.loads(line)["base_query_id"]] for line in f}


def build_pairs():
    excluded = hand_set_sources()
    pool, reasons = {"Male": [], "Female": []}, {"hand_set": 0, "no_single_gender": 0, "intrinsic_or_sensitive": 0,
                                                  "too_long": 0}
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            if case["case_id"] in excluded:
                reasons["hand_set"] += 1
                continue
            gender, counts = dominant_gender(case["facts"])
            if gender is None:
                reasons["no_single_gender"] += 1
                continue
            text = " ".join(case["facts"])
            if INTRINSIC.search(text) or SENSITIVE.search(text):
                reasons["intrinsic_or_sensitive"] += 1
                continue
            query, _ = build_query(case)
            if len(query.split()) > MAX_WORDS:
                reasons["too_long"] += 1
                continue
            pool[gender].append({"source_case_id": case["case_id"], "source_gender": gender, "query": query,
                                 "applicant_mentions": counts})
    print(f"Eligible test cases: Male {len(pool['Male'])}, Female {len(pool['Female'])}; excluded {reasons}")

    rng = random.Random(SEED)
    n_female = min(len(pool["Female"]), N_PAIRS // 2)
    chosen = rng.sample(pool["Female"], n_female) + rng.sample(pool["Male"], min(len(pool["Male"]), N_PAIRS - n_female))
    chosen.sort(key=lambda c: c["source_case_id"])

    print(f"Swapping {len(chosen)} queries (spaCy + GPT-2 her-resolution)...")
    swapped = swap_gender_many([c["query"] for c in chosen])
    back = swap_gender_many(swapped)
    pairs = []
    for i, (c, s, b) in enumerate(zip(chosen, swapped, back), start=1):
        male, female = (c["query"], s) if c["source_gender"] == "Male" else (s, c["query"])
        pairs.append({"pair_id": f"s{i:03d}", **{k: c[k] for k in ("source_case_id", "source_gender", "applicant_mentions")},
                      "male": male, "female": female, "round_trip_identical": b == c["query"],
                      "gendered_tokens": len(re.findall(r"\b(he|she|him|his|her|hers|himself|herself)\b", c["query"], re.I))})
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {PAIRS_PATH} "
          f"(swap round trip identical: {sum(p['round_trip_identical'] for p in pairs)}/{len(pairs)})")
    return pairs


def load_pairs():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_arm(pipeline, arm, pairs):
    path = f"{OUT_DIR}/{arm}.jsonl"
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["query_id"] for line in f}
    pipeline.arms = build_arms([] if arm == "baseline" else [arm])
    todo = [(p, g) for p in pairs for g in ("male", "female") if f"{p['pair_id']}_{g}" not in done]
    print(f"\n[{arm}] {len(todo)} queries left ({len(done)} already logged)")
    start = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i, (p, g) in enumerate(todo, start=1):
            effective_query, cases = pipeline.retrieve(p[g])
            row = {"query_id": f"{p['pair_id']}_{g}", "pair_id": p["pair_id"], "variant_type": g,
                   **({"rewritten_query": effective_query} if effective_query != p[g] else {}),
                   "arms": [a.name for a in pipeline.arms],
                   "retrieved_cases": [pipeline.case_log_entry(rank, c) for rank, c in enumerate(cases, start=1)]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                print(f"  [{i}/{len(todo)}] ({rate:.2f}/s, ~{(len(todo) - i) / rate / 60:.1f} min left)", flush=True)
    print(f"[{arm}] complete: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--rebuild-pairs", action="store_true")
    args = parser.parse_args()

    pairs = build_pairs() if args.rebuild_pairs or not os.path.exists(PAIRS_PATH) else load_pairs()
    pipeline = RAGPipeline()
    for arm in args.arms.split(","):
        run_arm(pipeline, arm, pairs)


if __name__ == "__main__":
    main()
