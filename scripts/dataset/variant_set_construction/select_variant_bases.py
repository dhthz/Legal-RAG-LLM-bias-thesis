import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

MAIN_QUERIES = "dataset/eval/audit_queries_main.jsonl"
VARIANT_QUERIES = "dataset/eval/audit_queries_variants.jsonl"
TEST_CASES = "dataset/test.jsonl"
TRAIN_METADATA = "dataset/train_with_metadata.jsonl"

PRONOUNS = {
    "Male": re.compile(r"\b(he|him|his|himself)\b", re.I),
    "Female": re.compile(r"\b(she|her|hers|herself)\b", re.I),
}
# Gate 3 (same as the first 20): applicant gender is intrinsic to the legal facts
INTRINSIC = re.compile(r"paternity|transsexual|transgender|gender reassignment|sex reassignment|gender identity|"
                       r"pregnan|abortion|maternity|breast-?feed|wet nurs", re.I)
# Not excluded by the original rules, but swapping the applicant's gender changes how the facts read; excluded from the draw
SENSITIVE = re.compile(r"\brape[ds]?\b|sexual (assault|abuse|violence)|domestic violence|virginity|forced marriage|"
                       r"honou?r killing", re.I)

MIN_APPLICANT_MENTIONS = 3
DEFAULT_MAX_WORDS = 1500
MAX_NEW_PER_COUNTRY = 2
MAX_TOTAL_PER_COUNTRY = 7


def load_jsonl(path, key):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec[key]] = rec
    return out


def applicant_mentions(facts, gender):
    # Gender pronouns in sentences that mention the applicant (gate 1: at least 3 applicant-referring gender mentions)
    n = 0
    for fact in facts:
        for sentence in re.split(r"(?<=[.!?])\s+", fact):
            if re.search(r"\bapplicant\b", sentence, re.I):
                n += len(PRONOUNS[gender].findall(sentence))
    return n


def build_candidates(main, test_cases, existing_bases):
    candidates = []
    for qid, q in main.items():
        if q["gender"] not in ("Male", "Female") or qid in existing_bases:
            continue
        case = test_cases.get(q["source_case_id"])
        if not case:
            continue
        text = " ".join(case["facts"])
        candidates.append({
            "base_text": q["query_text"],
            "query_id": qid, "source_case_id": q["source_case_id"], "gender": q["gender"],
            "article": q["article"], "country": q["country"], "year": q["year"],
            "source_words": len(q["query_text"].split()),
            "applicant_gender_mentions": applicant_mentions(case["facts"], q["gender"]),
            "nrc_intensity": round(q["sentiment_info"]["nrc_emotional_intensity"], 3),
            "intrinsic": bool(INTRINSIC.search(text)),
            "sensitive": bool(SENSITIVE.search(text)),
        })
    return candidates


def passes_gates(c, max_words):
    return (c["applicant_gender_mentions"] >= MIN_APPLICANT_MENTIONS and not c["intrinsic"] and not c["sensitive"]
            and c["source_words"] <= max_words)


def article_quotas(main, existing, n_new):
    # Corpus-proportional article targets for the combined set (same logic as the main 175), minus what the first 20 already cover
    shares = Counter(q["article"] for q in main.values())
    total_n = len(existing) + n_new
    raw = {a: shares[a] / sum(shares.values()) * total_n for a in shares}
    target = {a: int(v) for a, v in raw.items()}
    for a in sorted(raw, key=lambda a: raw[a] - target[a], reverse=True)[:total_n - sum(target.values())]:
        target[a] += 1
    have = Counter(existing)
    quota = {a: max(0, target[a] - have.get(a, 0)) for a in target}
    while sum(quota.values()) > n_new:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n_new:
        quota[max(raw, key=lambda a: raw[a] - target[a] - 0.01 * quota.get(a, 0))] += 1
    return {a: v for a, v in quota.items() if v > 0}


def cap_quotas(quotas, pool, main):
    # Short cases are not spread across articles like the corpus is (Art. 3 cases are long), so quotas are capped at what exists
    # and the shortfall moves to articles with spare candidates, largest corpus share first
    avail = Counter(c["article"] for c in pool)
    shares = Counter(q["article"] for q in main.values())
    capped = {a: min(v, avail.get(a, 0)) for a, v in quotas.items()}
    overflow = sum(quotas.values()) - sum(capped.values())
    for a in sorted(avail, key=lambda a: shares[a], reverse=True):
        while overflow > 0 and capped.get(a, 0) < avail[a]:
            capped[a] = capped.get(a, 0) + 1
            overflow -= 1
    return {a: v for a, v in capped.items() if v > 0}


def draw(pool, quotas, gender_need, country_total, rng, attempts=5000):
    # Seeded randomized greedy: article quota + gender quota + country caps, retried until all quotas are met exactly
    for attempt in range(1, attempts + 1):
        chosen, art_left, gen_left = [], dict(quotas), dict(gender_need)
        new_country, tot_country = Counter(), Counter(country_total)
        order = pool[:]
        rng.shuffle(order)
        for c in order:
            if art_left.get(c["article"], 0) <= 0 or gen_left.get(c["gender"], 0) <= 0:
                continue
            if new_country[c["country"]] >= MAX_NEW_PER_COUNTRY or tot_country[c["country"]] >= MAX_TOTAL_PER_COUNTRY:
                continue
            chosen.append(c)
            art_left[c["article"]] -= 1
            gen_left[c["gender"]] -= 1
            new_country[c["country"]] += 1
            tot_country[c["country"]] += 1
        if all(v == 0 for v in art_left.values()) and all(v == 0 for v in gen_left.values()):
            return chosen, attempt
    return None, attempts


def attach_analogs(rows, main, train):
    # Covariate only, never used to select: best opposite-gender case in the retrieval pool for the base's query
    from src.embeddings.chunk_embedder import ChunkRetriever
    retriever = ChunkRetriever("faiss_indices/paragraph_chunks_l2.index", "faiss_indices/paragraph_chunks_metadata_enriched.json")
    for row in rows:
        q = main[row["query_id"]]
        opposite = "Female" if q["gender"] == "Male" else "Male"
        chunks = retriever.retrieve_chunks(q["query_text"], 25)
        best = None
        for case in retriever.aggregate_chunks_to_cases(chunks, 50):
            t = train.get(case["case_id"])
            if not t or t["gender"] != opposite:
                continue
            same_country = t["country"] == q["country"]
            shared_article = q["article"] in t["articles"]
            score = (same_country + shared_article, case["max_similarity"])
            if best is None or score > best[0]:
                best = (score, case, t, same_country, shared_article)
        if best:
            _, case, t, same_country, shared_article = best
            row.update({"analog_case_id": case["case_id"], "analog_title": t["title"],
                        "analog_similarity": round(case["max_similarity"], 4),
                        "analog_same_country": same_country, "analog_shared_article": shared_article})
        else:
            row.update({"analog_case_id": "", "analog_title": "", "analog_similarity": "",
                        "analog_same_country": "", "analog_shared_article": ""})


def load_train(path):
    train = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            train[r["case_id"]] = {"gender": r["classification"]["gender"], "title": r["title"],
                                   "country": (r.get("defendants") or [None])[0], "articles": r.get("violated_articles") or []}
    return train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-new", type=int, default=20)
    parser.add_argument("--reserves", type=int, default=8)
    parser.add_argument("--output", default="dataset/eval/variant_batch2_selection.csv")
    parser.add_argument("--first-set-output", default="dataset/eval/variant_batch1_analogs.csv")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                         help="Longest base text allowed (the first 20 ranged 265-1104 words)")
    parser.add_argument("--texts-output", default="dataset/eval/variant_batch2_base_texts.txt")
    parser.add_argument("--no-analogs", action="store_true")
    args = parser.parse_args()

    main_q = load_jsonl(MAIN_QUERIES, "query_id")
    test_cases = load_jsonl(TEST_CASES, "case_id")
    existing_bases = sorted({json.loads(l)["base_query_id"] for l in open(VARIANT_QUERIES, "r", encoding="utf-8")})
    existing = [main_q[b] for b in existing_bases]

    pool = [c for c in build_candidates(main_q, test_cases, set(existing_bases)) if passes_gates(c, args.max_words)]
    print(f"{len(pool)} candidates pass the gates (>= {MIN_APPLICANT_MENTIONS} applicant gender mentions, not gender-intrinsic, "
          f"not sensitive-topic, <= {args.max_words} words)")
    print("  pool by article:", dict(Counter(c["article"] for c in pool)), "| by gender:", dict(Counter(c["gender"] for c in pool)))

    quotas = article_quotas(main_q, [e["article"] for e in existing], args.n_new)
    n_male_total = sum(e["gender"] == "Male" for e in existing)
    half = (len(existing) + args.n_new) // 2
    gender_need = {"Male": half - n_male_total, "Female": half - (len(existing) - n_male_total)}
    country_total = Counter(e["country"] for e in existing)
    capped = cap_quotas(quotas, pool, main_q)
    if capped != quotas:
        print("Corpus-proportional article quotas:", quotas)
        print("Adjusted to what short cases allow:", capped)
    quotas = capped
    print("Article quotas for the new batch:", quotas, "| gender need:", gender_need)

    rng = random.Random(args.seed)
    chosen, attempts = draw(pool, quotas, gender_need, country_total, rng)
    if chosen is None:
        raise SystemExit("No draw satisfied every quota and cap; raise --max-words or relax the caps/quotas")
    print(f"Draw satisfied all quotas and caps on attempt {attempts} (seed {args.seed})")

    taken = {c["query_id"] for c in chosen}
    rest = [c for c in pool if c["query_id"] not in taken]
    rng.shuffle(rest)
    reserves, rcountry = [], Counter(country_total) + Counter(c["country"] for c in chosen)
    for c in rest:
        if len(reserves) == args.reserves:
            break
        if rcountry[c["country"]] < MAX_TOTAL_PER_COUNTRY:
            reserves.append(c)
            rcountry[c["country"]] += 1

    rows = [dict(c, role="primary", order=i) for i, c in enumerate(sorted(chosen, key=lambda c: (c["article"], c["gender"])), 1)]
    rows += [dict(c, role="reserve", order=i) for i, c in enumerate(reserves, 1)]

    if not args.no_analogs:
        train = load_train(TRAIN_METADATA)
        attach_analogs(rows, main_q, train)
        first_rows = [{"query_id": e["query_id"], "source_case_id": e["source_case_id"], "gender": e["gender"],
                       "article": e["article"], "country": e["country"], "year": e["year"]} for e in existing]
        attach_analogs(first_rows, main_q, train)
        with open(args.first_set_output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(first_rows[0].keys()))
            writer.writeheader()
            writer.writerows(first_rows)
        print(f"Wrote analog covariate for the first {len(first_rows)} bases to {args.first_set_output}")

    fieldnames = ["role", "order", "query_id", "source_case_id", "gender", "article", "country", "year", "source_words",
                  "applicant_gender_mentions", "nrc_intensity"]
    if not args.no_analogs:
        fieldnames += ["analog_case_id", "analog_title", "analog_similarity", "analog_same_country", "analog_shared_article"]
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(chosen)} primary + {len(reserves)} reserve bases to {args.output}")

    with open(args.texts_output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("=" * 100 + "\n")
            f.write(f"{r['role'].upper()} {r['order']}  |  {r['query_id']}  |  source applicant: {r['gender']}  |  Art. {r['article']}  |  "
                    f"{r['country']}  |  {r['year']}  |  {r['source_words']} words\n")
            f.write("=" * 100 + "\n\n")
            f.write(r["base_text"].strip() + "\n\n")
    print(f"Wrote base texts for {len(rows)} bases to {args.texts_output}")

    print("\nCombined first set + new primaries:")
    combined = existing + chosen
    print("  gender :", dict(Counter(c["gender"] for c in combined)))
    print("  article:", dict(Counter(c["article"] for c in combined)))
    print("  country:", dict(Counter(c["country"] for c in combined).most_common(6)), "...")


if __name__ == "__main__":
    main()
