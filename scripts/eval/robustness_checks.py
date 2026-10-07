import argparse
import glob
import hashlib
import json
import math
import os
import re
from datetime import datetime
from itertools import combinations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binomtest, chi2, ncx2, norm, wilcoxon

from scripts.eval.bias_statistics import BiasAuditor, holm_bonferroni, wilson_ci
from scripts.eval.llm_output_bias import LLMOutputBiasAuditor
from scripts.eval.multi_run_stability import RUN_STATS

# Robustness checks of the Phase E findings (read-only; nothing in the pipeline or in logs/bias_audit changes):
#   r1  topic control: does query gender -> retrieved gender survive stratifying by legal topic and country?
#   r2  gender-relevant cases: do the arms lose precedents where gender is legally relevant?
#   r3  detectable effects: how large an effect could the null results have missed?
#   r3b the LLM with identical cases: does its answer change with the pronoun alone? (needs run_fixed_case_test.py)
#   r4  the pronoun flip on ~300 automatic he/she pairs, baseline and arms (needs run_swap_set_test.py)
# Output: logs/robustness/<check>.json  (question, method, inputs with sha256, results)

OUT_DIR = "logs/robustness"
MITIGATION_ROOT = "logs/mitigation"
BASELINE = "baseline"
TRAIN_PATH = "dataset/train_with_metadata.jsonl"
TEST_PATH = "dataset/test.jsonl"
MAIN_QUERIES_PATH = "dataset/eval/audit_queries_main.jsonl"
HARNESS_QUERIES_PATH = "dataset/eval/retrieval_harness_queries.jsonl"
PHASE_E_PREDICT_GLOB = "logs/bias_audit/predict/run_*.jsonl"
PHASE_E_MULTI_RUN = "logs/bias_audit/stats_multi_run/results.json"
FIXED_CASES_DIR = "logs/robustness/fixed_cases"
FIXED_TYPES = ("neutral", "male", "female")
SWAP_SET_DIR = "logs/robustness/swap_set"
SWAP_PAIRS_PATH = f"{SWAP_SET_DIR}/pairs.jsonl"
SWAP_ARMS = (BASELINE, "blind_query", "pcf", "leace")

ALPHA, POWER = 0.05, 0.80

# Sex/gender-related legal topics; Article 14 (discrimination) counts on its own
GENDER_TOPIC = re.compile(r"\b(pregnan\w*|maternity|abortion|rape[sd]?|sexual\w*|sex discrimination|domestic violence|"
                          r"gender|transsexual\w*|female genital)\b", re.IGNORECASE)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retrieval_logs():
    paths = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(f"{MITIGATION_ROOT}/*/retrieval.jsonl")}
    return {arm: paths[arm] for arm in sorted(paths, key=lambda a: (a != BASELINE, a))}


def save(name, question, method, inputs, results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    out = {"check": name, "question": question, "created": datetime.now().isoformat(timespec="seconds"),
           "method": method, "inputs": {p: file_sha256(p) for p in inputs}, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


# ---- statistics ----

# Cochran-Mantel-Haenszel over 2x2 tables (a, b, c, d) = (exposed & outcome, exposed & not, unexposed & outcome,
# unexposed & not): common odds ratio with the Robins-Breslow-Greenland 95% CI, and the CMH test with continuity
# correction (as R's mantelhaen.test). One table gives the crude (unstratified) result.
def mantel_haenszel(tables):
    tables = [t for t in tables if sum(t) > 1]
    sum_a = sum_e = sum_v = 0.0
    R = S = PR = PS_QR = QS = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        sum_a += a
        sum_e += (a + b) * (a + c) / n
        sum_v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        r, s, p, q = a * d / n, b * c / n, (a + d) / n, (b + c) / n
        R, S, PR, PS_QR, QS = R + r, S + s, PR + p * r, PS_QR + p * s + q * r, QS + q * s
    stat = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_v if sum_v else None
    out = {"n_strata": len(tables), "n": int(sum(sum(t) for t in tables)),
           "cmh_statistic": stat, "p": float(chi2.sf(stat, 1)) if stat is not None else None,
           "odds_ratio": None, "odds_ratio_95ci": None}
    if R and S:
        log_or = math.log(R / S)
        se = math.sqrt(PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S))
        out["odds_ratio"] = R / S
        out["odds_ratio_95ci"] = [math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)]
    return out


# Smallest Cramer's V a chi-square test of independence detects with the given power (noncentral chi-square);
# for a 2 x k table V equals Cohen's w
def chi2_mde(n, dof, alpha=ALPHA, power=POWER):
    crit = chi2.ppf(1 - alpha, dof)
    return brentq(lambda w: ncx2.sf(crit, dof, n * w * w) - power, 1e-6, 5.0)


# Smallest mean shift a two-sided paired test detects with the given power: (z_{1-alpha/2} + z_power) * SD / sqrt(n),
# with n scaled by the Wilcoxon test's minimum asymptotic relative efficiency (0.864), so the bound is conservative.
# (A bootstrap location shift is not used: half the bases have a difference of exactly 0, and shifting them all
# turns them into small positives the test detects too easily.)
WILCOXON_MIN_ARE = 0.864


def wilcoxon_mde(diffs, alpha=ALPHA, power=POWER):
    diffs = np.asarray(diffs, dtype=float)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * diffs.std(ddof=1) / math.sqrt(len(diffs) * WILCOXON_MIN_ARE))


# ---- r1: topic control ----

STRATA = {
    "article": lambda m: m["article"],
    "country": lambda m: m["country"],
    "article_x_country": lambda m: f"{m['article']}|{m['country']}",
}


def gender_tables(auditor, stratum_of, top_n):
    tables = {}
    for e in auditor.entries:
        meta = auditor.query_meta.get(e["query_id"])
        if not meta or meta["source"] != "main" or meta["gender"] not in ("Male", "Female"):
            continue
        t = tables.setdefault(stratum_of(meta), [0, 0, 0, 0])
        row = 0 if meta["gender"] == "Female" else 2
        for c in e["retrieved_cases"][:top_n]:
            t[row + (c["gender"] != "Female")] += 1
    return tables


def r1_topic_control():
    logs = retrieval_logs()
    results = {}
    for arm, path in logs.items():
        auditor = BiasAuditor(audit_log_path=path, out_dir=OUT_DIR)
        results[arm] = {}
        for unit, top_n in (("top1", 1), ("top3", 3)):
            by_article = gender_tables(auditor, STRATA["article"], top_n)
            pooled = [sum(t[i] for t in by_article.values()) for i in range(4)]
            block = {"crude": mantel_haenszel([pooled])}
            for name, stratum_of in STRATA.items():
                block[f"stratified_by_{name}"] = mantel_haenszel(list(gender_tables(auditor, stratum_of, top_n).values()))
            block["per_article"] = {k: {"table_female_query": t[:2], "table_male_query": t[2:]}
                                    for k, t in sorted(by_article.items())}
            results[arm][unit] = block

    print("\n" + "=" * 92)
    print("R1 TOPIC CONTROL: odds that a retrieved case is Female, female vs male query (main set)")
    print("=" * 92)
    print(f"{'arm':<14}{'unit':<6}{'crude OR':>16}{'| article':>18}{'| country':>18}{'| art x country':>20}")
    for arm, units in results.items():
        for unit, block in units.items():
            cells = [block["crude"]] + [block[f"stratified_by_{s}"] for s in STRATA]
            fmt = [f"{c['odds_ratio']:.2f} (p={c['p']:.0e})" if c["odds_ratio"] else "n/a" for c in cells]
            print(f"{arm:<14}{unit:<6}{fmt[0]:>16}{fmt[1]:>18}{fmt[2]:>18}{fmt[3]:>20}")

    save("r1_topic_control",
         "Does the association between query gender and retrieved-case gender survive controlling for legal topic "
         "(key article) and country, or is it explained by them?",
         {"population": "main-set audit queries with a Male or Female applicant (Phase E scope)",
          "outcome": "retrieved case labelled Female (current validated labels, as Phase E)",
          "exposure": "query from a Female-applicant case (vs Male)",
          "units": "top1 = one retrieved case per query (independent); top3 = all three, as the Phase E chi-square "
                   "(clustered within query, so p-values are optimistic)",
          "test": "Cochran-Mantel-Haenszel with continuity correction; Mantel-Haenszel common odds ratio with "
                  "Robins-Breslow-Greenland 95% CI; crude = one pooled table",
          "reading": "an odds ratio that stays near the crude value after stratifying means topic/country do not "
                     "explain the association; one that moves toward 1 means they do"},
         list(logs.values()) + [MAIN_QUERIES_PATH], results)


# ---- r2: gender-relevant cases ----

def topic_flags(record, text):
    articles = {str(a) for a in (record.get("violated_articles") or []) + (record.get("allegedly_violated_articles") or [])}
    keywords = sorted({k.lower() for k in GENDER_TOPIC.findall(text)})
    return {"article_14": "14" in articles, "keywords": keywords, "relevant": "14" in articles or bool(keywords)}


def load_cases(path, ids=None):
    cases = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ids is None or r["case_id"] in ids:
                cases[r["case_id"]] = r
    return cases


def r2_gender_relevant():
    logs = retrieval_logs()
    rows = {arm: {r["query_id"]: r for r in load_jsonl(p)} for arm, p in logs.items()}
    main = load_jsonl(MAIN_QUERIES_PATH)
    sources = load_cases(TEST_PATH, {q["source_case_id"] for q in main})
    retrieved_ids = {c["case_id"] for arm_rows in rows.values() for q in main for c in arm_rows[q["query_id"]]["retrieved_cases"]}
    retrieved = load_cases(TRAIN_PATH, retrieved_ids)
    case_relevant = {cid: topic_flags(r, " ".join(r["facts"]))["relevant"] for cid, r in retrieved.items()}

    queries = []
    for q in main:
        flags = topic_flags(sources[q["source_case_id"]], q["query_text"])
        entry = {"query_id": q["query_id"], "gender": q["gender"], "article": q["article"], **flags, "arms": {}}
        for arm in logs:
            top = [c["case_id"] for c in rows[arm][q["query_id"]]["retrieved_cases"]]
            entry["arms"][arm] = {"top3": top, "relevant_in_top3": sum(case_relevant[c] for c in top)}
        queries.append(entry)

    def retention(group, arm):
        vals = [q["arms"][arm]["relevant_in_top3"] / 3 for q in group]
        return {"n_queries": len(vals), "mean_share_relevant_in_top3": float(np.mean(vals)) if vals else None}

    relevant = [q for q in queries if q["relevant"]]
    other = [q for q in queries if not q["relevant"]]
    audit = {}
    for arm in logs:
        block = {"gender_relevant": retention(relevant, arm), "other": retention(other, arm)}
        if arm != BASELINE:
            d = [(q["arms"][arm]["relevant_in_top3"] - q["arms"][BASELINE]["relevant_in_top3"]) / 3 for q in relevant]
            block["change_vs_baseline_gender_relevant"] = {
                "mean": float(np.mean(d)), "queries_lost": sum(x < 0 for x in d), "queries_gained": sum(x > 0 for x in d),
                "p_wilcoxon": float(wilcoxon(d).pvalue) if any(d) else None}
        audit[arm] = block

    harness_queries = load_jsonl(HARNESS_QUERIES_PATH)
    harness_relevant = {h["query_id"] for h in harness_queries
                        if topic_flags({"violated_articles": h["violated_articles"]}, h["query_text"])["relevant"]}
    harness = {}
    for arm, path in logs.items():
        with open(os.path.join(os.path.dirname(path), "harness.json"), "r", encoding="utf-8") as f:
            per_query = json.load(f)["per_query"]
        sub = [p for p in per_query if p["query_id"] in harness_relevant]
        hits = sum(p["eval_rank"] == 1 for p in sub)
        harness[arm] = {"n_queries": len(sub), "recall@1": hits / len(sub) if sub else None,
                        "recall@1_wilson_95ci": wilson_ci(hits, len(sub)) if sub else None,
                        "missed": [p["query_id"] for p in sub if p["eval_rank"] != 1]}

    print("\n" + "=" * 92)
    print(f"R2 GENDER-RELEVANT CASES: {len(relevant)}/{len(queries)} main queries "
          f"({sum(q['article_14'] for q in relevant)} with Article 14), {len(harness_relevant)}/100 harness queries")
    print("=" * 92)
    print(f"{'arm':<14}{'relevant share (rel. q)':>26}{'(other q)':>12}{'lost/gained':>14}{'harness R@1 (rel.)':>22}")
    for arm in logs:
        a, h = audit[arm], harness[arm]
        ch = a.get("change_vs_baseline_gender_relevant")
        lg = f"{ch['queries_lost']}/{ch['queries_gained']}" if ch else "-"
        r1 = f"{h['recall@1']:.3f} (n={h['n_queries']})" if h["recall@1"] is not None else "n/a"
        print(f"{arm:<14}{a['gender_relevant']['mean_share_relevant_in_top3']:>26.3f}"
              f"{a['other']['mean_share_relevant_in_top3']:>12.3f}{lg:>14}{r1:>22}")

    save("r2_gender_relevant",
         "Where gender is legally relevant, do the mitigation arms lose the relevant precedents?",
         {"gender_relevant_query": "source case alleges or was found to violate Article 14, or the query text mentions "
                                   f"a sex/gender-related topic (regex: {GENDER_TOPIC.pattern})",
          "gender_relevant_case": "retrieved case alleges/violates Article 14, or its facts match the same regex",
          "audit_measure": "share of the top-3 retrieved cases that are gender-relevant, per query; paired against the "
                           "baseline (Wilcoxon); queries without the topic are the control",
          "harness_measure": "recall@1 of the B.1 harness on its gender-relevant queries (ground truth = source case)",
          "note": "small subsets; descriptive, read with the per-query list"},
         list(logs.values()) + [os.path.join(os.path.dirname(p), "harness.json") for p in logs.values()]
         + [MAIN_QUERIES_PATH, HARNESS_QUERIES_PATH],
         {"audit": audit, "harness": harness, "queries": relevant})


# ---- r3: detectable effects ----

def r3_detectable_effects():
    results, inputs = {}, []

    variant_path = os.path.join(RUN_STATS["A"], "paired_variant_tests.json")
    with open(variant_path, "r", encoding="utf-8") as f:
        variants = json.load(f)
    inputs.append(variant_path)
    tone, pronoun = variants["paired_framing_neutral_vs_emotional"], variants["paired_framing_male_vs_female"]
    results["tone_retrieval_flip"] = {
        "observed": f"{tone['top1_flips']}/{tone['n_pairs']}",
        "flip_rate_95ci": wilson_ci(tone["top1_flips"], tone["n_pairs"]),
        "pronoun_flip_for_comparison": f"{pronoun['top1_flips']}/{pronoun['n_pairs']}",
        "reading": "upper CI bound = the largest tone flip rate compatible with the data"}

    gen = {}
    for run, stats_dir in RUN_STATS.items():
        path = os.path.join(stats_dir, "master_results.json")
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)["tests"]["generation_stage"]
        inputs.append(path)
        gen[run] = {"n": g["n"], "dof": g["dof"], "observed_cramers_v": g["effect_size"], "p_raw": g["p_raw"],
                    "mde_cramers_v": chi2_mde(g["n"], g["dof"])}
    results["generation_stage_gender"] = {
        "per_run": gen, "reading": f"V the test detects with {POWER:.0%} power at alpha {ALPHA} (before Holm)"}

    with open(PHASE_E_MULTI_RUN, "r", encoding="utf-8") as f:
        split = json.load(f)["input_similarity_split"]["male_vs_female"]["identical input"]
    inputs.append(PHASE_E_MULTI_RUN)
    results["llm_cited_change_identical_input"] = {
        "observed": f"{split['rows_differ']}/{split['rows']}",
        "change_rate_95ci": wilson_ci(split["rows_differ"], split["rows"]),
        "reading": "few he/she bases reach the LLM with identical input under the baseline, so this bound is wide"}

    predict_logs = sorted(glob.glob(PHASE_E_PREDICT_GLOB))
    inputs += predict_logs
    llm = LLMOutputBiasAuditor(predict_logs)
    articles = {}
    for ta, tb in (("male", "female"), ("neutral", "emotional")):
        change, acc = [], []
        for b in llm.bases:
            pc, na, nb = llm.pair_change(b, ta, tb), llm.repeat_change(b, ta), llm.repeat_change(b, tb)
            if None not in (pc, na, nb):
                change.append(pc - (na + nb) / 2)
            x, y = llm.accuracy(b, ta), llm.accuracy(b, tb)
            if x is not None and y is not None:
                acc.append(y["jaccard"] - x["jaccard"])
        articles[f"{ta}_vs_{tb}"] = {
            "prediction_change_above_noise": {"n_bases": len(change), "observed_mean": float(np.mean(change)),
                                              "mde_mean_shift": wilcoxon_mde(change)},
            "jaccard_accuracy_difference": {"n_bases": len(acc), "observed_mean": float(np.mean(acc)),
                                            "mde_mean_shift": wilcoxon_mde(acc)}}
    results["predicted_articles"] = {
        "pairs": articles, "reading": "1 - Jaccard units (change) and Jaccard units (accuracy); a true mean shift "
                                      f"at least this large would have been detected with {POWER:.0%} power"}

    print("\n" + "=" * 92)
    print(f"R3 DETECTABLE EFFECTS (alpha {ALPHA}, power {POWER:.0%})")
    print("=" * 92)
    t = results["tone_retrieval_flip"]
    print(f"Tone flip: {t['observed']}, 95% CI {t['flip_rate_95ci'][0]:.1%}-{t['flip_rate_95ci'][1]:.1%} "
          f"(pronoun flip {t['pronoun_flip_for_comparison']})")
    for run, g in gen.items():
        print(f"Generation-stage gender, run {run}: observed V {g['observed_cramers_v']:.3f}, "
              f"detectable V {g['mde_cramers_v']:.3f} (n={g['n']})")
    c = results["llm_cited_change_identical_input"]
    print(f"LLM cited change on identical he/she input: {c['observed']}, 95% CI "
          f"{c['change_rate_95ci'][0]:.1%}-{c['change_rate_95ci'][1]:.1%}")
    for pair, v in articles.items():
        ch, ac = v["prediction_change_above_noise"], v["jaccard_accuracy_difference"]
        print(f"Predicted articles {pair}: change above noise {ch['observed_mean']:+.3f} (detectable {ch['mde_mean_shift']:.3f}), "
              f"Jaccard difference {ac['observed_mean']:+.3f} (detectable {ac['mde_mean_shift']:.3f})")

    save("r3_detectable_effects",
         "How large an effect could each Phase E null result have missed?",
         {"proportions": "Wilson 95% interval; the upper bound is the largest rate compatible with the data",
          "chi_square": "smallest Cramer's V detected with the stated power (noncentral chi-square)",
          "paired_wilcoxon": "smallest mean shift detected with the stated power: (z_{1-alpha/2} + z_power) * SD / "
                             f"sqrt(n * {WILCOXON_MIN_ARE}), the Wilcoxon minimum relative efficiency (conservative)",
          "alpha": ALPHA, "power": POWER},
         inputs, results)


# ---- r3b: the LLM with identical cases, only the pronoun changed ----

def cited_change_test(llm, ta, tb):
    def cited(run, b, t):
        return frozenset(llm.runs[run][llm.qid[(b, t)]]["cited_case_ids"] or [])

    wording, repeat = [], []
    for b in llm.bases:
        wording.append(np.mean([cited(r, b, ta) != cited(r, b, tb) for r in llm.runs]))
        repeat.append(np.mean([cited(r1, b, t) != cited(r2, b, t) for t in (ta, tb)
                               for r1, r2 in combinations(llm.runs, 2)]))
    diffs = [w - r for w, r in zip(wording, repeat)]
    rows = len(llm.bases) * len(llm.runs)
    differ = int(round(sum(wording) * len(llm.runs)))
    return {"rows_differ": differ, "rows": rows, "change_rate": differ / rows, "change_wilson_95ci": wilson_ci(differ, rows),
            "repeat_noise": float(np.mean(repeat)),
            "p_wilcoxon_change_above_noise": float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0}


def r3b_llm_pronoun_fixed_cases():
    logs = sorted(glob.glob(f"{FIXED_CASES_DIR}/run_*.jsonl"))
    llm = LLMOutputBiasAuditor(logs)
    identical = all(len({tuple(c["case_id"] for c in rows[llm.qid[(b, t)]]["retrieved_cases"]) for t in FIXED_TYPES}) == 1
                    for rows in llm.runs.values() for b in llm.bases)

    pairs, p_raw = {}, {}
    for ta, tb in (("male", "female"), ("neutral", "male"), ("neutral", "female")):
        key = f"{ta}_vs_{tb}"
        change, accuracy = llm.change_test(ta, tb, llm.bases), llm.accuracy_test(ta, tb, llm.bases)
        pairs[key] = {"cited_case": cited_change_test(llm, ta, tb), "predicted_articles_change": change,
                      "predicted_articles_accuracy": accuracy}
        p_raw[f"{key}_cited"] = pairs[key]["cited_case"]["p_wilcoxon_change_above_noise"]
        p_raw[f"{key}_articles_change"] = change["p_wilcoxon_change_above_noise"]
        p_raw[f"{key}_key_hit"] = accuracy["key_hit"]["p"]
    holm = holm_bonferroni(p_raw)

    print("\n" + "=" * 92)
    print(f"R3b LLM WITH IDENTICAL CASES ({len(logs)} runs, {len(llm.bases)} bases; identical cases in every row: {identical})")
    print("=" * 92)
    for key, v in pairs.items():
        c, ch, acc = v["cited_case"], v["predicted_articles_change"], v["predicted_articles_accuracy"]
        ta, tb = key.split("_vs_")
        print(f"{key:<20} cited differs {c['rows_differ']}/{c['rows']} ({c['change_rate']:.1%}) vs noise {c['repeat_noise']:.1%} "
              f"p={c['p_wilcoxon_change_above_noise']:.3f} | articles change {ch['mean_change_between_wordings']:.2f} "
              f"vs noise {ch['mean_change_same_query_repeat']:.2f} p={ch['p_wilcoxon_change_above_noise']:.4f} | "
              f"key hit {acc['key_hit'][f'{ta}_correct']} vs {acc['key_hit'][f'{tb}_correct']} p={acc['key_hit']['p']:.3f}")
    print("\nHolm-Bonferroni:")
    for name, p in sorted(holm.items(), key=lambda kv: kv[1]):
        print(f"  {name:<34} p={p_raw[name]:.4f}  corrected={p:.4f}  {'YES' if p < ALPHA else 'no'}")

    save("r3b_llm_pronoun_fixed_cases",
         "With the retrieved cases held identical, does the LLM's answer change with the applicant's pronoun?",
         {"design": "per base, the frozen system retrieves once with the hand-written neutral variant; the same cases in "
                    "the same order are answered with the neutral, male and female wording (scripts/eval/run_fixed_case_test.py)",
          "noise": "the same query across the 4 runs (all pairs of runs)",
          "tests": "Wilcoxon of per-base change minus repeat noise (cited case set; predicted articles, 1 - Jaccard); "
                   "exact McNemar on the key-article hit (majority over runs); Holm-Bonferroni over the family",
          "identical_cases_in_every_row": identical},
         logs, {"pairs": pairs, "p_raw": p_raw, "p_holm": holm})


# ---- R4: the pronoun flip on a larger automatic swap set ----

def rate_ci(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson_95ci": [lo, hi]}


def exact_mcnemar(b, c):
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else None


def swap_set_arm(rows, pairs, gender):
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["variant_type"]] = r
    flips, top3_changed, identical_query, female_only = {}, 0, 0, {"male": 0, "female": 0}
    for p in pairs:
        m, f = by_pair[p["pair_id"]]["male"], by_pair[p["pair_id"]]["female"]
        top_m, top_f = m["retrieved_cases"][0]["case_id"], f["retrieved_cases"][0]["case_id"]
        flips[p["pair_id"]] = top_m != top_f
        top3_changed += [c["case_id"] for c in m["retrieved_cases"]] != [c["case_id"] for c in f["retrieved_cases"]]
        identical_query += m.get("rewritten_query", "m") == f.get("rewritten_query", "f")
        m_female, f_female = gender.get(top_m) == "Female", gender.get(top_f) == "Female"
        if m_female != f_female:
            female_only["male" if m_female else "female"] += 1

    def flip_rate(ids):
        k = sum(flips[i] for i in ids)
        return rate_ci(k, len(ids))

    groups = {"all": [p["pair_id"] for p in pairs],
              "source_male": [p["pair_id"] for p in pairs if p["source_gender"] == "Male"],
              "source_female": [p["pair_id"] for p in pairs if p["source_gender"] == "Female"],
              "swap_round_trip_exact": [p["pair_id"] for p in pairs if p["round_trip_identical"]]}
    return {"top1_flip": {g: flip_rate(ids) for g, ids in groups.items()},
            "top3_order_changed": rate_ci(top3_changed, len(pairs)),
            "mcnemar_top1_female": {"top1_female_only_with_he": female_only["male"],
                                    "top1_female_only_with_she": female_only["female"],
                                    "p_exact": exact_mcnemar(female_only["male"], female_only["female"])},
            "identical_query_after_arm": identical_query,
            "flips": flips}


def r4_large_swap_set():
    pairs = load_jsonl(SWAP_PAIRS_PATH)
    gender = BiasAuditor._load_case_gender()
    logs = {arm: p for arm in SWAP_ARMS if os.path.exists(p := f"{SWAP_SET_DIR}/{arm}.jsonl")}
    hand = {arm: json.load(open(f"{MITIGATION_ROOT}/{arm}/results.json", encoding="utf-8"))["pronoun_flip"]
            for arm in logs if os.path.exists(f"{MITIGATION_ROOT}/{arm}/results.json")}

    arms = {}
    for arm, path in logs.items():
        rows = load_jsonl(path)
        assert len(rows) == 2 * len(pairs), f"{path}: {len(rows)} rows, expected {2 * len(pairs)} (rerun the runner)"
        arms[arm] = swap_set_arm(rows, pairs, gender)

    # Paired comparison with the baseline: exact McNemar on the per-pair flip indicator
    for arm, res in arms.items():
        if arm == BASELINE or BASELINE not in arms:
            continue
        base = arms[BASELINE]["flips"]
        fixed = sum(base[i] and not res["flips"][i] for i in base)
        new = sum(res["flips"][i] and not base[i] for i in base)
        res["vs_baseline"] = {"flips_removed": fixed, "flips_added": new, "p_exact_mcnemar": exact_mcnemar(fixed, new)}
    holm = holm_bonferroni({arm: r["mcnemar_top1_female"]["p_exact"] for arm, r in arms.items()
                            if r["mcnemar_top1_female"]["p_exact"] is not None})

    n_female = sum(p["source_gender"] == "Female" for p in pairs)
    print("\n" + "=" * 92)
    print(f"R4 PRONOUN FLIP, AUTOMATIC SWAP SET ({len(pairs)} pairs: {n_female} female / {len(pairs) - n_female} male "
          f"source; swap round trip exact {sum(p['round_trip_identical'] for p in pairs)}/{len(pairs)})")
    print("=" * 92)
    print(f"{'arm':<13}{'top-1 flips':>20}{'95% CI':>16}{'hand set (40)':>15}{'female top-1 he/she':>21}"
          f"{'McNemar p':>11}{'Holm':>8}")
    for arm, r in arms.items():
        a, mc = r["top1_flip"]["all"], r["mcnemar_top1_female"]
        h = hand.get(arm)
        hand_txt = f"{h['top1_flips']}/{h['n_pairs']}" if h else "n/a"
        p_txt = f"{mc['p_exact']:.2g}" if mc["p_exact"] is not None else "n/a"
        holm_txt = f"{holm[arm]:.2g}" if arm in holm else "n/a"
        print(f"{arm:<13}{a['k']:>8}/{a['n']} ({a['rate']:.1%}){a['wilson_95ci'][0]:>8.1%}-{a['wilson_95ci'][1]:.1%}"
              f"{hand_txt:>15}{mc['top1_female_only_with_he']:>12} / {mc['top1_female_only_with_she']:<6}"
              f"{p_txt:>11}{holm_txt:>8}")
    for arm, r in arms.items():
        g = r["top1_flip"]
        extra = (f" | vs baseline: {r['vs_baseline']['flips_removed']} removed, {r['vs_baseline']['flips_added']} added, "
                 f"p={r['vs_baseline']['p_exact_mcnemar']:.2g}") if "vs_baseline" in r else ""
        print(f"  {arm:<11} by source: male {g['source_male']['rate']:.1%}, female {g['source_female']['rate']:.1%}; "
              f"exact-swap subset {g['swap_round_trip_exact']['rate']:.1%}; top-3 order changed "
              f"{r['top3_order_changed']['rate']:.1%}; identical query after arm {r['identical_query_after_arm']}{extra}")

    for r in arms.values():
        r["flipped_pairs"] = sorted(i for i, f in r.pop("flips").items() if f)
    save("r4_large_swap_set",
         "Does the pronoun flip, and its removal by the arms, hold on a larger, automatically built set of he/she pairs?",
         {"design": "test cases outside the index with a single dominant applicant gender (scripts/eval/run_swap_set_test.py "
                    "gates); query built as the B.1 harness queries; twin = swap_gender() (two-sided CDA swap, third "
                    "parties swapped too); retrieval only, each arm on both versions",
          "tests": "top-1 flip rate with Wilson CI; exact McNemar on 'top-1 case is Female-gendered' (he vs she), Holm "
                   "over arms; exact McNemar of each arm's per-pair flips against the baseline's",
          "gender_labels": "current validated case labels (train_with_metadata.jsonl), as in Phase E",
          "hand_set_reference": "logs/mitigation/<arm>/results.json pronoun_flip (the 40 hand-written bases)"},
         [SWAP_PAIRS_PATH, *logs.values()],
         {"n_pairs": len(pairs), "n_source_female": n_female, "arms": arms, "mcnemar_holm": holm,
          "hand_set": {arm: {k: h[k] for k in ("top1_flips", "n_pairs", "mcnemar_exact_p")} for arm, h in hand.items()}})


CHECKS = {"r1": r1_topic_control, "r2": r2_gender_relevant, "r3": r3_detectable_effects,
          "r3b": r3b_llm_pronoun_fixed_cases, "r4": r4_large_swap_set}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", default="r1,r2,r3", help=f"Comma-separated: {', '.join(CHECKS)}")
    args = parser.parse_args()
    for name in args.checks.split(","):
        CHECKS[name.strip()]()


if __name__ == "__main__":
    main()
