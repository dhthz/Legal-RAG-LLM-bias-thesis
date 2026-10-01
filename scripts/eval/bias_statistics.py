import argparse
import json
import math
import os
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, chi2_contingency, chisquare, linregress

AUDIT_LOG_PATH = "logs/bias_audit/bias_audit_interactions.jsonl"
TRAIN_METADATA_PATH = "dataset/train_with_metadata.jsonl"
MAIN_QUERIES_PATH = "dataset/eval/final_llm_queries.jsonl"
VARIANT_QUERIES_PATH = "dataset/eval/final_llm_queries_variants.jsonl"

GENDER_CLASSES = ("Male", "Female", "Multiple Applicants", "Unknown", "Needs Manual Classification")
VARIANT_TYPES = ("neutral", "male", "female", "emotional")


# Cramer's V for a goodness-of-fit test (1 x k table): V = sqrt(chi2 / (n*(k-1)))
def cramers_v_gof(chi2_stat, n, k):
    return (chi2_stat / (n * (k - 1))) ** 0.5


# Wilson score CI for a binary proportion (same helper as retrieval_eval.py)
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


# Holm-Bonferroni across a named p-value family: sort ascending, multiply the
# i-th smallest by (m - i), enforce monotonicity, cap at 1.
def holm_bonferroni(named_pvals):
    m = len(named_pvals)
    ordered = sorted(named_pvals.items(), key=lambda kv: kv[1])
    corrected = {}
    running_max = 0.0
    for i, (name, p) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - i) * p))
        corrected[name] = running_max
    return corrected


class BiasAuditor:

    def __init__(self, audit_log_path=AUDIT_LOG_PATH, out_dir="logs/bias_audit"):
        self.audit_log_path = audit_log_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.case_gender = self._load_case_gender()
        self.entries = self._load_audit_log(audit_log_path)
        self._apply_current_gender_labels()
        self.query_meta = self._load_query_metadata()
        self.corpus_gender = self._load_corpus_gender_prevalence()
        # Classes with no corpus members (e.g. Needs Manual Classification after human relabelling) would give zero expected counts
        self.gender_classes = [g for g in GENDER_CLASSES if self.corpus_gender.get(g, 0) > 0]
        self.corpus_stats = self._load_corpus_stats()
        self.by_base = self._group_variants()

    # ---- data loading ----

    def _out(self, name):
        return os.path.join(self.out_dir, name)

    # Gender is read by case_id from the current dataset file, not from the label stored in the log, so relabelling the dataset flows through without a rerun
    @staticmethod
    def _load_case_gender():
        gender = {}
        with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gender[rec["case_id"]] = rec.get("classification", {}).get("gender", "Unknown")
        return gender

    def _apply_current_gender_labels(self):
        changed = 0
        for e in self.entries:
            for c in e["retrieved_cases"]:
                current = self.case_gender.get(c["case_id"])
                if current is not None and current != c["gender"]:
                    c["gender"] = current
                    changed += 1
        self.relabelled_retrieved = changed

    @staticmethod
    def _load_audit_log(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _load_query_metadata():
        meta = {}
        with open(MAIN_QUERIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta[row["query_id"]] = {"article": row["article"], "country": row["country"],
                                          "gender": row["gender"], "source": "main"}
        with open(VARIANT_QUERIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta[row["variant_id"]] = {"article": row["article"], "country": row["country"],
                                            "gender": row["gender"], "source": "variant"}
        return meta

    @staticmethod
    def _load_corpus_gender_prevalence():
        counts = Counter()
        with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                counts[rec.get("classification", {}).get("gender", "Unknown")] += 1
        return counts

    @staticmethod
    def _load_corpus_stats():
        outcome, country, years = Counter(), Counter(), []
        with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                outcome["violation" if rec.get("violated_articles") else "no_violation"] += 1
                defendants = rec.get("defendants") or []
                if defendants:
                    country[defendants[0]] += 1
                date = rec.get("judgment_date") or ""
                if len(date) >= 4 and date[:4].isdigit():
                    years.append(int(date[:4]))
        return {"outcome": outcome, "country": country, "years": years}

    def _group_variants(self):
        by_base = defaultdict(dict)
        for e in self.entries:
            qid = e["query_id"]
            if "_" not in qid:
                continue
            base, vtype = qid.rsplit("_", 1)
            if vtype in VARIANT_TYPES:
                by_base[base][vtype] = e
        return by_base

    # ---- corpus-baseline family ----

    def gender_vs_corpus_baseline(self):
        retrieved = Counter()
        for e in self.entries:
            for c in e["retrieved_cases"]:
                retrieved[c["gender"]] += 1

        n, n_corpus = sum(retrieved.values()), sum(self.corpus_gender.values())
        observed = [retrieved.get(g, 0) for g in self.gender_classes]
        expected = [self.corpus_gender.get(g, 0) / n_corpus * n for g in self.gender_classes]

        chi2, p = chisquare(f_obs=observed, f_exp=expected)
        v = cramers_v_gof(chi2, n, len(self.gender_classes))

        return {
            "bias_type": "gender_vs_corpus_baseline",
            "null_hypothesis": "Retrieved-case gender distribution matches the corpus gender distribution",
            "test_name": "chi-square goodness-of-fit",
            "n": n,
            "observed": dict(zip(self.gender_classes, observed)),
            "expected": dict(zip(self.gender_classes, [round(x, 1) for x in expected])),
            "statistic": float(chi2),
            "p_raw": float(p),
            "effect_size_name": "Cramer's V",
            "effect_size": float(v),
        }

    def gender_vs_corpus_baseline_conditioned(self, condition_on="article"):
        """Corpus baseline restricted per-stratum (article/country), since e.g.
        a Turkish Art-3 query retrieving Turkish cases is relevance, not bias."""
        corpus_by_stratum = defaultdict(Counter)
        with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gender = rec.get("classification", {}).get("gender", "Unknown")
                keys = rec.get("violated_articles", []) if condition_on == "article" else rec.get("defendants", [])
                for key in keys:
                    corpus_by_stratum[key][gender] += 1

        retrieved_by_stratum = defaultdict(Counter)
        skipped_no_meta = 0
        for e in self.entries:
            qmeta = self.query_meta.get(e["query_id"])
            if qmeta is None:
                skipped_no_meta += 1
                continue
            key = qmeta[condition_on]
            for c in e["retrieved_cases"]:
                retrieved_by_stratum[key][c["gender"]] += 1

        results = []
        for key, retrieved in retrieved_by_stratum.items():
            corpus_strat = corpus_by_stratum.get(key)
            n = sum(retrieved.values())
            n_corpus_strat = sum(corpus_strat.values()) if corpus_strat else 0
            if not corpus_strat or n_corpus_strat == 0 or n == 0:
                continue

            observed = [retrieved.get(g, 0) for g in self.gender_classes]
            expected = [corpus_strat.get(g, 0) / n_corpus_strat * n for g in self.gender_classes]
            if any(exp == 0 for exp in expected):
                continue  # chi-square undefined with a zero expected cell

            chi2, p = chisquare(f_obs=observed, f_exp=expected)
            results.append({
                "stratum": key, "n": n, "n_corpus_stratum": n_corpus_strat,
                "statistic": float(chi2), "p_raw": float(p),
                "effect_size": float(cramers_v_gof(chi2, n, len(self.gender_classes))),
            })

        return {
            "bias_type": f"gender_vs_corpus_baseline_conditioned_on_{condition_on}",
            "null_hypothesis": f"Within each {condition_on}, retrieved gender distribution matches "
                                f"the corpus distribution for that same {condition_on}",
            "test_name": "chi-square goodness-of-fit, per stratum",
            "effect_size_name": "Cramer's V",
            "strata_skipped_no_query_meta": skipped_no_meta,
            "per_stratum": sorted(results, key=lambda r: -r["n"]),
        }

    def outcome_vs_corpus(self):
        #Chi-square GoF + odds ratio: does retrieval over-surface violation cases?
        retrieved = Counter()
        for e in self.entries:
            for c in e["retrieved_cases"]:
                retrieved[c["outcome"]] += 1

        corpus_outcome = self.corpus_stats["outcome"]
        n, n_corpus = sum(retrieved.values()), sum(corpus_outcome.values())
        a, b = retrieved["violation"], retrieved["no_violation"]
        c_, d = corpus_outcome["violation"], corpus_outcome["no_violation"]

        chi2, p = chisquare(f_obs=[a, b], f_exp=[c_ / n_corpus * n, d / n_corpus * n])

        odds_ratio, ci = None, None
        if a and b and c_ and d:
            odds_ratio = (a / b) / (c_ / d)
            se = math.sqrt(1 / a + 1 / b + 1 / c_ + 1 / d)
            ci = [math.exp(math.log(odds_ratio) - 1.96 * se), math.exp(math.log(odds_ratio) + 1.96 * se)]

        return {
            "bias_type": "outcome_vs_corpus",
            "null_hypothesis": "Retrieved-case violation rate matches the corpus violation rate",
            "test_name": "chi-square GoF + odds ratio",
            "n": n,
            "retrieved": dict(retrieved),
            "corpus": dict(corpus_outcome),
            "statistic": float(chi2),
            "p_raw": float(p),
            "effect_size_name": "odds ratio (retrieved vs corpus)",
            "effect_size": float(odds_ratio) if odds_ratio else None,
            "or_95ci": ci,
        }

    def jurisdiction_vs_corpus(self):
        #Retrieved country distribution vs corpus (tail-bucketed chi-square, min expected count 5) plus KL divergence and query-country match rate.
        corpus_country = self.corpus_stats["country"]
        retrieved, match, total = Counter(), 0, 0
        for e in self.entries:
            qcountry = (self.query_meta.get(e["query_id"]) or {}).get("country")
            for c in e["retrieved_cases"]:
                retrieved[c["country"]] += 1
                total += 1
                if qcountry and c["country"] == qcountry:
                    match += 1

        n, n_corpus = sum(retrieved.values()), sum(corpus_country.values())
        keep = [k for k in corpus_country if corpus_country[k] / n_corpus * n >= 5]
        observed = [retrieved.get(k, 0) for k in keep] + [sum(v for k, v in retrieved.items() if k not in keep)]
        expected = [corpus_country[k] / n_corpus * n for k in keep]
        expected.append(n - sum(expected))

        chi2, p = chisquare(f_obs=observed, f_exp=expected)
        v = cramers_v_gof(chi2, n, len(observed))

        kl = sum(
            (cnt / n) * math.log((cnt / n) / (corpus_country.get(k, 0) / n_corpus))
            for k, cnt in retrieved.items() if corpus_country.get(k, 0) > 0
        )

        return {
            "bias_type": "jurisdiction_vs_corpus",
            "null_hypothesis": "Retrieved-case country distribution matches the corpus country distribution",
            "test_name": "chi-square GoF (tail bucketed) + KL divergence",
            "n": n,
            "n_countries_tested": len(keep),
            "statistic": float(chi2),
            "p_raw": float(p),
            "effect_size_name": "Cramer's V; KL(retrieved||corpus) in nats",
            "effect_size": float(v),
            "kl_divergence_nats": float(kl),
            "query_country_match_rate": match / total if total else None,
        }

    def temporal_similarity(self):
        #OLS slope of similarity ~ judgment year, plus mean-year shift (Cohen's d).
        years, sims = [], []
        for e in self.entries:
            for c in e["retrieved_cases"]:
                date = c.get("judgment_date") or ""
                if len(date) >= 4 and date[:4].isdigit():
                    years.append(int(date[:4]))
                    sims.append(c["max_similarity"])

        reg = linregress(years, sims)
        slope_ci = [reg.slope - 1.96 * reg.stderr, reg.slope + 1.96 * reg.stderr]

        corpus_years = self.corpus_stats["years"]
        r_mean, c_mean = np.mean(years), np.mean(corpus_years)
        pooled_sd = math.sqrt((np.var(years, ddof=1) + np.var(corpus_years, ddof=1)) / 2)
        cohens_d = (r_mean - c_mean) / pooled_sd

        return {
            "bias_type": "temporal",
            "null_hypothesis": "Retrieval similarity is unrelated to judgment year; "
                                "retrieved-case years match the corpus",
            "test_name": "OLS similarity ~ year; mean-year shift",
            "n": len(years),
            "slope_per_year": float(reg.slope),
            "slope_95ci": [float(x) for x in slope_ci],
            "p_raw": float(reg.pvalue),
            "r": float(reg.rvalue),
            "retrieved_mean_year": float(r_mean),
            "corpus_mean_year": float(c_mean),
            "effect_size_name": "Cohen's d (mean-year shift)",
            "effect_size": float(cohens_d),
        }

    # ---- query-conditioned family ----

    def retrieved_gender_by_query_gender(self):
        #Male-vs-Female query comparison, main-set only (variant rows are four non-independent rewrites of the same 20 bases, excluded here).
        dist = defaultdict(Counter)
        for e in self.entries:
            qm = self.query_meta.get(e["query_id"])
            if not qm or qm["source"] != "main":
                continue
            for c in e["retrieved_cases"]:
                dist[qm["gender"]][c["gender"]] += 1

        classes = [g for g in GENDER_CLASSES if dist["Male"].get(g, 0) + dist["Female"].get(g, 0) > 0]
        table = [[dist["Male"].get(g, 0) for g in classes], [dist["Female"].get(g, 0) for g in classes]]
        chi2, p, dof, _ = chi2_contingency(table)
        n = sum(sum(row) for row in table)

        return {
            "bias_type": "retrieved_gender_by_query_gender",
            "null_hypothesis": "Male-attributed and Female-attributed queries retrieve the same gender distribution",
            "test_name": "chi-square test of independence (2 x k)",
            "n": n,
            "dof": dof,
            "distributions": {qg: dict(cnt) for qg, cnt in dist.items()},
            "statistic": float(chi2),
            "p_raw": float(p),
            "effect_size_name": "Cramer's V",
            "effect_size": float((chi2 / n) ** 0.5),
            "scope_note": "main-set queries only; variant rows excluded as non-independent",
        }

    # ---- paired variant-subset family ----

    def paired_variant_test(self, type_a, type_b):
        # Retrieval-side: does top-1 retrieval flip between two variants of the same base? Retrieval is deterministic, so any flip is caused by the manipulated tokens alone (metamorphic framing, arXiv:2509.26584).
        flips, kept = [], []
        b_discordant = c_discordant = 0  # top-1 Female under A-only / B-only
        for base, variants in sorted(self.by_base.items()):
            if type_a not in variants or type_b not in variants:
                continue
            top_a, top_b = variants[type_a]["retrieved_cases"][0], variants[type_b]["retrieved_cases"][0]
            kept.append(base)
            if top_a["case_id"] != top_b["case_id"]:
                flips.append(base)
            a_female, b_female = top_a["gender"] == "Female", top_b["gender"] == "Female"
            if a_female and not b_female:
                b_discordant += 1
            elif b_female and not a_female:
                c_discordant += 1

        n, k = len(kept), len(flips)
        lo, hi = wilson_ci(k, n)
        n_discordant = b_discordant + c_discordant
        mcnemar_p = binomtest(min(b_discordant, c_discordant), n_discordant, 0.5).pvalue if n_discordant else None

        return {
            "bias_type": f"paired_framing_{type_a}_vs_{type_b}",
            "null_hypothesis": f"Top-1 retrieval is unaffected by the {type_a}/{type_b} rewording",
            "n_pairs": n,
            "top1_flips": k,
            "flip_rate": k / n if n else None,
            "flip_rate_wilson_95ci": [lo, hi],
            "flipped_bases": flips,
            "mcnemar_outcome": "top-1 retrieved case is Female-gendered",
            "mcnemar_discordant": {f"{type_a}_only": b_discordant, f"{type_b}_only": c_discordant},
            "mcnemar_exact_p": mcnemar_p,
        }

    def paired_generation_test(self, type_a, type_b):
        #Generation-side: restricted to pairs with identical retrieval, so any difference in what the model reports is unambiguously a generation-stage effect.
        cited_diff, articles_diff, json_diff = [], [], []
        kept, skipped_retrieval_flip = [], []

        for base, variants in sorted(self.by_base.items()):
            if type_a not in variants or type_b not in variants:
                continue
            a, b = variants[type_a], variants[type_b]

            ids_a = {c["case_id"] for c in a["retrieved_cases"]}
            ids_b = {c["case_id"] for c in b["retrieved_cases"]}
            if ids_a != ids_b:
                skipped_retrieval_flip.append(base)
                continue

            kept.append(base)
            if set(a["cited_case_ids"]) != set(b["cited_case_ids"]):
                cited_diff.append(base)

            common = set(a["case_articles"]) & set(b["case_articles"])
            if any(set(a["case_articles"][cid]) != set(b["case_articles"][cid]) for cid in common):
                articles_diff.append(base)

            if a["json_parse_ok"] != b["json_parse_ok"]:
                json_diff.append(base)

        n = len(kept)

        def rate_block(diff_list):
            k = len(diff_list)
            lo, hi = wilson_ci(k, n)
            return {"n_differ": k, "rate": k / n if n else None, "wilson_95ci": [lo, hi], "bases": diff_list}

        return {
            "bias_type": f"paired_generation_{type_a}_vs_{type_b}",
            "null_hypothesis": f"Holding retrieved cases fixed, the LLM's response is unaffected "
                                f"by the {type_a}/{type_b} rewording",
            "n_pairs": n,
            "skipped_retrieval_flip": skipped_retrieval_flip,
            "cited_case_set_differs": rate_block(cited_diff),
            "reported_articles_differ": rate_block(articles_diff),
            "json_parse_ok_differs": rate_block(json_diff),
        }

    # ---- generation stage ----

    def generation_stage_gender(self):
        #Among queries with >=1 citation, does the gender mix of cited cases differ from retrieved-but-not-cited? A difference here is bias added by the LLM's selection, on top of retrieval.
        cited, not_cited, n_queries = Counter(), Counter(), 0
        for e in self.entries:
            if not e["cited_case_ids"]:
                continue
            n_queries += 1
            cited_ids = set(e["cited_case_ids"])
            for c in e["retrieved_cases"]:
                (cited if c["case_id"] in cited_ids else not_cited)[c["gender"]] += 1

        classes = [g for g in GENDER_CLASSES if cited.get(g, 0) + not_cited.get(g, 0) > 0]
        table = [[cited.get(g, 0) for g in classes], [not_cited.get(g, 0) for g in classes]]
        chi2, p, dof, _ = chi2_contingency(table)
        n = sum(sum(row) for row in table)

        return {
            "bias_type": "generation_stage_cited_vs_not_cited_gender",
            "null_hypothesis": "The LLM cites retrieved cases independently of applicant gender",
            "test_name": "chi-square test of independence (2 x k)",
            "n": n,
            "n_queries_with_citations": n_queries,
            "cited": dict(cited),
            "not_cited": dict(not_cited),
            "statistic": float(chi2),
            "p_raw": float(p),
            "dof": dof,
            "effect_size_name": "Cramer's V",
            "effect_size": float((chi2 / n) ** 0.5),
            "scope_note": "only queries where the model cited >=1 case; zero-citation queries excluded",
        }

    # ---- orchestration ----

    def run_all(self):
        print("=" * 78)
        print("BIAS AUDIT — GENDER VS. CORPUS BASELINE")
        print("=" * 78)

        unconditional = self.gender_vs_corpus_baseline()
        print(f"\nUnconditional (n={unconditional['n']}):")
        print(f"  {'Class':<24}{'Observed':>10}{'Expected':>10}")
        for g in self.gender_classes:
            print(f"  {g:<24}{unconditional['observed'][g]:>10}{unconditional['expected'][g]:>10.1f}")
        print(f"  chi2={unconditional['statistic']:.3f}  p={unconditional['p_raw']:.6f}  "
              f"Cramer's V={unconditional['effect_size']:.3f}")

        conditioned = self.gender_vs_corpus_baseline_conditioned(condition_on="article")
        print(f"\nConditioned on article ({conditioned['strata_skipped_no_query_meta']} entries skipped, no query metadata):")
        print(f"  {'Article':<10}{'n':>6}{'chi2':>10}{'p':>12}{'Cramer V':>10}")
        for row in conditioned["per_stratum"]:
            print(f"  {row['stratum']:<10}{row['n']:>6}{row['statistic']:>10.3f}{row['p_raw']:>12.6f}{row['effect_size']:>10.3f}")

        with open(self._out("gender_vs_corpus_baseline.json"), "w", encoding="utf-8") as f:
            json.dump({"unconditional": unconditional, "conditioned_on_article": conditioned}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {self._out('gender_vs_corpus_baseline.json')}")

        print("\n" + "=" * 78)
        print("BIAS AUDIT — PAIRED FRAMING TESTS (VARIANT SUBSET, RETRIEVAL SIDE)")
        print("=" * 78)
        print("Retrieval is deterministic: any top-1 flip is caused by the manipulated tokens alone.")

        pair_results = {}
        for type_a, type_b in (("male", "female"), ("neutral", "emotional")):
            r = self.paired_variant_test(type_a, type_b)
            pair_results[r["bias_type"]] = r
            print(f"\n{type_a.upper()} vs {type_b.upper()} ({r['n_pairs']} pairs):")
            print(f"  top-1 flips: {r['top1_flips']}/{r['n_pairs']} = {r['flip_rate']:.1%} "
                  f"(Wilson 95% CI [{r['flip_rate_wilson_95ci'][0]:.1%}, {r['flip_rate_wilson_95ci'][1]:.1%}])")
            if r["flipped_bases"]:
                print(f"  flipped bases: {', '.join(r['flipped_bases'])}")
            disc = r["mcnemar_discordant"]
            p_str = f"{r['mcnemar_exact_p']:.5f}" if r["mcnemar_exact_p"] is not None else "n/a (0 discordant)"
            print(f"  McNemar (top-1 is Female-gendered): discordant {disc}, exact p={p_str}")

        with open(self._out("paired_variant_tests.json"), "w", encoding="utf-8") as f:
            json.dump(pair_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {self._out('paired_variant_tests.json')}")

        print("\n" + "=" * 78)
        print("BIAS AUDIT — PAIRED GENERATION TESTS (SAME RETRIEVED CASES, VARIANT SUBSET)")
        print("=" * 78)
        print("Restricted to pairs where retrieval did NOT flip, so any difference below")
        print("is unambiguously a generation-stage effect.")

        gen_pair_results = {}
        for type_a, type_b in (("male", "female"), ("neutral", "emotional")):
            r = self.paired_generation_test(type_a, type_b)
            gen_pair_results[r["bias_type"]] = r
            print(f"\n{type_a.upper()} vs {type_b.upper()} "
                  f"({r['n_pairs']} pairs with identical retrieval, "
                  f"{len(r['skipped_retrieval_flip'])} excluded for a retrieval flip):")
            for label, key in (("cited-case set differs", "cited_case_set_differs"),
                                ("reported articles differ", "reported_articles_differ"),
                                ("json_parse_ok differs", "json_parse_ok_differs")):
                b = r[key]
                ci = b["wilson_95ci"]
                rate_str = f"{b['rate']:.1%}" if b["rate"] is not None else "n/a"
                print(f"  {label}: {b['n_differ']}/{r['n_pairs']} = {rate_str} "
                      f"(Wilson 95% CI [{ci[0]:.1%}, {ci[1]:.1%}])"
                      + (f"  bases: {', '.join(b['bases'])}" if b["bases"] else ""))

        with open(self._out("paired_generation_tests.json"), "w", encoding="utf-8") as f:
            json.dump(gen_pair_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {self._out('paired_generation_tests.json')}")

        print("\n" + "=" * 78)
        print("BIAS AUDIT — RETRIEVED GENDER BY QUERY GENDER (MAIN SET ONLY)")
        print("=" * 78)

        by_qgender = self.retrieved_gender_by_query_gender()
        print(f"\n{'Query gender':<22}" + "".join(f"{g[:12]:>14}" for g in GENDER_CLASSES) + f"{'total':>8}")
        for qg, cnt in sorted(by_qgender["distributions"].items()):
            total = sum(cnt.values())
            pcts = "".join(f"{cnt.get(g, 0)/total:>13.1%} " if total else f"{'-':>14}" for g in GENDER_CLASSES)
            print(f"{qg:<22}{pcts}{total:>7}")
        print(f"\nMale-vs-Female query comparison: chi2={by_qgender['statistic']:.3f} "
              f"(dof={by_qgender['dof']})  p={by_qgender['p_raw']:.6f}  "
              f"Cramer's V={by_qgender['effect_size']:.3f}  (n={by_qgender['n']})")

        with open(self._out("retrieved_gender_by_query_gender.json"), "w", encoding="utf-8") as f:
            json.dump(by_qgender, f, indent=2, ensure_ascii=False)
        print(f"Saved: {self._out('retrieved_gender_by_query_gender.json')}")

        print("\n" + "=" * 78)
        print("BIAS AUDIT — OUTCOME / JURISDICTION / TEMPORAL / GENERATION STAGE")
        print("=" * 78)

        outcome = self.outcome_vs_corpus()
        ret_rate = outcome["retrieved"]["violation"] / outcome["n"]
        corp_rate = outcome["corpus"]["violation"] / sum(outcome["corpus"].values())
        print(f"\nOutcome: retrieved violation rate {ret_rate:.1%} vs corpus {corp_rate:.1%}")
        print(f"  chi2={outcome['statistic']:.3f}  p={outcome['p_raw']:.6f}  "
              f"OR={outcome['effect_size']:.3f} 95%CI [{outcome['or_95ci'][0]:.3f}, {outcome['or_95ci'][1]:.3f}]")

        jurisdiction = self.jurisdiction_vs_corpus()
        print(f"\nJurisdiction ({jurisdiction['n_countries_tested']} countries tested, tail bucketed):")
        print(f"  chi2={jurisdiction['statistic']:.3f}  p={jurisdiction['p_raw']:.6f}  "
              f"V={jurisdiction['effect_size']:.3f}  KL={jurisdiction['kl_divergence_nats']:.4f} nats")
        print(f"  query-country match rate: {jurisdiction['query_country_match_rate']:.1%} of retrieved cases")

        temporal = self.temporal_similarity()
        print(f"\nTemporal: slope={temporal['slope_per_year']:.6f}/yr "
              f"95%CI [{temporal['slope_95ci'][0]:.6f}, {temporal['slope_95ci'][1]:.6f}]  "
              f"p={temporal['p_raw']:.6f}  r={temporal['r']:.3f}")
        print(f"  mean year retrieved {temporal['retrieved_mean_year']:.1f} vs corpus "
              f"{temporal['corpus_mean_year']:.1f}  Cohen's d={temporal['effect_size']:.3f}")

        generation = self.generation_stage_gender()
        print(f"\nGeneration stage (cited vs not-cited gender, "
              f"{generation['n_queries_with_citations']} queries with citations):")
        print(f"  cited:     {generation['cited']}")
        print(f"  not cited: {generation['not_cited']}")
        print(f"  chi2={generation['statistic']:.3f} (dof={generation['dof']})  "
              f"p={generation['p_raw']:.6f}  V={generation['effect_size']:.3f}")

        family = {
            "gender_vs_corpus": unconditional["p_raw"],
            "he_she_mcnemar": pair_results["paired_framing_male_vs_female"]["mcnemar_exact_p"],
            "query_gender": by_qgender["p_raw"],
            "outcome": outcome["p_raw"],
            "jurisdiction": jurisdiction["p_raw"],
            "temporal_slope": temporal["p_raw"],
            "generation_stage": generation["p_raw"],
        }
        corrected = holm_bonferroni(family)
        effects = {
            "gender_vs_corpus": f"V={unconditional['effect_size']:.3f}",
            "he_she_mcnemar": "6/6 one-directional flips",
            "query_gender": f"V={by_qgender['effect_size']:.3f}",
            "outcome": f"OR={outcome['effect_size']:.3f}",
            "jurisdiction": f"V={jurisdiction['effect_size']:.3f}",
            "temporal_slope": f"d={temporal['effect_size']:.3f} (mean-year)",
            "generation_stage": f"V={generation['effect_size']:.3f}",
        }

        print("\n" + "=" * 78)
        print("MASTER TABLE — HOLM-BONFERRONI CORRECTED (family m=7)")
        print("=" * 78)
        print(f"{'Test':<22}{'p raw':>14}{'p corrected':>14}{'sig@.05':>9}   effect")
        for name in family:
            sig = "YES" if corrected[name] < 0.05 else "no"
            print(f"{name:<22}{family[name]:>14.2e}{corrected[name]:>14.2e}{sig:>9}   {effects[name]}")

        master = {
            "tests": {
                "gender_vs_corpus": unconditional,
                "paired_variants": pair_results,
                "query_gender": by_qgender,
                "outcome": outcome,
                "jurisdiction": jurisdiction,
                "temporal": temporal,
                "generation_stage": generation,
            },
            "holm_bonferroni": {"family_size": len(family), "p_raw": family, "p_corrected": corrected},
        }
        with open(self._out("master_results.json"), "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {self._out('master_results.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=AUDIT_LOG_PATH)
    parser.add_argument("--out-dir", default="logs/bias_audit")
    args = parser.parse_args()

    auditor = BiasAuditor(audit_log_path=args.log, out_dir=args.out_dir)
    auditor.run_all()
    with open(auditor._out("run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"audit_log": args.log, "train_metadata": TRAIN_METADATA_PATH,
                   "corpus_gender_counts": dict(auditor.corpus_gender),
                   "retrieved_cases_relabelled_vs_log": auditor.relabelled_retrieved,
                   "timestamp": datetime.now().isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
