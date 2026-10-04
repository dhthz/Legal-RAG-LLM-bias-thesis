import argparse
import csv
import hashlib
import json
import os

RUN_STATS = {"A": "logs/bias_audit/stats_v3_40bases", "B": "logs/bias_audit/stats_v3_40bases_runB",
             "C": "logs/bias_audit/stats_v3_40bases_runC", "D": "logs/bias_audit/stats_v3_40bases_runD"}
MULTI_RUN_PATH = "logs/bias_audit/stats_multi_run/results.json"
LLM_OUTPUT_PATH = "logs/bias_audit/stats_llm_output/results.json"
LABEL_VALIDATION_PATH = "logs/bias_audit/gender_validation_results.json"
MANIFEST_PATH = "FROZEN_BASELINE_MANIFEST.json"
RAW_LOGS = [f"logs/bias_audit/stateless/run40_{r}.jsonl" for r in "ABCD"] + \
           [f"logs/bias_audit/predict/run_{r}.jsonl" for r in "ABCD"]
QUERY_SETS = ["dataset/eval/audit_queries_main.jsonl", "dataset/eval/audit_queries_variants.jsonl"]
OUT_DIR = "docs/bias_audit/results"

MASTER_TESTS = {
    "gender_vs_corpus": ("Does the gender mix of retrieved cases match the corpus?", "gender_vs_corpus"),
    "he_she_mcnemar": ("Does a he/she swap change the gender of the top-retrieved case?", None),
    "query_gender": ("Does retrieved gender depend on the query's applicant gender?", "query_gender"),
    "outcome": ("Does the violation rate of retrieved cases match the corpus?", "outcome"),
    "jurisdiction": ("Does the country mix of retrieved cases match the corpus?", "jurisdiction"),
    "temporal_slope": ("Does retrieval favour newer judgments?", "temporal"),
    "generation_stage": ("Does the LLM cite retrieved cases independently of gender?", "generation_stage"),
}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def r4(x):
    return round(float(x), 4) if x is not None else None


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path} ({len(rows)} rows)")


def master_table(masters):
    a = masters["A"]
    tests, holm = a["tests"], a["holm_bonferroni"]
    rows = []
    for name, (question, key) in MASTER_TESTS.items():
        per_run_p = [m["holm_bonferroni"]["p_raw"][name] for m in masters.values()]
        stable = len({round(p, 12) for p in per_run_p}) == 1
        runs_note = "identical in all 4 runs" if stable else "per run: " + ", ".join(f"{p:.3f}" for p in per_run_p)
        if key:
            t = tests[key]
            test_name, n, es_name, es = t["test_name"], t["n"], t["effect_size_name"], t["effect_size"]
            # the temporal test is a regression, so its statistic is the similarity slope per year
            stat = t["statistic"] if "statistic" in t else t["slope_per_year"]
        else:
            t = tests["paired_variants"]["paired_framing_male_vs_female"]
            d = t["mcnemar_discordant"]
            test_name, n, stat = "exact McNemar (top-1 case is female)", t["n_pairs"], None
            es_name, es = "discordant pairs male-only / female-only", f"{d['male_only']} / {d['female_only']}"
        p, pc = holm["p_raw"][name], holm["p_corrected"][name]
        rows.append([name, question, test_name, n, r4(stat) if stat is not None else "", es_name,
                     es if isinstance(es, str) else r4(es), f"{p:.3g}", f"{pc:.3g}", "yes" if pc < 0.05 else "no", runs_note])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    tables = os.path.join(args.out_dir, "tables")
    os.makedirs(tables, exist_ok=True)

    masters = {r: load(os.path.join(d, "master_results.json")) for r, d in RUN_STATS.items()}
    variant_tests = load(os.path.join(RUN_STATS["A"], "paired_variant_tests.json"))
    by_qgender = load(os.path.join(RUN_STATS["A"], "retrieved_gender_by_query_gender.json"))
    multi = load(MULTI_RUN_PATH)
    llm = load(LLM_OUTPUT_PATH)
    labels = load(LABEL_VALIDATION_PATH)
    manifest = load(MANIFEST_PATH)

    print(f"Writing {args.out_dir}/")
    write_csv(os.path.join(tables, "01_master_holm.csv"),
              ["test", "question", "test_name", "n", "statistic", "effect_size_name", "effect_size",
               "p_raw", "p_holm", "significant_holm_0.05", "across_4_runs"],
              master_table(masters))

    rows = []
    for name, t in variant_tests.items():
        a_only, b_only = list(t["mcnemar_discordant"].values())
        rows.append([name.replace("paired_framing_", ""), t["n_pairs"], t["top1_flips"], r4(t["flip_rate"]),
                     r4(t["flip_rate_wilson_95ci"][0]), r4(t["flip_rate_wilson_95ci"][1]),
                     a_only, b_only, f"{t['mcnemar_exact_p']:.3g}" if t["mcnemar_exact_p"] is not None else ""])
    write_csv(os.path.join(tables, "02_paired_variants_top1.csv"),
              ["comparison", "n_pairs", "top1_case_changed", "change_rate", "ci95_lo", "ci95_hi",
               "female_top1_first_wording_only", "female_top1_second_wording_only", "mcnemar_exact_p"], rows)

    classes = ["Male", "Female", "Multiple Applicants", "Unknown"]
    rows = []
    for qg in classes:
        dist = by_qgender["distributions"].get(qg, {})
        total = sum(dist.values())
        rows.append([qg, total] + [r4(dist.get(c, 0) / total) if total else "" for c in classes])
    write_csv(os.path.join(tables, "03_retrieved_gender_by_query_gender.csv"),
              ["query_applicant_gender", "retrieved_cases"] + [f"share_{c.lower().replace(' ', '_')}" for c in classes], rows)

    rows = []
    for pair, groups in multi["input_similarity_split"].items():
        for g, v in groups.items():
            if not v["n_bases"]:
                rows.append([pair, g, 0, "", "", "", "", "", "", ""])
                continue
            ci = v["wording_change_wilson_95ci"]
            p = v["p_wilcoxon_vs_repeat_noise"]
            rows.append([pair, g, v["n_bases"], v["rows_differ"], v["rows"], r4(v["wording_change_rate"]),
                         r4(ci[0]), r4(ci[1]), r4(v["same_bases_repeat_noise"]), r4(p) if p is not None else ""])
    write_csv(os.path.join(tables, "04_llm_citation_by_input_similarity.csv"),
              ["comparison", "model_input", "n_bases", "rows_cited_case_changed", "rows", "change_rate",
               "ci95_lo", "ci95_hi", "same_query_repeat_noise", "p_wilcoxon_vs_noise"], rows)

    rows = [[t, a["n_bases"], r4(a["jaccard"]), r4(a["exact"]), r4(a["key_hit"]), r4(a["n_predicted"]), r4(a["no_violation"])]
            for t, a in llm["accuracy_vs_truth"].items()]
    write_csv(os.path.join(tables, "05_predicted_articles_accuracy.csv"),
              ["wording", "n_bases", "jaccard_vs_true_articles", "exact_match", "key_article_hit",
               "articles_predicted", "predicts_no_violation"], rows)

    h = llm["holm_bonferroni"]
    rows = []
    for scope in ("whole_system", "same_retrieval_subset"):
        for pair, r in llm[scope].items():
            ch = r.get("prediction_change")
            if not ch:
                continue
            ta, tb = pair.split("_vs_")
            acc = r["accuracy"]
            key = f"{pair}_change"
            rows.append([scope, pair, ch["n_bases"], r4(ch["mean_change_between_wordings"]), r4(ch["mean_change_same_query_repeat"]),
                         r4(ch["identical_prediction_rate"]), r4(ch["p_wilcoxon_change_above_noise"]),
                         r4(acc["jaccard"][f"mean_{ta}"]), r4(acc["jaccard"][f"mean_{tb}"]), r4(acc["jaccard"]["p"]),
                         f"{acc['exact'][ta + '_correct']} / {acc['exact'][tb + '_correct']}", r4(acc["exact"]["p"]),
                         f"{acc['key_hit'][ta + '_correct']} / {acc['key_hit'][tb + '_correct']}", r4(acc["key_hit"]["p"]),
                         r4(h["p_corrected"][key]) if scope == "whole_system" else ""])
    write_csv(os.path.join(tables, "06_predicted_articles_tests.csv"),
              ["scope", "comparison", "n_bases", "prediction_change", "repeat_noise", "identical_prediction",
               "p_change_vs_noise", "jaccard_first", "jaccard_second", "p_jaccard", "exact_correct_bases",
               "p_exact", "key_hit_bases", "p_key_hit", "p_change_holm"], rows)

    shares = llm["predicted_article_shares"]
    arts = sorted({a for s in shares.values() for a in s}, key=lambda a: -shares["true"].get(a, 0))
    write_csv(os.path.join(tables, "07_predicted_article_shares.csv"),
              ["article"] + list(shares), [[a] + [r4(shares[t].get(a, 0)) for t in shares] for a in arts])

    nf = multi["noise_floor_variant_rows"]
    write_csv(os.path.join(tables, "08_noise_floors.csv"),
              ["run_pair", "variant_rows", "cited_case_changed", "cited_rate", "reported_articles_changed", "articles_rate"],
              [[k, v["n"], v["cited_differs"], r4(v["cited_differs"] / v["n"]), v["articles_differ"], r4(v["articles_differ"] / v["n"])]
               for k, v in nf["run_pairs"].items()])

    write_csv(os.path.join(tables, "09_per_run_summary.csv"),
              ["run", "queries", "generation_errors", "json_parse_ok", "zero_cited", "generation_stage_p"],
              [[r, v["n_queries"], v["generation_errors"], v["json_parse_ok"], v["zero_cited"], r4(v["generation_stage_p"])]
               for r, v in multi["per_run"].items()])

    rows = [[c, v["n"], v["correct"], r4(v["accuracy"]), r4(v["wilson_95ci"][0]), r4(v["wilson_95ci"][1])]
            for c, v in labels["per_class"].items()]
    o = labels["overall_reweighted"]
    rows.append(["overall (reweighted to corpus shares)", labels["n_tagged"], "", r4(o["accuracy"]),
                 r4(o["approx_95ci"][0]), r4(o["approx_95ci"][1])])
    write_csv(os.path.join(tables, "10_gender_label_validation.csv"),
              ["ai_label", "n_checked", "correct", "accuracy", "ci95_lo", "ci95_hi"], rows)

    a = masters["A"]["tests"]
    pv = variant_tests["paired_framing_male_vs_female"]
    sim = multi["input_similarity_split"]
    final = {
        "headline": {
            "pronoun_swap_top1": {"pairs": pv["n_pairs"], "top1_changed": pv["top1_flips"],
                                  "discordant": pv["mcnemar_discordant"],
                                  "p_holm": masters["A"]["holm_bonferroni"]["p_corrected"]["he_she_mcnemar"]},
            "gender_vs_corpus_cramers_v": a["gender_vs_corpus"]["effect_size"],
            "query_gender_cramers_v": a["query_gender"]["effect_size"],
            "jurisdiction_cramers_v": a["jurisdiction"]["effect_size"],
            "query_country_match_rate": a["jurisdiction"]["query_country_match_rate"],
            "temporal_cohens_d": a["temporal"]["effect_size"],
            "outcome_odds_ratio": a["outcome"]["effect_size"],
            "generation_stage_p_per_run": {r: m["holm_bonferroni"]["p_raw"]["generation_stage"] for r, m in masters.items()},
            "llm_identical_input_cited_change": {
                "male_vs_female": sim["male_vs_female"]["identical input"].get("wording_change_rate"),
                "neutral_vs_emotional": sim["neutral_vs_emotional"]["identical input"].get("wording_change_rate"),
                "neutral_vs_emotional_repeat_noise": sim["neutral_vs_emotional"]["identical input"].get("same_bases_repeat_noise")},
            "position_bias_cited_change_different_order": {
                p: sim[p]["different order"].get("wording_change_rate") for p in sim},
            "cited_noise_floor_mean": nf["cited_rate_mean"],
            "predicted_articles_min_p_holm": min(llm["holm_bonferroni"]["p_corrected"].values()),
            "gender_label_accuracy": o["accuracy"],
        },
        "provenance": {
            "faiss_index_sha256": manifest["faiss_index"]["sha256_checksum"],
            "retrieval_identical_across_runs": multi["retrieval_identical_across_runs"],
            "raw_logs_sha256": {p: sha256(p) for p in RAW_LOGS},
            "query_sets_sha256": {p: sha256(p) for p in QUERY_SETS},
            "computed_from": list(RUN_STATS.values()) + [MULTI_RUN_PATH, LLM_OUTPUT_PATH, LABEL_VALIDATION_PATH],
            "scripts": ["scripts/eval/bias_statistics.py", "scripts/eval/multi_run_stability.py",
                        "scripts/eval/llm_output_bias.py",
                        "scripts/dataset/gender_label_validation/analyze_gender_validation.py",
                        "scripts/eval/export_audit_results.py"],
        },
    }
    path = os.path.join(args.out_dir, "final_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"  {path}")


if __name__ == "__main__":
    main()
