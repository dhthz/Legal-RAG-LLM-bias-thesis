import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

from scipy.stats import wilcoxon

from scripts.eval.bias_statistics import wilson_ci

# Phase F results from the logs, as export_audit_results.py does for Phase E: mitigation_results.json (one block per
# fix, SHA256 of every file read), table 11 (compared cell by cell with the earlier copy), tables 12-15 and the
# robustness summaries R1-R5.
#   python -m scripts.eval.export_mitigation_results

OUT_DIR = "docs/bias_audit/results"
ROBUSTNESS_DIR = "logs/robustness"
ARCHIVE_ROBUSTNESS_DIR = "archive/arm4_first_build/robustness"
BASELINE_PREDICT = "logs/bias_audit/predict"
ROBUSTNESS_FILES = {"r1": "r1_topic_control.json", "r2": "r2_gender_relevant.json", "r3": "r3_detectable_effects.json",
                    "r3b": "r3b_llm_pronoun_fixed_cases.json", "r4": "r4_large_swap_set.json",
                    "r5": "r5_retrieval_cost.json"}
WHOLE_SYSTEM_NOTE = ("whole-system rates: rows (base x run, 4 runs of 40 bases) where the two versions cite different "
                     "cases; p vs baseline: Wilcoxon on the per-base differences in runs (non-zero only)")


@dataclass
class Fix:
    name: str
    method: str
    source: str
    point: str
    status: str
    needs: str
    logs: str = ""                    # results.json and harness.json (default logs/mitigation/<name>)
    key: str = ""                     # name in the robustness summaries (default: name)
    robustness: str = ROBUSTNESS_DIR
    whole_system: bool = True         # 4 predict runs, analysed in logs/mitigation/<name>/whole_system
    same_retrieval: bool = False      # retrieval identical to the baseline

    def __post_init__(self):
        self.logs = self.logs or f"logs/mitigation/{self.name}"
        self.key = self.key or self.name


FIXES = [
    Fix("baseline", "frozen system (no arm)", "", "", "reference", "", same_retrieval=True),
    Fix("blind_query", "rule-based gender-neutral rewrite of the query", "Sun et al. 2021; Garg et al. AIES 2019",
        "query text (pre-retrieval)", "done",
        "query rewrite per request (spaCy + GPT-2 for her); changes the text the user wrote"),
    Fix("pcf", "counterfactual averaging of retrieval scores (he and she version)", "Zhou et al. NeurIPS 2024",
        "retrieval scoring", "done", "gender swap + second retrieval per gendered query; corpus gender prior"),
    Fix("blind_index", "gender-neutral rewrite of every stored chunk (unawareness)",
        "classic baseline (fairness through unawareness)", "corpus / index", "done",
        "one-off corpus rewrite (2.5 h CPU) + re-embed (36 min GPU); new index"),
    Fix("permutation", "permutation self-consistency over the 6 case orders (Kemeny)", "Tang et al. NAACL 2024",
        "generation", "done", "6 LLM calls per query (~6x latency); retrieval unchanged",
        logs="logs/mitigation/baseline", key="baseline", same_retrieval=True),
    Fix("leace_case_gender", "LEACE erasure of each case's gender from chunk + query vectors (first build)",
        "Belrose et al. NeurIPS 2023", "embedding (documents + query)", "documented row: fails",
        "eraser fit on all chunk vectors; new index", logs="logs/mitigation/leace", key="leace", whole_system=False),
    Fix("leace_query", "LEACE erasure of the query-pronoun direction from the query vector",
        "Belrose et al. NeurIPS 2023; concept as Bolukbasi et al. NeurIPS 2016", "query embedding", "done",
        "rank-1 eraser fit on 412 dev he/she twins; one matrix product per query; no new index"),
    Fix("cda_first_build", "embedder fine-tune with CDA twins; positive = own case chunk",
        "Kim et al. ACL Findings 2025 recipe", "embedder (documents + query)", "documented row: drifts",
        "fine-tune sweep (~3 h GPU) + re-embed; new model + index", logs="archive/arm4_first_build/logs_mitigation/cda",
        key="cda", robustness=ARCHIVE_ROBUSTNESS_DIR, whole_system=False),
    Fix("cda", "embedder fine-tune with CDA twins; positive = frozen top chunk of the neutral query",
        "Kim et al. ACL Findings 2025 recipe", "embedder (documents + query)", "done: fails the cost bar",
        "fine-tune sweep (~6 h GPU) + re-embed; new model + index"),
    Fix("ft_ccd", "embedder fine-tune with content conditional debiasing (he/she equally close to neutral + anchor "
        "on neutral), beta 10", "Deng et al. 2024 (arXiv); anchor as Kaneko & Bollegala EACL 2021",
        "embedder (documents + query)", "done", "triplet build (CPU) + 3 one-epoch runs (~30 min GPU) + checkpoint "
        "scoring (~1.5 h) + re-embed (36 min); new model + index"),
    Fix("blind_query+blind_index", "both sides gender-neutral", "combination", "query text + corpus",
        "done (retrieval only)", "both arms' costs; adds nothing over blind_query on flips", whole_system=False),
    Fix("leace_query+blind_index", "query-pronoun erasure + neutral corpus", "combination", "query embedding + corpus",
        "done (retrieval only)", "both arms' costs; adds nothing over leace_query on flips", whole_system=False),
]


# Reads the logs and records the SHA256 of every file read, for the provenance block
class Reader:

    def __init__(self):
        self.hashes = {}

    def json(self, path):
        self._hash(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def runs(self, folder):
        runs = []
        for run in "ABCD":
            path = f"{folder}/run_{run}.jsonl"
            self._hash(path)
            with open(path, "r", encoding="utf-8") as f:
                runs.append({row["query_id"]: row for row in map(json.loads, f)})
        return runs

    def _hash(self, path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        self.hashes[path] = h.hexdigest()


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def interval(ci, spec=".3f", scale=1):
    return f"{scale * ci[0]:{spec}}-{scale * ci[1]:{spec}}"


def rounded(x):
    return round(float(x), 4)


# Per base: in how many of the 4 runs the two versions cite different cases (as a set, or only the first one)
def cited_differs(runs, a, b, first):
    key = (lambda ids: (ids or [])[:1]) if first else (lambda ids: set(ids or []))
    bases = sorted(q[:-len("_male")] for q in runs[0] if q.endswith("_male"))
    return {base: sum(key(run[f"{base}_{a}"]["cited_case_ids"]) != key(run[f"{base}_{b}"]["cited_case_ids"])
                      for run in runs) for base in bases}


def whole_system(runs, baseline_runs, stability):
    out = {}
    for pair, (a, b) in {"he_she": ("male", "female"), "tone": ("neutral", "emotional")}.items():
        for measure, first in (("cited_set", False), ("first_cited", True)):
            per_base = cited_differs(runs, a, b, first)
            k, n = sum(per_base.values()), 4 * len(per_base)
            block = {"rows_differ": k, "rows": n, "rate": rounded(k / n),
                     "wilson_95ci": [rounded(v) for v in wilson_ci(k, n)]}
            if baseline_runs:
                base = cited_differs(baseline_runs, a, b, first)
                diffs = [per_base[q] - base[q] for q in base if per_base[q] != base[q]]
                block.update({"baseline_rows_differ": sum(base.values()),
                              "bases_lower": sum(d < 0 for d in diffs), "bases_higher": sum(d > 0 for d in diffs),
                              "p_wilcoxon_vs_baseline": float(wilcoxon(diffs).pvalue) if diffs else None})
            out[f"{pair}_{measure}"] = block
    noise = stability["noise_floor_variant_rows"]
    out["repeat_noise_cited"] = {"rate_mean": rounded(noise["cited_rate_mean"]),
                                 "range": [rounded(v) for v in noise["cited_rate_range"]]}
    return out


def fix_block(fix, reader, robustness, baseline_runs):
    results = reader.json(f"{fix.logs}/results.json")
    harness = reader.json(f"{fix.logs}/harness.json")["metrics"]
    r2, r4, r5 = (robustness[fix.robustness][k]["results"] for k in ("r2", "r4", "r5"))
    swap = r4["arms"][fix.key]
    block = {
        "method": fix.method, "source": fix.source, "intervention_point": fix.point, "status": fix.status,
        "needs": fix.needs,
        "retrieval_logs": fix.logs + (" (retrieval unchanged)" if fix.same_retrieval and fix.name != "baseline" else ""),
        "retrieval_40_bases": {k: results[k] for k in ("pronoun_flip", "tone_flip", "retrieved_gender_by_query_gender",
                                                       "gender_vs_corpus")},
        "r4_swap_set_300": {"top1_flip": swap["top1_flip"], "mcnemar_top1_female": swap["mcnemar_top1_female"],
                            "mcnemar_holm": r4["mcnemar_holm"].get(fix.key),
                            "vs_baseline": None if fix.same_retrieval else swap.get("vs_baseline")},
        "harness": {k: harness[k] for k in ("recall@1", "recall@5", "recall@10", "mrr@10")},
        "r5_retrieval_cost": r5["arms"][fix.key],
        "r2_gender_relevant": {"audit": r2["audit"][fix.key], "harness": r2["harness"][fix.key]},
    }
    if fix.whole_system:
        is_baseline = fix.name == "baseline"
        runs = baseline_runs if is_baseline else reader.runs(f"logs/mitigation/{fix.name}/predict")
        stability = reader.json(f"logs/mitigation/{fix.name}/whole_system/stability/results.json")
        block["whole_system"] = whole_system(runs, None if is_baseline else baseline_runs, stability)
    return block


def tradeoff_row(fix, b):
    r40, swap, r5, ws = b["retrieval_40_bases"], b["r4_swap_set_300"], b["r5_retrieval_cost"], b.get("whole_system")
    changed, precision = r5.get("changed_cases"), r5.get("article_precision_change")
    if fix.same_retrieval:
        removed_added, cases_changed = ("" if fix.name == "baseline" else "0/0"), 0
    else:
        removed_added = f"{swap['vs_baseline']['flips_removed']}/{swap['vs_baseline']['flips_added']}"
        cases_changed = changed["cases_changed"] if changed else ""
    return {
        "arm": fix.name, "method": fix.method, "source": fix.source, "intervention_point": fix.point,
        "status": fix.status, "flips_40_bases": r40["pronoun_flip"]["top1_flips"],
        "r4_flips_300": swap["top1_flip"]["all"]["k"], "r4_removed_added_vs_baseline": removed_added,
        "tone_flips_40": r40["tone_flip"]["top1_flips"],
        "harness_recall1": f"{b['harness']['recall@1']['value']:.2f}",
        "harness_mrr10": f"{b['harness']['mrr@10']['value']:.3f}",
        "r5_article_precision3": f"{r5['audit_article_precision_at_3']:.3f}",
        "r5_precision_p_vs_baseline": fmt(precision["p_wilcoxon"], ".2g") if precision and not fix.same_retrieval else "",
        "r5_audit_top3_overlap": f"{r5['audit_top3_overlap']:.3f}",
        "r5_top3_cases_changed": cases_changed,
        "r5_article_precision_left_entered":
            f"{changed['left_article_precision']:.3f}/{changed['entered_article_precision']:.3f}" if changed else "",
        "r5_query_gender_share_left_entered":
            f"{changed['left_same_gender_share']:.3f}/{changed['entered_same_gender_share']:.3f}" if changed else "",
        "r2_gender_relevant_share": f"{b['r2_gender_relevant']['audit']['gender_relevant']['mean_share_relevant_in_top3']:.3f}",
        "v_query_to_retrieved_gender": f"{r40['retrieved_gender_by_query_gender']['cramers_v']:.3f}",
        "whole_cited_differs_he_she_pct": f"{100 * ws['he_she_cited_set']['rate']:.1f}" if ws else "",
        "whole_first_cited_differs_pct": f"{100 * ws['he_she_first_cited']['rate']:.1f}" if ws else "",
        "whole_tone_cited_differs_pct": f"{100 * ws['tone_cited_set']['rate']:.1f}" if ws else "",
        "needs": fix.needs,
    }


def retrieval_row(fix, b):
    flip40, swap, h, r5 = b["retrieval_40_bases"]["pronoun_flip"], b["r4_swap_set_300"], b["harness"], b["r5_retrieval_cost"]
    flip, vs, change = swap["top1_flip"]["all"], swap["vs_baseline"] or {}, r5.get("article_precision_change") or {}
    v = b["retrieval_40_bases"]["retrieved_gender_by_query_gender"]
    return {
        "arm": fix.name, "flips_40": flip40["top1_flips"], "pairs_40": flip40["n_pairs"],
        "flips_40_mcnemar_p": fmt(flip40["mcnemar_exact_p"], ".3g"),
        "r4_flips": flip["k"], "r4_pairs": flip["n"], "r4_flip_rate": f"{flip['rate']:.3f}",
        "r4_flip_rate_ci95": interval(flip["wilson_95ci"]), "r4_mcnemar_p_holm": fmt(swap["mcnemar_holm"], ".3g"),
        "r4_flips_removed": vs.get("flips_removed", ""), "r4_flips_added": vs.get("flips_added", ""),
        "r4_removed_vs_added_p": fmt(vs.get("p_exact_mcnemar"), ".3g"),
        "harness_recall1": f"{h['recall@1']['value']:.2f}", "harness_recall1_ci95": interval(h["recall@1"]["wilson_95ci"]),
        "harness_mrr10": f"{h['mrr@10']['value']:.3f}", "harness_mrr10_ci95": interval(h["mrr@10"]["bootstrap_95ci"]),
        "article_precision3": f"{r5['audit_article_precision_at_3']:.3f}",
        "article_precision3_change": fmt(change.get("mean"), "+.3f"),
        "queries_lower": change.get("queries_lower", ""), "queries_higher": change.get("queries_higher", ""),
        "article_precision3_p": fmt(change.get("p_wilcoxon"), ".2g"),
        "audit_top3_overlap": f"{r5['audit_top3_overlap']:.3f}",
        "v_query_to_retrieved_gender": f"{v['cramers_v']:.3f}", "v_p": f"{v['p']:.3g}",
    }


def whole_system_row(fix, b):
    row = {"arm": fix.name}
    for measure, label in (("he_she_cited_set", "he_she_cited"), ("he_she_first_cited", "he_she_first_cited"),
                           ("tone_cited_set", "tone_cited")):
        m = b["whole_system"][measure]
        row.update({f"{label}_rows": f"{m['rows_differ']}/{m['rows']}", f"{label}_pct": f"{100 * m['rate']:.1f}",
                    f"{label}_ci95": interval(m["wilson_95ci"], ".1f", 100),
                    f"{label}_bases_lower": m.get("bases_lower", ""), f"{label}_bases_higher": m.get("bases_higher", ""),
                    f"{label}_p_vs_baseline": fmt(m.get("p_wilcoxon_vs_baseline"), ".2g")})
    noise = b["whole_system"]["repeat_noise_cited"]
    row.update({"repeat_noise_cited_pct": f"{100 * noise['rate_mean']:.1f}",
                "repeat_noise_cited_range": interval(noise["range"], ".1f", 100)})
    return row


def changed_cases_row(fix, b):
    c = b["r5_retrieval_cost"].get("changed_cases") or {}
    r2, r2_harness = b["r2_gender_relevant"]["audit"], b["r2_gender_relevant"]["harness"]
    change = r2.get("change_vs_baseline_gender_relevant") or {}
    return {
        "arm": fix.name, "top3_cases_changed": c.get("cases_changed", 0 if fix.same_retrieval else ""),
        "left_article_precision": fmt(c.get("left_article_precision"), ".3f"),
        "entered_article_precision": fmt(c.get("entered_article_precision"), ".3f"),
        "left_same_gender_share": fmt(c.get("left_same_gender_share"), ".3f"),
        "entered_same_gender_share": fmt(c.get("entered_same_gender_share"), ".3f"),
        "r2_gender_relevant_share": f"{r2['gender_relevant']['mean_share_relevant_in_top3']:.3f}",
        "r2_change_vs_baseline": fmt(change.get("mean"), "+.3f"), "r2_queries_lost": change.get("queries_lost", ""),
        "r2_queries_gained": change.get("queries_gained", ""), "r2_p": fmt(change.get("p_wilcoxon"), ".2g"),
        "r2_harness_recall1_gender_relevant": f"{r2_harness['recall@1']:.3f}", "r2_harness_queries": r2_harness["n_queries"],
    }


# One line per check with its main numbers; the full summaries are copied next to the tables
def robustness_rows(rob):
    r1 = rob["r1"]["results"]["baseline"]["top1"]
    r2, r3, r4, r5 = (rob[k]["results"] for k in ("r2", "r3", "r4", "r5"))
    r3b = rob["r3b"]["results"]["pairs"]["male_vs_female"]["cited_case"]
    share = {a: v["gender_relevant"]["mean_share_relevant_in_top3"] for a, v in r2["audit"].items()}
    lowest = min((a for a in share if a != "baseline"), key=share.get)
    smallest_p = min(v["change_vs_baseline_gender_relevant"]["p_wilcoxon"]
                     for a, v in r2["audit"].items() if a != "baseline")
    full = r1["stratified_by_article_x_country"]
    identical = r3["llm_cited_change_identical_input"]
    base4 = r4["arms"]["baseline"]
    results = {
        "r1": f"baseline top-1: odds ratio {r1['crude']['odds_ratio']:.1f} crude, {full['odds_ratio']:.1f} within article "
              f"x country (CMH p = {full['p']:.2g}, {full['n_strata']} strata, n = {full['n']})",
        "r2": f"gender-relevant share in the top 3: baseline {share['baseline']:.3f}, lowest {lowest} {share[lowest]:.3f}; "
              f"smallest p vs baseline {smallest_p:.2g} ({r2['audit']['baseline']['gender_relevant']['n_queries']} queries)",
        "r3": f"tone flip {r3['tone_retrieval_flip']['observed']} (up to {100 * r3['tone_retrieval_flip']['flip_rate_95ci'][1]:.1f}%); "
              f"cited vs not cited: detectable V {r3['generation_stage_gender']['per_run']['A']['mde_cramers_v']:.3f} per run; "
              f"identical-input change {identical['observed']} (CI {interval(identical['change_rate_95ci'], '.1f', 100)}%)",
        "r3b": f"he/she cited change {r3b['rows_differ']}/{r3b['rows']} ({100 * r3b['change_rate']:.1f}%) vs repeat noise "
               f"{100 * r3b['repeat_noise']:.2f}%, p = {r3b['p_wilcoxon_change_above_noise']:.2g}",
        "r4": f"baseline {base4['top1_flip']['all']['k']}/{r4['n_pairs']} top-1 flips, she-only female top-1 "
              f"{base4['mcnemar_top1_female']['top1_female_only_with_she']} vs he-only "
              f"{base4['mcnemar_top1_female']['top1_female_only_with_he']} (Holm p = {r4['mcnemar_holm']['baseline']:.2g}); "
              + ", ".join(f"{a} {v['top1_flip']['all']['k']}" for a, v in r4["arms"].items() if a != "baseline"),
        "r5": f"harness margin median {r5['harness_margin']['median']:.3f}, {r5['harness_margin']['below_0.01']} of "
              f"{r5['harness_margin']['n']} queries below 0.01; article precision@3: "
              + ", ".join(f"{a} {v['audit_article_precision_at_3']:.3f}" for a, v in r5["arms"].items()),
    }
    return [{"check": "R" + k[1:], "question": rob[k]["question"], "result": v} for k, v in results.items()]


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path} ({len(rows)} rows)")


def compare(old, path):
    new = read_csv(path)
    if old is None:
        print(f"  {path}: no earlier copy to compare with")
        return
    if old == new:
        print(f"  {path}: identical to the earlier copy ({len(new) - 1} rows, {len(new[0])} columns)")
        return
    print(f"  {path}: DIFFERS from the earlier copy (rows {len(old)} -> {len(new)})")
    for i, (a, b) in enumerate(zip(old, new)):
        for col, x, y in zip(new[0], a, b):
            if x != y:
                print(f"    row {i} {col}: {x!r} -> {y!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    tables = f"{args.out_dir}/tables"
    os.makedirs(tables, exist_ok=True)
    os.makedirs(f"{args.out_dir}/robustness", exist_ok=True)

    reader = Reader()
    robustness = {d: {k: reader.json(f"{d}/{ROBUSTNESS_FILES[k]}") for k in ("r2", "r4", "r5")}
                  for d in (ROBUSTNESS_DIR, ARCHIVE_ROBUSTNESS_DIR)}
    baseline_runs = reader.runs(BASELINE_PREDICT)
    blocks = {fix.name: fix_block(fix, reader, robustness, baseline_runs) for fix in FIXES}

    print("Phase F results:")
    tradeoff_path = f"{tables}/11_mitigation_tradeoff.csv"
    earlier = read_csv(tradeoff_path)
    write_csv(tradeoff_path, [tradeoff_row(fix, blocks[fix.name]) for fix in FIXES])
    compare(earlier, tradeoff_path)
    write_csv(f"{tables}/12_mitigation_retrieval.csv", [retrieval_row(fix, blocks[fix.name]) for fix in FIXES])
    write_csv(f"{tables}/13_mitigation_whole_system.csv",
              [whole_system_row(fix, blocks[fix.name]) for fix in FIXES if fix.whole_system])
    write_csv(f"{tables}/14_mitigation_changed_cases.csv", [changed_cases_row(fix, blocks[fix.name]) for fix in FIXES])

    rob = {k: reader.json(f"{ROBUSTNESS_DIR}/{f}") for k, f in ROBUSTNESS_FILES.items()}
    for f in ROBUSTNESS_FILES.values():
        shutil.copyfile(f"{ROBUSTNESS_DIR}/{f}", f"{args.out_dir}/robustness/{f}")
    print(f"  {args.out_dir}/robustness/ ({len(ROBUSTNESS_FILES)} summaries)")
    write_csv(f"{tables}/15_robustness_checks.csv", robustness_rows(rob))

    path = f"{args.out_dir}/mitigation_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(timespec="seconds"),
                   "script": "scripts/eval/export_mitigation_results.py", "note": WHOLE_SYSTEM_NOTE,
                   "arms": blocks, "computed_from": dict(sorted(reader.hashes.items()))}, f, indent=2, ensure_ascii=False)
    print(f"  {path} ({len(reader.hashes)} files read)")


if __name__ == "__main__":
    main()
