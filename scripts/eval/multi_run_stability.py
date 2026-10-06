import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon

from scripts.eval.bias_statistics import wilson_ci

RUN_LOGS = {r: f"logs/bias_audit/stateless/run40_{r}.jsonl" for r in "ABCD"}
RUN_STATS = {"A": "logs/bias_audit/stats_v3_40bases", "B": "logs/bias_audit/stats_v3_40bases_runB",
             "C": "logs/bias_audit/stats_v3_40bases_runC", "D": "logs/bias_audit/stats_v3_40bases_runD"}
VARIANT_QUERIES_PATH = "dataset/eval/audit_queries_variants.jsonl"
OUT_DIR = "logs/bias_audit/stats_multi_run"
PAIRS = (("male", "female"), ("neutral", "emotional"))
SIMILARITY_GROUPS = ("different order", "same order, different chunks shown", "identical input")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cited(row):
    return set(row["cited_case_ids"] or [])


# Union of all articles the model reported, the definition behind the article noise floor
def reported_articles(row):
    return {str(a) for arts in (row["case_articles"] or {}).values() for a in arts}


# What the LLM actually reads: case order plus the two chunks shown per case (prompts.py shows chunks[:2])
def model_input(row, with_chunks=True):
    return [(c["case_id"],) + (tuple(ch["chunk_id"] for ch in c["chunks"][:2]) if with_chunks else ())
            for c in row["retrieved_cases"]]


def retrieved_set(row):
    return {c["case_id"] for c in row["retrieved_cases"]}


def similarity_group(a, b):
    if model_input(a) == model_input(b):
        return "identical input"
    if model_input(a, False) == model_input(b, False):
        return "same order, different chunks shown"
    return "different order"


class MultiRunStability:

    def __init__(self, run_logs=RUN_LOGS, run_stats=RUN_STATS):
        self.runs = {r: {row["query_id"]: row for row in load_jsonl(p)} for r, p in run_logs.items()}
        self.run_stats = run_stats
        variants = load_jsonl(VARIANT_QUERIES_PATH)
        self.variant_ids = [v["variant_id"] for v in variants]
        self.qid = {(v["base_query_id"], v["variant_type"]): v["variant_id"] for v in variants}
        self.bases = sorted({v["base_query_id"] for v in variants})

    def retrieval_identical(self):
        first = next(iter(self.runs.values()))
        same = sum(all(model_input(rows[q]) == model_input(first[q]) for rows in self.runs.values()) for q in first)
        return {"identical_queries": same, "of": len(first)}

    def per_run_summary(self):
        out = {}
        for r, rows in self.runs.items():
            out[r] = {
                "json_parse_ok": sum(row["json_parse_ok"] for row in rows.values()),
                "zero_cited": sum(not row["cited_case_ids"] for row in rows.values()),
                "generation_errors": sum(bool(row["generation_error"]) for row in rows.values()),
                "n_queries": len(rows),
            }
            # Per-run bias_statistics output exists only for the Phase E runs
            if not self.run_stats:
                continue
            with open(os.path.join(self.run_stats[r], "master_results.json"), "r", encoding="utf-8") as f:
                master = json.load(f)
            with open(os.path.join(self.run_stats[r], "paired_generation_tests.json"), "r", encoding="utf-8") as f:
                gen = json.load(f)
            out[r]["generation_stage_p"] = master["holm_bonferroni"]["p_raw"]["generation_stage"]
            out[r]["paired_cited_set_differs"] = {k.replace("paired_generation_", ""): {
                "n_differ": v["cited_case_set_differs"]["n_differ"], "n_pairs": v["n_pairs"]} for k, v in gen.items()}
        return out

    # Same variant query, two runs: how often the output changes with nothing changed
    def noise_floor(self):
        pairs, cited_rates, article_rates = {}, [], []
        n = len(self.variant_ids)
        for r1, r2 in combinations(self.runs, 2):
            c = sum(cited(self.runs[r1][q]) != cited(self.runs[r2][q]) for q in self.variant_ids)
            a = sum(reported_articles(self.runs[r1][q]) != reported_articles(self.runs[r2][q]) for q in self.variant_ids)
            pairs[f"{r1}-{r2}"] = {"cited_differs": c, "articles_differ": a, "n": n}
            cited_rates.append(c / n)
            article_rates.append(a / n)
        return {"run_pairs": pairs,
                "cited_rate_mean": float(np.mean(cited_rates)), "cited_rate_range": [min(cited_rates), max(cited_rates)],
                "articles_rate_mean": float(np.mean(article_rates)), "articles_rate_range": [min(article_rates), max(article_rates)]}

    # Splits pairs with the same retrieved set by how similar the model's input really was; the input
    # depends only on retrieval, so the group of a base is the same in every run
    def input_similarity_split(self, ta, tb):
        first = next(iter(self.runs.values()))
        groups = defaultdict(list)
        for b in self.bases:
            a_row, b_row = first[self.qid[(b, ta)]], first[self.qid[(b, tb)]]
            if retrieved_set(a_row) == retrieved_set(b_row):
                groups[similarity_group(a_row, b_row)].append(b)

        out = {}
        n_runs = len(self.runs)
        for g in SIMILARITY_GROUPS:
            bases = groups.get(g, [])
            if not bases:
                out[g] = {"n_bases": 0}
                continue
            wording, repeat = [], []
            for b in bases:
                qa, qb = self.qid[(b, ta)], self.qid[(b, tb)]
                wording.append(np.mean([cited(rows[qa]) != cited(rows[qb]) for rows in self.runs.values()]))
                repeat.append(np.mean([cited(self.runs[r1][q]) != cited(self.runs[r2][q])
                                       for q in (qa, qb) for r1, r2 in combinations(self.runs, 2)]))
            rows_differ = int(round(sum(wording) * n_runs))
            diffs = [w - r for w, r in zip(wording, repeat)]
            out[g] = {
                "n_bases": len(bases),
                "bases": bases,
                "rows_differ": rows_differ,
                "rows": len(bases) * n_runs,
                "wording_change_rate": float(np.mean(wording)),
                "wording_change_wilson_95ci": wilson_ci(rows_differ, len(bases) * n_runs),
                "same_bases_repeat_noise": float(np.mean(repeat)),
                "p_wilcoxon_vs_repeat_noise": float(wilcoxon(diffs).pvalue) if len(bases) > 1 and any(diffs) else None,
            }
        return out

    def run_all(self):
        return {
            "runs": list(self.runs),
            "retrieval_identical_across_runs": self.retrieval_identical(),
            "per_run": self.per_run_summary(),
            "noise_floor_variant_rows": self.noise_floor(),
            "input_similarity_split": {f"{ta}_vs_{tb}": self.input_similarity_split(ta, tb) for ta, tb in PAIRS},
        }


def print_report(res):
    print("=" * 78)
    print(f"MULTI-RUN STABILITY ({len(res['runs'])} full runs, frozen prompt)")
    print("=" * 78)
    ri = res["retrieval_identical_across_runs"]
    print(f"Retrieval identical in every run: {ri['identical_queries']}/{ri['of']} queries")
    print("\nPer run:")
    for r, v in res["per_run"].items():
        line = f"  {r}: JSON ok {v['json_parse_ok']}/{v['n_queries']}, generation errors {v['generation_errors']}"
        if "generation_stage_p" in v:
            cs = " | ".join(f"{k} {x['n_differ']}/{x['n_pairs']}" for k, x in v["paired_cited_set_differs"].items())
            line += f", generation-stage p={v['generation_stage_p']:.3f}, same-set cited differs: {cs}"
        print(line)
    nf = res["noise_floor_variant_rows"]
    print(f"\nNoise floor (same variant query, two runs): cited {nf['cited_rate_mean']:.1%} "
          f"(range {nf['cited_rate_range'][0]:.1%}-{nf['cited_rate_range'][1]:.1%}), "
          f"articles {nf['articles_rate_mean']:.1%} (range {nf['articles_rate_range'][0]:.1%}-{nf['articles_rate_range'][1]:.1%})")
    print("\nCited case changes, split by how similar the model's input was:")
    for pair, groups in res["input_similarity_split"].items():
        print(f"  {pair}")
        for g, v in groups.items():
            if not v["n_bases"]:
                print(f"    {g:36s} no bases")
                continue
            p = f"p={v['p_wilcoxon_vs_repeat_noise']:.3f}" if v["p_wilcoxon_vs_repeat_noise"] is not None else "p=n/a"
            print(f"    {g:36s} {v['n_bases']:>2} bases  {v['rows_differ']}/{v['rows']} rows ({v['wording_change_rate']:.1%})"
                  f"  vs same-base repeat noise {v['same_bases_repeat_noise']:.1%}  {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=None, help="Run logs (default: the Phase E run40_A-D)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.logs:
        stability = MultiRunStability({os.path.splitext(os.path.basename(p))[0]: p for p in args.logs}, None)
    else:
        stability = MultiRunStability()
    res = stability.run_all()
    print_report(res)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
