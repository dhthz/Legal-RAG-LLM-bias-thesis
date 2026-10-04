import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import binomtest, wilcoxon

from scripts.eval.bias_statistics import holm_bonferroni, wilson_ci

PREDICT_LOG_GLOB = "logs/bias_audit/predict/run_*.jsonl"
BASELINE_LOG_PATH = "logs/bias_audit/stateless/run40_A.jsonl"
MAIN_QUERIES_PATH = "dataset/eval/audit_queries_main.jsonl"
VARIANT_QUERIES_PATH = "dataset/eval/audit_queries_variants.jsonl"
GROUND_TRUTH_PATH = "dataset/test.jsonl"
OUT_DIR = "logs/bias_audit/stats_llm_output"

TYPES = ("original", "neutral", "male", "female", "emotional")
PAIRS = (("neutral", "emotional"), ("male", "female"))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Maps every (base, type) to its query_id; "original" is the main query the variants were written from
def load_design():
    variants = load_jsonl(VARIANT_QUERIES_PATH)
    main_by_source = {r["source_case_id"]: r["query_id"] for r in load_jsonl(MAIN_QUERIES_PATH)}
    qid, source, key_article = {}, {}, {}
    for v in variants:
        b = v["base_query_id"]
        qid[(b, v["variant_type"])] = v["variant_id"]
        qid[(b, "original")] = main_by_source[v["source_case_id"]]
        source[b] = v["source_case_id"]
        key_article[b] = str(v["article"])
    return sorted(source), qid, source, key_article


def load_truth(case_ids):
    truth = {}
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["case_id"] in case_ids:
                truth[r["case_id"]] = {str(a) for a in r.get("violated_articles") or []}
    return truth


# Maps the model's free-form labels onto the dataset's ("6", "P1-1"): "6(1)"/"6-1"/"Article 6" -> "6",
# "1P1"/"1 of Protocol No. 1" -> "P1-1"; labels with no article number ("Not specified") are dropped
def normalize_article(label):
    s = str(label).strip()
    m = re.fullmatch(r"P(\d+)-(\d+)", s, re.IGNORECASE)
    if m:
        return f"P{m.group(1)}-{m.group(2)}"
    m = re.fullmatch(r"(\d+)P(\d+)(?:-\d+)?", s, re.IGNORECASE)
    if m:
        return f"P{m.group(2)}-{m.group(1)}"
    m = re.search(r"(\d+)\D+protocol\D*(\d+)", s, re.IGNORECASE)
    if m:
        return f"P{m.group(2)}-{m.group(1)}"
    m = re.search(r"protocol\D*(\d+)\D+article\D*(\d+)", s, re.IGNORECASE)
    if m:
        return f"P{m.group(1)}-{m.group(2)}"
    m = re.search(r"\d+", s)
    return m.group(0) if m else None


# A row counts only if the JSON block parsed and held a predicted_articles list
def predicted(row):
    p = row.get("predicted_articles")
    if not isinstance(p, list):
        return None
    return {a for a in (normalize_article(x) for x in p) if a}


def retrieved_ids(row):
    return [c["case_id"] for c in row["retrieved_cases"]]


class LLMOutputBiasAuditor:

    def __init__(self, log_paths):
        self.bases, self.qid, self.source, self.key_article = load_design()
        self.truth = load_truth(set(self.source.values()))
        self.runs = {os.path.basename(p): {r["query_id"]: r for r in load_jsonl(p)} for p in sorted(log_paths)}
        self.P = {run: {(b, t): predicted(rows[self.qid[(b, t)]]) if self.qid[(b, t)] in rows else None
                        for b in self.bases for t in TYPES}
                  for run, rows in self.runs.items()}

    def coverage(self):
        out = {}
        for t in TYPES:
            ok = sum(self.P[run][(b, t)] is not None for run in self.runs for b in self.bases)
            out[t] = {"parsed": ok, "of": len(self.runs) * len(self.bases)}
        return out

    # Prompt-only change: retrieval must match the baseline run for every query
    def retrieval_matches_baseline(self):
        if not os.path.exists(BASELINE_LOG_PATH):
            return None
        base = {r["query_id"]: r for r in load_jsonl(BASELINE_LOG_PATH)}
        same = total = 0
        for rows in self.runs.values():
            for q, r in rows.items():
                if q in base:
                    total += 1
                    same += retrieved_ids(r) == retrieved_ids(base[q])
        return {"identical": same, "of": total}

    def accuracy(self, b, t):
        vals = [self.P[run][(b, t)] for run in self.runs if self.P[run][(b, t)] is not None]
        if not vals:
            return None
        T = self.truth[self.source[b]]
        return {
            "jaccard": float(np.mean([jaccard(p, T) for p in vals])),
            "exact": float(np.mean([p == T for p in vals])),
            "key_hit": float(np.mean([self.key_article[b] in p for p in vals])),
            "n_predicted": float(np.mean([len(p) for p in vals])),
            "no_violation": float(np.mean([len(p) == 0 for p in vals])),
        }

    def accuracy_table(self):
        table = {}
        for t in TYPES:
            rows = [a for a in (self.accuracy(b, t) for b in self.bases) if a is not None]
            if not rows:
                continue
            table[t] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} | {"n_bases": len(rows)}
        return table

    # Mean disagreement (1 - Jaccard) of one base between two types within the same run
    def pair_change(self, b, ta, tb):
        d = [1 - jaccard(self.P[run][(b, ta)], self.P[run][(b, tb)]) for run in self.runs
             if self.P[run][(b, ta)] is not None and self.P[run][(b, tb)] is not None]
        return float(np.mean(d)) if d else None

    # Mean disagreement of the same query with itself across runs (noise floor)
    def repeat_change(self, b, t):
        d = [1 - jaccard(self.P[r1][(b, t)], self.P[r2][(b, t)]) for r1, r2 in combinations(self.runs, 2)
             if self.P[r1][(b, t)] is not None and self.P[r2][(b, t)] is not None]
        return float(np.mean(d)) if d else None

    def change_test(self, ta, tb, bases):
        rows = []
        for b in bases:
            pc, na, nb = self.pair_change(b, ta, tb), self.repeat_change(b, ta), self.repeat_change(b, tb)
            if pc is not None and na is not None and nb is not None:
                rows.append((pc, (na + nb) / 2))
        diffs = [pc - noise for pc, noise in rows]
        p = float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0
        identical = [self.P[run][(b, ta)] == self.P[run][(b, tb)] for run in self.runs for b in bases
                     if self.P[run][(b, ta)] is not None and self.P[run][(b, tb)] is not None]
        return {
            "n_bases": len(rows),
            "mean_change_between_wordings": float(np.mean([r[0] for r in rows])),
            "mean_change_same_query_repeat": float(np.mean([r[1] for r in rows])),
            "identical_prediction_rate": float(np.mean(identical)),
            "p_wilcoxon_change_above_noise": p,
        }

    # Wilcoxon on run-averaged Jaccard, and exact McNemar on the majority vote over runs (correct in more than half the runs)
    def accuracy_test(self, ta, tb, bases):
        out = {}
        pairs = [(self.accuracy(b, ta), self.accuracy(b, tb)) for b in bases]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        d = [y["jaccard"] - x["jaccard"] for x, y in pairs]
        out["jaccard"] = {"n_bases": len(pairs), f"mean_{ta}": float(np.mean([x["jaccard"] for x, _ in pairs])),
                          f"mean_{tb}": float(np.mean([y["jaccard"] for _, y in pairs])),
                          "p": float(wilcoxon(d).pvalue) if any(d) else 1.0}
        for m in ("exact", "key_hit"):
            a_only = sum(x[m] > 0.5 and not y[m] > 0.5 for x, y in pairs)
            b_only = sum(y[m] > 0.5 and not x[m] > 0.5 for x, y in pairs)
            n_a = sum(x[m] > 0.5 for x, _ in pairs)
            n_b = sum(y[m] > 0.5 for _, y in pairs)
            out[m] = {f"{ta}_correct": n_a, f"{tb}_correct": n_b, "n_bases": len(pairs),
                      f"{ta}_wilson_95ci": wilson_ci(n_a, len(pairs)), f"{tb}_wilson_95ci": wilson_ci(n_b, len(pairs)),
                      f"only_{ta}": a_only, f"only_{tb}": b_only,
                      "p": float(binomtest(a_only, a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0}
        return out

    def agreement_with_original(self):
        out = {}
        for t in TYPES[1:]:
            vals = [(jaccard(self.P[run][(b, t)], self.P[run][(b, "original")]),
                     self.P[run][(b, t)] == self.P[run][(b, "original")])
                    for run in self.runs for b in self.bases
                    if self.P[run][(b, t)] is not None and self.P[run][(b, "original")] is not None]
            out[t] = {"jaccard": float(np.mean([v[0] for v in vals])), "identical": float(np.mean([v[1] for v in vals]))}
        rep = [self.repeat_change(b, "original") for b in self.bases]
        rep = [r for r in rep if r is not None]
        out["original_repeat_noise"] = {"jaccard": 1 - float(np.mean(rep))}
        return out

    # Bases whose two wordings retrieved the same cases: any change there is the LLM step alone
    def same_retrieval_bases(self, ta, tb):
        first = next(iter(self.runs.values()))
        return [b for b in self.bases
                if self.qid[(b, ta)] in first and self.qid[(b, tb)] in first
                and retrieved_ids(first[self.qid[(b, ta)]]) == retrieved_ids(first[self.qid[(b, tb)]])]

    def predicted_article_shares(self):
        out = {}
        for t in TYPES:
            c, n = Counter(), 0
            for run in self.runs:
                for b in self.bases:
                    p = self.P[run][(b, t)]
                    if p is not None:
                        n += 1
                        c.update(p)
            out[t] = {a: c[a] / n for a, _ in c.most_common(8)}
        c = Counter(a for b in self.bases for a in self.truth[self.source[b]])
        out["true"] = {a: c[a] / len(self.bases) for a, _ in c.most_common(8)}
        return out

    def run_all(self):
        results = {"runs": list(self.runs), "coverage": self.coverage(),
                   "retrieval_matches_baseline": self.retrieval_matches_baseline(),
                   "accuracy_vs_truth": self.accuracy_table(),
                   "agreement_with_original_prediction": self.agreement_with_original(),
                   "predicted_article_shares": self.predicted_article_shares(),
                   "whole_system": {}, "same_retrieval_subset": {}}

        family = {}
        for ta, tb in PAIRS:
            name = f"{ta}_vs_{tb}"
            ch = self.change_test(ta, tb, self.bases)
            acc = self.accuracy_test(ta, tb, self.bases)
            results["whole_system"][name] = {"prediction_change": ch, "accuracy": acc}
            family[f"{name}_change"] = ch["p_wilcoxon_change_above_noise"]
            for m in ("jaccard", "exact", "key_hit"):
                family[f"{name}_{m}"] = acc[m]["p"]

            sub = self.same_retrieval_bases(ta, tb)
            results["same_retrieval_subset"][name] = {"n_bases": len(sub),
                                                      "prediction_change": self.change_test(ta, tb, sub) if len(sub) > 1 else None,
                                                      "accuracy": self.accuracy_test(ta, tb, sub) if len(sub) > 1 else None}

        results["holm_bonferroni"] = {"family_size": len(family), "p_raw": family,
                                      "p_corrected": holm_bonferroni(family)}
        return results


def print_report(res):
    print("=" * 78)
    print(f"LLM OUTPUT BIAS: predicted violated articles ({len(res['runs'])} runs)")
    print("=" * 78)
    print("Parsed predictions:", {t: f"{v['parsed']}/{v['of']}" for t, v in res["coverage"].items()})
    print("Retrieval identical to baseline run:", res["retrieval_matches_baseline"])

    print("\nAccuracy vs the input case's true violated articles (mean over 40 bases, runs averaged)")
    print(f"  {'type':10s} {'Jaccard':>8s} {'exact':>7s} {'key hit':>8s} {'#pred':>6s} {'no-viol':>8s}")
    for t, a in res["accuracy_vs_truth"].items():
        print(f"  {t:10s} {a['jaccard']:8.2f} {a['exact']:7.0%} {a['key_hit']:8.0%} {a['n_predicted']:6.2f} {a['no_violation']:8.0%}")

    print("\nAgreement with the prediction on the original case (same run)")
    for t, a in res["agreement_with_original_prediction"].items():
        extra = f" identical {a['identical']:.0%}" if "identical" in a else ""
        print(f"  {t:22s} Jaccard {a['jaccard']:.2f}{extra}")

    for scope in ("whole_system", "same_retrieval_subset"):
        print(f"\n{scope.replace('_', ' ').upper()}")
        for name, r in res[scope].items():
            if r.get("n_bases") is not None and r.get("prediction_change") is None:
                print(f"  {name}: only {r['n_bases']} bases, skipped")
                continue
            ch, acc = r["prediction_change"], r["accuracy"]
            print(f"  {name} (n={ch['n_bases']}): prediction change {ch['mean_change_between_wordings']:.2f} "
                  f"vs repeat noise {ch['mean_change_same_query_repeat']:.2f} (p={ch['p_wilcoxon_change_above_noise']:.3f}), "
                  f"identical {ch['identical_prediction_rate']:.0%}")
            ta, tb = name.split("_vs_")
            j = acc["jaccard"]
            print(f"    Jaccard {j['mean_' + ta]:.2f} -> {j['mean_' + tb]:.2f} (p={j['p']:.3f})"
                  f" | exact {acc['exact'][ta + '_correct']} -> {acc['exact'][tb + '_correct']} (p={acc['exact']['p']:.3f})"
                  f" | key hit {acc['key_hit'][ta + '_correct']} -> {acc['key_hit'][tb + '_correct']} (p={acc['key_hit']['p']:.3f})")

    print("\nPredicted article shares (of answers) vs true articles of the 40 cases")
    for t, s in res["predicted_article_shares"].items():
        print(f"  {t:10s} " + " ".join(f"{a}:{v:.0%}" for a, v in s.items()))

    h = res["holm_bonferroni"]
    print(f"\nHOLM-BONFERRONI (family m={h['family_size']})")
    for name, p in sorted(h["p_raw"].items(), key=lambda kv: kv[1]):
        pc = h["p_corrected"][name]
        print(f"  {name:32s} p={p:.4f}  corrected={pc:.4f}  {'YES' if pc < 0.05 else 'no'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=None, help="Predict-run logs (default: all in logs/bias_audit/predict/)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    logs = args.logs or sorted(glob.glob(PREDICT_LOG_GLOB))
    res = LLMOutputBiasAuditor(logs).run_all()
    print_report(res)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
