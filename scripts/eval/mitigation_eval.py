import argparse
import json
import os
from datetime import datetime

from scripts.eval.bias_statistics import BiasAuditor, wilson_ci
from scripts.eval.retrieval_eval import run_harness
from scripts.eval.run_bias_audit import BiasAuditRunner
from src.mitigation import build_arms
from src.rag.pipeline import RAGPipeline

# Retrieval-side evaluation of a mitigation arm (or a combination), with the Phase E instruments:
#   pronoun flip (he/she top-1, exact McNemar), tone flip, agreement with the hand-written neutral variant,
#   female share of top-1, retrieved gender vs corpus and by query gender, and the B.1 harness (recall@1 / MRR).
# Output, one folder per arm combination (the frozen system is "baseline"):
#   logs/mitigation/<arms>/retrieval.jsonl  retrieval of all 335 audit queries, audit-log format
#   logs/mitigation/<arms>/harness.json     B.1 harness output
#   logs/mitigation/<arms>/results.json     summary, with deltas against the baseline

OUT_ROOT = "logs/mitigation"
BASELINE = "baseline"


def arm_tag(names):
    return "+".join(names) if names else BASELINE


def write_retrieval_log(pipeline, path):
    queries = BiasAuditRunner.load_queries()
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            effective_query, cases = pipeline.retrieve(q["query_text"])
            row = {
                "query_id": q["query_id"],
                "query": q["query_text"],
                **({"rewritten_query": effective_query} if effective_query != q["query_text"] else {}),
                "arms": [arm.name for arm in pipeline.arms],
                "retrieved_cases": [pipeline.case_log_entry(rank, c) for rank, c in enumerate(cases, start=1)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def top1(entry):
    return entry["retrieved_cases"][0]


def rate(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson_95ci": [lo, hi]}


# Top-1 of the male and female variants compared with the top-1 of the hand-written neutral variant
# under the frozen system: how close the mitigated ranking gets to a human-written gender-free query
def neutral_reference_agreement(auditor, reference):
    out = {}
    for vtype in ("male", "female"):
        bases = [b for b in auditor.by_base if vtype in auditor.by_base[b] and "neutral" in reference.by_base.get(b, {})]
        same = sum(top1(auditor.by_base[b][vtype])["case_id"] == top1(reference.by_base[b]["neutral"])["case_id"] for b in bases)
        out[vtype] = rate(same, len(bases))
    return out


def female_top1_share(auditor):
    out = {}
    for vtype in ("male", "female"):
        rows = [v[vtype] for v in auditor.by_base.values() if vtype in v]
        out[vtype] = rate(sum(top1(r)["gender"] == "Female" for r in rows), len(rows))
    return out


def summarise(auditor, reference, harness, n_rewritten):
    flip = auditor.paired_variant_test("male", "female")
    tone = auditor.paired_variant_test("neutral", "emotional")
    by_query = auditor.retrieved_gender_by_query_gender()
    corpus = auditor.gender_vs_corpus_baseline()
    m = harness["metrics"]
    return {
        "pronoun_flip": {key: flip[key] for key in ("n_pairs", "top1_flips", "flip_rate", "flip_rate_wilson_95ci",
                                                     "mcnemar_discordant", "mcnemar_exact_p", "flipped_bases")},
        "tone_flip": {key: tone[key] for key in ("n_pairs", "top1_flips", "flip_rate", "mcnemar_exact_p")},
        "neutral_reference_agreement": neutral_reference_agreement(auditor, reference),
        "female_top1_share": female_top1_share(auditor),
        "retrieved_gender_by_query_gender": {"cramers_v": by_query["effect_size"], "p": by_query["p_raw"]},
        "gender_vs_corpus": {"cramers_v": corpus["effect_size"], "p": corpus["p_raw"]},
        "harness": {"recall@1": m["recall@1"]["value"], "mrr@10": m["mrr@10"]["value"],
                    "prod_recall@1": m["prod_recall@1"]["value"]},
        "queries_rewritten": n_rewritten,
    }


HEADLINE = [
    ("pronoun flips (of 40)", lambda r: r["pronoun_flip"]["top1_flips"]),
    ("McNemar p", lambda r: r["pronoun_flip"]["mcnemar_exact_p"]),
    ("tone flips (of 40)", lambda r: r["tone_flip"]["top1_flips"]),
    ("male top-1 = neutral ref", lambda r: r["neutral_reference_agreement"]["male"]["rate"]),
    ("female top-1 = neutral ref", lambda r: r["neutral_reference_agreement"]["female"]["rate"]),
    ("female share top-1, he", lambda r: r["female_top1_share"]["male"]["rate"]),
    ("female share top-1, she", lambda r: r["female_top1_share"]["female"]["rate"]),
    ("query->retrieved gender V", lambda r: r["retrieved_gender_by_query_gender"]["cramers_v"]),
    ("recall@1", lambda r: r["harness"]["recall@1"]),
    ("MRR@10", lambda r: r["harness"]["mrr@10"]),
]


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def print_comparison(tag, results, baseline):
    print("\n" + "=" * 70)
    print(f"MITIGATION EVAL: {tag}")
    print("=" * 70)
    print(f"{'metric':<30}{'baseline':>12}{tag[:14]:>14}{'delta':>12}")
    print("-" * 70)
    for label, get in HEADLINE:
        b, a = get(baseline), get(results)
        delta = a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) and label != "McNemar p" else None
        print(f"{label:<30}{fmt(b):>12}{fmt(a):>14}{fmt(delta) if delta is not None else '':>12}")


def evaluate(names, pipeline, reference_log):
    tag = arm_tag(names)
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    pipeline.arms = build_arms(names)

    retrieval_path = os.path.join(out_dir, "retrieval.jsonl")
    print(f"[{tag}] retrieving the 335 audit queries...")
    write_retrieval_log(pipeline, retrieval_path)
    with open(retrieval_path, "r", encoding="utf-8") as f:
        n_rewritten = sum("rewritten_query" in json.loads(line) for line in f)

    print(f"[{tag}] running the B.1 harness...")
    harness = run_harness(pipeline.retriever, pipeline.config, arms=pipeline.arms, verbose=False)
    with open(os.path.join(out_dir, "harness.json"), "w", encoding="utf-8") as f:
        json.dump(harness, f, indent=2, ensure_ascii=False)

    auditor = BiasAuditor(audit_log_path=retrieval_path, out_dir=out_dir)
    reference = auditor if reference_log is None else BiasAuditor(audit_log_path=reference_log, out_dir=out_dir)
    results = {
        "arms": names,
        "created": datetime.now().isoformat(timespec="seconds"),
        "index_path": pipeline.config.index_path,
        "files": {"retrieval_log": retrieval_path, "harness": os.path.join(out_dir, "harness.json")},
        **summarise(auditor, reference, harness, n_rewritten),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="", help="Comma-separated arms (src/mitigation ARM_REGISTRY); empty = baseline")
    parser.add_argument("--rerun-baseline", action="store_true")
    args = parser.parse_args()
    names = [a for a in args.arms.split(",") if a]

    pipeline = RAGPipeline()
    baseline_dir = os.path.join(OUT_ROOT, BASELINE)
    baseline_results = os.path.join(baseline_dir, "results.json")
    if args.rerun_baseline or not os.path.exists(baseline_results):
        evaluate([], pipeline, None)
    with open(baseline_results, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    results = baseline if not names else evaluate(names, pipeline, os.path.join(baseline_dir, "retrieval.jsonl"))
    print_comparison(arm_tag(names), results, baseline)
    print(f"\nSaved: {os.path.join(OUT_ROOT, arm_tag(names))}/ (retrieval.jsonl, harness.json, results.json)")


if __name__ == "__main__":
    main()
