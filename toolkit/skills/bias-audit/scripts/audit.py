import argparse
import json
import sys
from itertools import combinations

from stats import BASELINE, FIXED_DOCUMENTS, audit_arms, compare_to_baseline, load, load_topics, relevance
from validate import check_file, validate


def pct(x):
    return f"{x:.1%}" if x is not None else "n/a"


def print_summary(arms, res, attribute, groups):
    print("=" * 78)
    print(f"{arms}: {len(res['runs'])} run(s) x {res['queries_per_run']} queries, attribute {attribute}")
    print("=" * 78)
    ri = res["retrieval_identical_across_runs"]
    print(f"Retrieval identical in every run: {ri['identical_queries']}/{ri['of']} queries")
    holm = res["holm_bonferroni"]["p_corrected"]

    c = res["attribute_vs_collection"]
    print(f"\nRetrieved vs collection (n={c['n']}): V={c['effect_size']:.3f}, p={c['p_raw']:.2e}, Holm {holm['attribute_vs_collection']:.2e}")
    for v in c["observed"]:
        print(f"  {v:<24}{c['observed'][v]:>8} retrieved{c['expected'][v]:>10.1f} expected")

    q = res["retrieved_by_query_attribute"]
    if q:
        print(f"\nRetrieved by query attribute ({groups[0]} vs {groups[1]} queries, n={q['n']}): "
              f"V={q['effect_size']:.3f}, p={q['p_raw']:.2e}, Holm {holm['retrieved_by_query_attribute']:.2e}")
    else:
        print(f"\nRetrieved by query attribute: skipped (needs non-twin queries with attribute_value {groups[0]} and {groups[1]})")

    for name, t in res["flip_tests"].items():
        p = f"p={t['p_raw']:.2e}" if t["p_raw"] is not None else "p=n/a (0 discordant)"
        lo, hi = t["flip_rate_wilson_95ci"]
        print(f"\nTop document flip, {name}: {t['top1_flips']}/{t['n_pairs']} = {pct(t['flip_rate'])} "
              f"(95% CI {pct(lo)}-{pct(hi)}); McNemar ({t['mcnemar_outcome']}) {t['mcnemar_discordant']}, {p}")

    print("\nCited vs not cited (generation side), per run:")
    for run, g in res["cited_vs_not_cited_per_run"].items():
        print(f"  {run}: " + (f"V={g['effect_size']:.3f}, p={g['p_raw']:.3f}" if g else "no citations"))

    nf = res["noise_floor_twin_rows"]
    if nf:
        print(f"\nRepeat-run noise (same twin query, two runs): cited set changes {pct(nf['cited_rate_mean'])} "
              f"(range {pct(nf['cited_rate_range'][0])}-{pct(nf['cited_rate_range'][1])})")
    else:
        print("\nRepeat-run noise: needs 2+ runs")

    print("\nCited set changes between twins with the same retrieved documents, by what the LLM saw:")
    for pair, split in res["input_similarity_split"].items():
        print(f"  {pair}")
        for g, v in split.items():
            if not v["n_pairs"]:
                print(f"    {g:38s} no pairs")
                continue
            p = f"p={v['p_wilcoxon_vs_repeat_noise']:.3f}" if v["p_wilcoxon_vs_repeat_noise"] is not None else "p=n/a"
            print(f"    {g:38s} {v['n_pairs']:>3} pairs  {v['rows_differ']}/{v['rows']} ({pct(v['wording_change_rate'])})"
                  f"  vs repeat noise {pct(v['same_pairs_repeat_noise'])}  {p}")


def print_comparison(arms, comp):
    print(f"{arms} vs baseline")
    for pair, t in comp["twin_answer_change"].items():
        for name, c in t.items():
            p = f"p={c['p_wilcoxon_vs_baseline']:.2g}" if c["p_wilcoxon_vs_baseline"] is not None else "p=n/a"
            print(f"  {name.replace('_', ' ')} differs, {pair}: {c['rows_differ']}/{c['rows']} ({pct(c['rate'])}, "
                  f"95% CI {pct(c['wilson_95ci'][0])}-{pct(c['wilson_95ci'][1])}), baseline {c['baseline_rows_differ']}/"
                  f"{c['baseline_rows']}; pairs lower {c['pairs_lower']}, higher {c['pairs_higher']}, {p}")
    d = comp["changed_docs"]
    print(f"  retrieval: {pct(d['retrieved_overlap_with_baseline'])} of documents same as baseline over {d['queries']} queries; "
          f"{d['docs_changed']} documents changed, matching the query's attribute: left {pct(d['left_same_attribute_share'])}, "
          f"entered {pct(d['entered_same_attribute_share'])}")


def main():
    parser = argparse.ArgumentParser(description="Run the bias audit statistics on the three audit files.")
    parser.add_argument("--run-logs", nargs="+", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--attribute", default=None, help="Needed only if the labels file holds several attributes")
    parser.add_argument("--groups", nargs=2, required=True, metavar=("A", "B"),
                        help="The two attribute values to compare, the group you are checking for last (e.g. Male Female)")
    parser.add_argument("--twin-pairs", nargs="+", default=None, metavar="VA:VB",
                        help="Twin variants to compare, e.g. male:female neutral:emotional (default: every pair of variants)")
    parser.add_argument("--topics", help="Optional: documents' topics (rows doc_id, attribute, value; several per document "
                                         "allowed) for the relevance check against the questions' topics")
    parser.add_argument("--out", default="bias_audit_results.json")
    args = parser.parse_args()

    errors, warnings, summary = validate(args.run_logs, args.labels, args.pairs, args.attribute)
    if errors:
        print(f"The audit files are not valid ({len(errors)} problems). Run validate.py for the full list. First one:")
        print(f"  - {errors[0]}")
        sys.exit(1)
    for w in warnings:
        print(f"warning: {w}")
    attribute = summary["attribute"]
    missing = [g for g in args.groups if g not in summary["collection_mix"]]
    if missing:
        print(f"No document has {attribute} value {missing}; the values are {list(summary['collection_mix'])}")
        sys.exit(1)

    variants = list(summary["twin_variants"])
    if args.twin_pairs:
        twin_pairs = [tuple(p.split(":", 1)) for p in args.twin_pairs]
    else:
        # The twin pair named after the two groups (e.g. male:female) goes first: its McNemar test is the one in the Holm family
        named = {v.lower(): v for v in variants}
        main_pair = tuple(named[g.lower()] for g in args.groups) if all(g.lower() in named for g in args.groups) else None
        twin_pairs = ([main_pair] if main_pair else []) + [p for p in combinations(variants, 2)
                                                           if main_pair is None or set(p) != set(main_pair)]
    unknown = sorted({v for p in twin_pairs for v in p} - set(variants))
    if unknown:
        print(f"Unknown twin variants {unknown}; the pairs file has {variants}")
        sys.exit(1)

    by_arms, labels, queries, twins = load(args.run_logs, args.labels, args.pairs, attribute)
    results = {"attribute": attribute, "groups": args.groups, "twin_pairs": [list(p) for p in twin_pairs],
               "collection_mix": summary["collection_mix"], "arms": {}}
    for arms, runs in by_arms.items():
        results["arms"][arms] = audit_arms(runs, labels, queries, twins, args.groups, twin_pairs)
        print_summary(arms, results["arms"][arms], attribute, args.groups)
        print()

    fixes = [arms for arms in by_arms if arms not in (BASELINE, FIXED_DOCUMENTS)]
    if BASELINE in by_arms and fixes:
        print("=" * 78)
        print("FIXES COMPARED WITH THE BASELINE")
        print("=" * 78)
        results["comparison"] = {}
        for arms in fixes:
            runs = by_arms[arms]
            results["comparison"][arms] = compare_to_baseline(by_arms[BASELINE], runs, labels, queries, twins,
                                                              args.groups, twin_pairs)
            print_comparison(arms, results["comparison"][arms])
        print()
    elif fixes:
        print("No baseline rows (arms empty) in the run logs, so the fixes are not compared.\n")

    if args.topics:
        _, topic_errors = check_file(args.topics, "doc_labels")
        if topic_errors:
            print(f"The topics file is not valid ({len(topic_errors)} problems). First one:\n  - {topic_errors[0]}")
            sys.exit(1)
        results["relevance"] = relevance(by_arms, load_topics(args.topics), queries)
        if not results["relevance"]:
            print("Relevance: no question without twins has topics in the pairs file, so it is skipped\n")
        for arms, r in results["relevance"].items():
            ch = r.get("precision_change")
            print(f"Relevance {arms}: top document on topic {pct(r['top1_on_topic'])}, documents on topic "
                  f"{pct(r['precision_at_k'])} over {r['queries']} questions"
                  + (f"; vs baseline {ch['mean']:+.3f} (lower {ch['queries_lower']}, higher {ch['queries_higher']})" if ch else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
