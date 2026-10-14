import argparse
import json

from stats import BASELINE

ALPHA = 0.05
MIN_EXPECTED = 5  # chi-square is unreliable when a cell expects fewer documents than this


def pct(x):
    return f"{x:.1%}" if x is not None else "n/a"


def p_text(p):
    if p is None:
        return "n/a"
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def small_cells(t):
    return (f" Unreliable: one group is too small (smallest expected count {t['min_expected']:.1f}, below {MIN_EXPECTED})."
            if t["min_expected"] < MIN_EXPECTED else "")


def main_pair(res):
    return next(iter(res["flip_tests"]), None)


# Layer 1: does the search itself treat the twins differently?
def retrieval_section(res, groups, holm):
    lines, finding = ["## 1. Search (retrieval)", ""], False
    name = main_pair(res)
    if name:
        t = res["flip_tests"][name]
        va, vb = name.split("_vs_")
        a_only, b_only = t["mcnemar_discordant"].values()
        p = holm.get("flip_mcnemar")
        finding = p is not None and p < ALPHA
        lines += [f"Changing only the {va}/{vb} wording of a question changes the top document in **{t['top1_flips']} of "
                  f"{t['n_pairs']} pairs ({pct(t['flip_rate'])}**, 95% CI {pct(t['flip_rate_wilson_95ci'][0])} to "
                  f"{pct(t['flip_rate_wilson_95ci'][1])}).",
                  "",
                  f"The top document is about a {groups[1]} person for the {vb} wording only in {b_only} pairs, and for "
                  f"the {va} wording only in {a_only} pairs (exact McNemar, Holm-corrected p = {p_text(p)}). "
                  + ("**The search leans one way: this is where the bias enters.**" if finding else
                     "No significant lean in one direction.")]
    c = res["attribute_vs_collection"]
    lines += ["", "Background (these mixes also reflect what the questions are about, so they describe the results "
                  "but do not by themselves show that the wording causes a difference):", "",
              f"Across all questions, the documents found differ from the collection's mix with Cramér's V = "
              f"{c['effect_size']:.3f} (p = {p_text(c['p_raw'])}).{small_cells(c)}", "",
              "| Value | Found | Expected from the collection |", "|---|---:|---:|"]
    lines += [f"| {v} | {c['observed'][v]} | {c['expected'][v]:.1f} |" for v in c["observed"]]
    q = res["retrieved_by_query_attribute"]
    if q:
        lines += ["", f"Questions about {groups[0]} and {groups[1]} people find a different mix of documents "
                      f"(Cramér's V = {q['effect_size']:.3f}, Holm-corrected p = {p_text(holm.get('retrieved_by_query_attribute'))})."
                      f"{small_cells(q)}"]
    return lines, finding


# Layer 2: does the LLM, given the same documents, pick citations by the attribute or by the wording?
def llm_section(res):
    lines = ["## 2. The answer (generation)", ""]
    nf = res["noise_floor_twin_rows"]
    if nf:
        lines += ["", f"Asking the same question twice changes the cited documents in **{pct(nf['cited_rate_mean'])}** of "
                      "cases. This is the noise floor: smaller differences mean nothing."]
    own_bias, amplifies = False, False
    name = main_pair(res)
    if name:
        split = res["input_similarity_split"][name]
        lines += ["", "Twins that found the same documents, split by what the LLM was shown:", "",
                  "| LLM input | Pairs | Cited documents differ | Repeat noise | p |", "|---|---:|---:|---:|---:|"]
        for g, v in split.items():
            if v["n_pairs"]:
                lines.append(f"| {g} | {v['n_pairs']} | {v['rows_differ']}/{v['rows']} ({pct(v['wording_change_rate'])}) | "
                             f"{pct(v['same_pairs_repeat_noise'])} | {p_text(v['p_wilcoxon_vs_repeat_noise'])} |")
        same = split["identical input"]
        own_bias |= bool(same["n_pairs"] and same["p_wilcoxon_vs_repeat_noise"] is not None
                         and same["p_wilcoxon_vs_repeat_noise"] < ALPHA)
        order = split["different order"]
        amplifies = bool(order["n_pairs"] and same["n_pairs"]
                         and order["wording_change_rate"] > 2 * max(same["wording_change_rate"], nf["cited_rate_mean"] if nf else 0))
    lines.append("")
    if not name or not res["input_similarity_split"][name]["identical input"]["n_pairs"]:
        lines.append("**Not measurable:** no twins got identical input, so the LLM's own effect cannot be separated.")
    elif own_bias:
        lines.append("**The LLM adds a bias of its own:** with the same documents in the same order, its citations "
                     "change more often between the twins than between repeat runs.")
    else:
        lines.append("**The LLM adds no bias of its own:** with the same documents in the same order, its citations "
                     "change no more than repeat noise.")
    if amplifies:
        lines.append("It does react to the **order** of the documents: when the search reorders the same documents, the "
                     "citations change far more often. So a small difference in the search becomes a bigger one in the answer.")
    gens = {r: g for r, g in res["cited_vs_not_cited_per_run"].items() if g}
    sig = [r for r, g in gens.items() if g["p_raw"] < ALPHA]
    unreliable = [r for r, g in gens.items() if g["min_expected"] < MIN_EXPECTED]
    lines += ["", f"Background: the documents the answer cites and the ones it leaves out differ in their mix in {len(sig)} of "
                  f"{len(gens)} runs (p per run: {', '.join(p_text(g['p_raw']) for g in gens.values())}). This can come "
                  "from what the documents are about, not their attribute."
              + (f" Unreliable in {len(unreliable)} run(s): a group is too small (expected count below {MIN_EXPECTED})."
                 if unreliable else "")]
    return lines, own_bias


def comparison_section(results):
    comp = results.get("comparison")
    if not comp:
        return []
    base = results["arms"][BASELINE]
    name = main_pair(base)
    lines = ["## Fixes compared with the baseline", "",
             "| Setting | Top document changes | Answers cite different documents | First cited differs | p vs baseline | "
             "Search results same as baseline | Changed documents matching the question's attribute (dropped → added) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    t = base["flip_tests"][name]
    first = next(iter(comp.values()))["twin_answer_change"][name]
    lines.append(f"| baseline | {t['top1_flips']}/{t['n_pairs']} | {first['cited_set']['baseline_rows_differ']}/"
                 f"{first['cited_set']['baseline_rows']} | {first['first_cited']['baseline_rows_differ']}/"
                 f"{first['first_cited']['baseline_rows']} | | 100% | |")
    for arms, c in comp.items():
        t = results["arms"][arms]["flip_tests"].get(name)
        w = c["twin_answer_change"][name]
        d = c["changed_docs"]
        lines.append(f"| {arms} | {t['top1_flips']}/{t['n_pairs'] if t else ''} | {w['cited_set']['rows_differ']}/"
                     f"{w['cited_set']['rows']} ({pct(w['cited_set']['rate'])}) | {w['first_cited']['rows_differ']}/"
                     f"{w['first_cited']['rows']} | {p_text(w['cited_set']['p_wilcoxon_vs_baseline'])} | "
                     f"{pct(d['retrieved_overlap_with_baseline'])} | {pct(d['left_same_attribute_share'])} → "
                     f"{pct(d['entered_same_attribute_share'])} |")
    lines += ["", "How to read it: a good fix brings the top-document changes and the different-citation rate down "
                  "towards repeat noise. In the last column, dropped documents matching the question's attribute more "
                  "often than the added ones means the fix removed attribute-driven results instead of just reshuffling. "
                  "This report does not measure whether the new results are as relevant: check that with your own "
                  "retrieval test (for example recall@k on questions with a known answer) before keeping a fix."]
    return lines


def build(results):
    groups = results["groups"]
    lines = [f"# Bias audit report: {results['attribute']}", "",
             f"Groups compared: **{groups[0]}** and **{groups[1]}**. Settings audited: {', '.join(results['arms'])}.", ""]
    for arms, res in results["arms"].items():
        holm = res["holm_bonferroni"]["p_corrected"]
        lines += [f"# Setting: {arms}", "",
                  f"{len(res['runs'])} run(s) of {res['queries_per_run']} questions. Search results identical in every run "
                  f"for {res['retrieval_identical_across_runs']['identical_queries']}/{res['retrieval_identical_across_runs']['of']} "
                  "questions.", ""]
        ret, ret_found = retrieval_section(res, groups, holm)
        llm, llm_found = llm_section(res)
        where = ("enters at the search" if ret_found and not llm_found else "enters at the LLM" if llm_found and not ret_found
                 else "enters at both the search and the LLM" if ret_found
                 else "is not detected: swapping the wording changes neither the search nor the LLM significantly")
        lines += [f"**In short: the effect of the {results['twin_pairs'][0][0]}/{results['twin_pairs'][0][1]} wording {where}.**",
                  ""] + ret + [""] + llm + [""]
    return "\n".join(lines + comparison_section(results)) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Turn the audit results into a plain-language report.")
    parser.add_argument("--results", default="bias_audit_results.json")
    parser.add_argument("--out", default="bias_audit_report.md")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build(results))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
