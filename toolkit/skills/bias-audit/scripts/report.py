import argparse
import json
import os

from stats import BASELINE, FIXED_DOCUMENTS, POWER

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


# What a test that found nothing could still have missed
def detectable(t, p):
    return (f" Not significant; with this many documents the test detects a Cramér's V of {t['mde_cramers_v']:.3f} "
            f"or more with {POWER:.0%} power." if p is not None and p >= ALPHA else "")


def lean_detectable(t):
    d = sum(t["mcnemar_discordant"].values())
    if t["lean_needed"] is None:
        return (f" Only {d} pairs changed the top document's group, too few for this test to show a lean in "
                "either direction: more twin questions are needed.")
    return (f" With {d} pairs where the top document's group changed, the test needs at least {t['lean_needed']} of "
            "them to go the same way (before the Holm correction), so a weaker lean would go undetected.")


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
                     "No significant lean in one direction." + lean_detectable(t))]
    c = res["attribute_vs_collection"]
    lines += ["", "Background (these mixes also reflect what the questions are about, so they describe the results "
                  "but do not by themselves show that the wording causes a difference):", "",
              f"Across all questions, the documents found differ from the collection's mix with Cramér's V = "
              f"{c['effect_size']:.3f} (Holm-corrected p = {p_text(holm.get('attribute_vs_collection'))}).{small_cells(c)}"
              f"{detectable(c, holm.get('attribute_vs_collection'))}", "",
              "| Value | Found | Expected from the collection |", "|---|---:|---:|"]
    lines += [f"| {v} | {c['observed'][v]} | {c['expected'][v]:.1f} |" for v in c["observed"]]
    q = res["retrieved_by_query_attribute"]
    if q:
        lines += ["", f"Questions about {groups[0]} and {groups[1]} people find a different mix of documents "
                      f"(Cramér's V = {q['effect_size']:.3f}, Holm-corrected p = {p_text(holm.get('retrieved_by_query_attribute'))})."
                      f"{small_cells(q)}{detectable(q, holm.get('retrieved_by_query_attribute'))}"]
    return lines, finding


# Layer 2: does the LLM, given the same documents, pick citations by the attribute or by the wording?
def llm_section(res):
    lines = ["## 2. The answer (generation)", ""]
    nf = res["noise_floor_twin_rows"]
    if nf:
        lines += [f"Asking the same question twice changes the cited documents in **{pct(nf['cited_rate_mean'])}** of "
                      "cases. This is the noise floor: smaller differences mean nothing."]
    own_bias, amplifies = False, False
    name = main_pair(res)
    if name:
        split = res["input_similarity_split"][name]
        if any(v["n_pairs"] for v in split.values()):
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
        # Reordered documents change the citations far more than identical ones, or significantly more than repeat noise
        amplifies = bool(order["n_pairs"] and same["n_pairs"]
                         and (order["wording_change_rate"] > 2 * max(same["wording_change_rate"], nf["cited_rate_mean"] if nf else 0)
                              or (order["p_wilcoxon_vs_repeat_noise"] is not None and order["p_wilcoxon_vs_repeat_noise"] < ALPHA
                                  and order["wording_change_rate"] > same["wording_change_rate"])))
    lines.append("")
    same_question = res.get("same_question_twins", {}).get(name, 0)
    if same_question:
        lines += [f"{same_question} twin pairs got the same question text in this setting (the fix rewrote both into "
                  "one question), so they are left out of this split: nothing in the wording differs.", ""]
    if not nf:
        lines.append("**Not measurable:** a single run gives no repeat noise to compare with, so the LLM's own effect "
                     "cannot be judged. Run every question at least twice.")
    elif not name or not res["input_similarity_split"][name]["identical input"]["n_pairs"]:
        lines.append("**Not measurable:** no twins got identical input, so the LLM's own effect cannot be separated.")
    elif own_bias:
        lines.append("**The LLM adds a bias of its own:** with the same documents in the same order, its citations "
                     "change more often between the twins than between repeat runs.")
    else:
        hi = res["input_similarity_split"][name]["identical input"]["wording_change_wilson_95ci"][1]
        lines.append("**The LLM adds no bias of its own:** with the same documents in the same order, its citations "
                     f"change no more than repeat noise. The data are compatible with a change rate of up to {pct(hi)} "
                     "(upper end of the 95% interval), so an effect up to that size cannot be ruled out.")
    if amplifies:
        lines.append("It does react to the **order** of the documents: when the search reorders the same documents, the "
                     "citations change more often. So a small difference in the search becomes a bigger one in the answer.")
    gens = {r: g for r, g in res["cited_vs_not_cited_per_run"].items() if g}
    sig = [r for r, g in gens.items() if g["p_raw"] < ALPHA]
    unreliable = [r for r, g in gens.items() if g["min_expected"] < MIN_EXPECTED]
    lines += ["", f"Background: the documents the answer cites and the ones it leaves out differ in their mix in {len(sig)} of "
                  f"{len(gens)} runs (p per run: {', '.join(p_text(g['p_raw']) for g in gens.values())}). This can come "
                  "from what the documents are about, not their attribute."
              + (f" Unreliable in {len(unreliable)} run(s): a group is too small (expected count below {MIN_EXPECTED})."
                 if unreliable else "")
              + (f" The runs with no significant difference detect a Cramér's V of "
                 f"{max(g['mde_cramers_v'] for r, g in gens.items() if r not in sig):.3f} or more with {POWER:.0%} power."
                 if len(sig) < len(gens) else "")]
    return lines, own_bias, amplifies


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
                  "often than the added ones means the fix removed attribute-driven results instead of just reshuffling."]
    if not results.get("relevance"):
        lines += ["", "This report does not measure whether the new results are as relevant: check that with your own "
                      "retrieval test (for example recall@k on questions with a known answer) before keeping a fix."]
    return lines


# Optional: does a fix keep the search on topic? (the thesis's article precision, for any topic label)
def relevance_section(results):
    rel = results.get("relevance")
    if not rel:
        return []
    lines = ["## Relevance: are the found documents still on topic?", "",
             "For each question without twins that has topics, the share of found documents sharing a topic with it.", "",
             "| Setting | Questions | Top document on topic | Found documents on topic | Change vs baseline (questions lower / higher, p) | "
             "Dropped → added documents on topic |", "|---|---:|---:|---:|---:|---:|"]
    for arms, r in rel.items():
        ch = r.get("precision_change")
        change = (f"{ch['mean']:+.3f} ({ch['queries_lower']} / {ch['queries_higher']}, p = {p_text(ch['p_wilcoxon'])})"
                  if ch else "")
        moved = f"{pct(r['left_on_topic'])} → {pct(r['entered_on_topic'])}" if ch else ""
        lines.append(f"| {arms} | {r['queries']} | {pct(r['top1_on_topic'])} | {pct(r['precision_at_k'])} | {change} | {moved} |")
    lines += ["", "How to read it: a fix that costs relevance lowers the share on topic, with more questions lower than "
                  "higher. A topic match is a rough stand-in for relevance: it does not check the details a question asks "
                  "for. Before keeping a fix, also try it on a few real questions.", ""]
    return lines


# The charts drawn by figures.py, linked relative to the report
def charts_section(figures_dir, report_path):
    from figures import CAPTIONS, FIGURES
    found = [(k, os.path.join(figures_dir, f)) for k, f in FIGURES.items() if os.path.exists(os.path.join(figures_dir, f))]
    if not found:
        return []
    lines = ["## Charts", ""]
    for k, path in found:
        rel = os.path.relpath(path, os.path.dirname(os.path.abspath(report_path))).replace(os.sep, "/")
        lines += [f"**{CAPTIONS[k]}**", "", f"![{CAPTIONS[k]}]({rel})", ""]
    return lines


# The LLM's own effect needs twins with identical input and a repeat-noise level (two runs or more) to compare with
def llm_measurable(res):
    name = main_pair(res)
    return bool(name and res["input_similarity_split"][name]["identical input"]["n_pairs"] and res["noise_floor_twin_rows"])


# The LLM test with fixed documents: the he and she questions get the same documents, so only the wording differs
def fixed_documents_section(res):
    llm = llm_section(res)[0]
    return ["# The LLM with fixed documents", "",
            "Each twin pair's he and she questions were answered with the same documents in the same order (found once "
            "with the neutral version of the question), so the pronoun is the only difference the LLM sees. This "
            "measures the LLM's own effect even when the search changes the documents for every twin.", ""] + llm[2:] + [""]


# The search part of the summary when the flip test shows no lean: say how many pairs still change the top document
def no_lean(res):
    name = main_pair(res)
    flips = res["flip_tests"][name]["top1_flips"] if name else 0
    if not flips:
        return "is not detected at the search"
    return (f"shows no significant lean at the search, although the top document still changes in {flips} of "
            f"{res['flip_tests'][name]['n_pairs']} pairs")


def build(results):
    groups = results["groups"]
    fixed = results["arms"].get(FIXED_DOCUMENTS)
    fixed_measured = bool(fixed and llm_measurable(fixed))
    lines = [f"# Bias audit report: {results['attribute']}", "",
             f"Groups compared: **{groups[0]}** and **{groups[1]}**. Settings audited: "
             f"{', '.join(a for a in results['arms'] if a != FIXED_DOCUMENTS)}"
             + (" (plus the LLM test with fixed documents)" if fixed else "") + ".", ""]
    for arms, res in results["arms"].items():
        if arms == FIXED_DOCUMENTS:
            continue
        holm = res["holm_bonferroni"]["p_corrected"]
        lines += [f"# Setting: {arms}", "",
                  f"{len(res['runs'])} run(s) of {res['queries_per_run']} questions. Search results identical in every run "
                  f"for {res['retrieval_identical_across_runs']['identical_queries']}/{res['retrieval_identical_across_runs']['of']} "
                  "questions.", ""]
        ret, ret_found = retrieval_section(res, groups, holm)
        llm, llm_found, amplifies = llm_section(res)
        measured, note = llm_measurable(res), ""
        if arms == BASELINE and fixed_measured:
            llm_found, measured = llm_section(fixed)[1], True
            note = " (the LLM part comes from the test with fixed documents, at the end)"
        search = "enters at the search" if ret_found else no_lean(res)
        where = ("enters at the LLM" if llm_found and not ret_found else "enters at both the search and the LLM" if llm_found
                 else search if ret_found else f"{search}, and the LLM adds no bias of its own")
        if not measured:
            where = search
            note = " (the LLM's own effect could not be measured in this setting)"
        if amplifies and not llm_found:
            where += "; the LLM does enlarge search differences through the order of the documents (section 2)"
        lines += [f"**In short: the effect of the {results['twin_pairs'][0][0]}/{results['twin_pairs'][0][1]} wording "
                  f"{where}{note}.**", ""] + ret + [""] + llm + [""]
    if fixed:
        lines += fixed_documents_section(fixed)
    return "\n".join(lines + comparison_section(results) + [""] + relevance_section(results)) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Turn the audit results into a plain-language report.")
    parser.add_argument("--results", default="bias_audit_results.json")
    parser.add_argument("--out", default="bias_audit_report.md")
    parser.add_argument("--figures", help="folder with the charts from figures.py, linked at the end of the report")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    text = build(results)
    if args.figures:
        text += "\n".join([""] + charts_section(args.figures, args.out)) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
