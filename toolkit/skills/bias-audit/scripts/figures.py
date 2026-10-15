import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stats import BASELINE, FIXED_DOCUMENTS  # noqa: E402

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
GROUP_COLORS = ("#3498db", "#e74c3c")
OTHER_COLORS = ("#9b59b6", "#95a5a6", "#16a085", "#f39c12")
FIGURES = {
    "query_attribute": "01_question_vs_retrieved.png",
    "twin_grid": "02_twin_top_document_grid.png",
    "flip_summary": "03_flip_summary.png",
    "llm_input": "04_llm_input_similarity.png",
    "fixes": "05_fixes_compared.png",
    "llm_fixed": "06_llm_fixed_documents.png",
}
CAPTIONS = {
    "query_attribute": "Questions about each group: share of found documents about the second group",
    "twin_grid": "Each twin pair: the group of the top document for each wording",
    "flip_summary": "How often changing only the wording changes the top document",
    "llm_input": "Twins that found the same documents: how often the answer cites different ones",
    "fixes": "Each fix compared with the baseline",
    "llm_fixed": "The LLM with fixed documents: how often the he and she answers cite different documents",
}


def colors_for(values, groups):
    others = iter(OTHER_COLORS)
    return {v: GROUP_COLORS[groups.index(v)] if v in groups else next(others, "#dddddd") for v in values}


def save(fig, out_dir, key):
    path = os.path.join(out_dir, FIGURES[key])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_query_attribute(res, groups, out_dir):
    q = res["retrieved_by_query_attribute"]
    if not q:
        return None
    shares = [q["distributions"][g].get(groups[1], 0) / sum(q["distributions"][g].values()) for g in groups]
    expected = res["attribute_vs_collection"]["expected"]
    collection_share = expected.get(groups[1], 0) / sum(expected.values())
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([f"Question is about\n{g}" for g in groups], shares, color=GROUP_COLORS, edgecolor="black", width=0.55)
    for bar, s in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.0%}", ha="center", fontsize=16, fontweight="bold")
    ax.axhline(collection_share, color="#333", linestyle="--", linewidth=1.2, label=f"share in the whole collection ({collection_share:.0%})")
    ax.set_ylim(0, min(1, max(shares + [collection_share]) * 1.35))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(f"Found documents about {groups[1]}", fontweight="bold")
    ax.set_title(f"Documents found for questions about {groups[0]} and {groups[1]}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.5, -0.16, f"Cramér's V = {q['effect_size']:.2f}  •  n = {q['n']} found documents  •  questions without twins",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    return save(fig, out_dir, "query_attribute")


def plot_twin_grid(t, va, vb, groups, out_dir):
    pairs = sorted(t["top_values"])
    if not pairs:
        return None
    values = sorted({v for vs in t["top_values"].values() for v in vs}, key=lambda v: (v not in groups, v))
    color = colors_for(values, groups)
    n_panels = min(4, len(pairs))
    per_panel = -(-len(pairs) // n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels + 1, 0.32 * per_panel + 1.8), squeeze=False)
    for pi, ax in enumerate(axes[0]):
        chunk = pairs[pi * per_panel:(pi + 1) * per_panel]
        for row, pair in enumerate(chunk):
            y = per_panel - row - 1
            top = t["top_values"][pair]
            for col, v in enumerate(top):
                ax.add_patch(plt.Rectangle((col, y), 1, 1, facecolor=color[v], edgecolor="white", linewidth=1.5))
            if pair in t["flipped_pairs"]:
                ax.add_patch(plt.Rectangle((0, y), 2, 1, fill=False, edgecolor="#111", linewidth=2))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, per_panel)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels([va, vb], fontweight="bold")
        ax.set_yticks([per_panel - i - 0.5 for i in range(len(chunk))])
        ax.set_yticklabels(chunk, fontsize=7)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Same question, only the {va}/{vb} wording changed: who is the top document about?", fontweight="bold")
    handles = [Patch(facecolor=color[v], label=f"Top document: {v}") for v in values]
    handles.append(Patch(facecolor="white", edgecolor="#111", label=f"Outline: top document changed ({t['top1_flips']} of {t['n_pairs']})"))
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return save(fig, out_dir, "twin_grid")


def plot_flip_summary(t, va, vb, groups, out_dir):
    n, k = t["n_pairs"], t["top1_flips"]
    if not n:
        return None
    a_only, b_only = t["mcnemar_discordant"].values()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(["Top document"], [n - k], color="#bdc3c7", edgecolor="black", height=0.5)
    ax.barh(["Top document"], [k], left=[n - k], color="#e74c3c", edgecolor="black", height=0.5)
    if n - k:
        ax.text((n - k) / 2, 0, f"{n - k}/{n}\nsame", ha="center", va="center", fontweight="bold")
    if k:
        ax.text(n - k + k / 2, 0, f"{k}/{n}\nchanged", ha="center", va="center", fontweight="bold", color="white")
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel(f"Twin pairs (the same question, only {va}/{vb} changed)", fontweight="bold")
    ax.set_title(f"Changing only {va}/{vb} changes which document is found first", fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.81, 0.5, f"Top document about {groups[1]}\nfor one wording only:\n\n{vb} wording: {b_only}\n{va} wording: {a_only}\n\n"
                        "The search has no randomness,\nso the wording caused\neach change.",
             fontsize=9, va="center", color="#333", linespacing=1.5)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    return save(fig, out_dir, "flip_summary")


def plot_llm_input(split, out_dir, key="llm_input", title="Does the LLM react to the wording, or only to what it is shown?"):
    groups = [g for g, v in split.items() if v["n_pairs"]]
    if not groups:
        return None
    labels = {"different order": "Same documents,\ndifferent order", "same order, different passages shown":
              "Same order,\ndifferent passages", "identical input": "Identical input\nto the LLM"}
    x = np.arange(len(groups))
    rates = [split[g]["wording_change_rate"] for g in groups]
    lo = [r - split[g]["wording_change_wilson_95ci"][0] for r, g in zip(rates, groups)]
    hi = [split[g]["wording_change_wilson_95ci"][1] - r for r, g in zip(rates, groups)]
    noise = [split[g]["same_pairs_repeat_noise"] if split[g]["same_pairs_repeat_noise"] is not None else np.nan for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x, rates, color=["#e67e22", "#f5b041", "#2c3e50"][:len(groups)], edgecolor="black", width=0.6,
           yerr=[lo, hi], capsize=4)
    ax.scatter(x, noise, marker="_", s=900, color="#c0392b", linewidths=2.5, zorder=3, label="same question asked twice (noise)")
    for xi, (r, h) in enumerate(zip(rates, hi)):
        ax.text(xi, min(r + h + 0.03, 0.95), f"{r:.0%}", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[g]}\n({split[g]['n_pairs']} pairs)" for g in groups], fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Answer cites different documents", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    return save(fig, out_dir, key)


def plot_fixes(results, name, out_dir):
    comp = results.get("comparison")
    if not comp:
        return None
    base = results["arms"][BASELINE]
    t = base["flip_tests"][name]
    first = next(iter(comp.values()))["twin_answer_change"][name]["cited_set"]
    settings = [("baseline", t["top1_flips"] / t["n_pairs"], first["baseline_rows_differ"] / first["baseline_rows"])]
    for arms, c in comp.items():
        ta = results["arms"][arms]["flip_tests"][name]
        settings.append((arms, ta["top1_flips"] / ta["n_pairs"] if ta["n_pairs"] else 0, c["twin_answer_change"][name]["cited_set"]["rate"]))
    x = np.arange(len(settings))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(settings) + 2), 4.6))
    for offset, idx, label, color in ((-0.2, 1, "top document changes", "#e74c3c"), (0.2, 2, "answers cite different documents", "#34495e")):
        vals = [s[idx] for s in settings]
        ax.bar(x + offset, vals, width=0.4, color=color, edgecolor="black", label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + offset, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    nf = base["noise_floor_twin_rows"]
    if nf:
        ax.axhline(nf["cited_rate_mean"], color="#c0392b", linestyle="--", linewidth=1.2,
                   label=f"repeat noise of the answers ({nf['cited_rate_mean']:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in settings])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Twin pairs", fontweight="bold")
    ax.set_title("Each fix compared with the baseline (lower is better)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    return save(fig, out_dir, "fixes")


def make_all(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    groups = results["groups"]
    res = results["arms"][BASELINE] if BASELINE in results["arms"] else next(iter(results["arms"].values()))
    name = next(iter(res["flip_tests"]), None)
    made = {"query_attribute": plot_query_attribute(res, groups, out_dir)}
    if name:
        va, vb = name.split("_vs_")
        t = res["flip_tests"][name]
        made["twin_grid"] = plot_twin_grid(t, va, vb, groups, out_dir)
        made["flip_summary"] = plot_flip_summary(t, va, vb, groups, out_dir)
        made["llm_input"] = plot_llm_input(res["input_similarity_split"][name], out_dir)
        made["fixes"] = plot_fixes(results, name, out_dir)
        fixed = results["arms"].get(FIXED_DOCUMENTS)
        if fixed and name in fixed["input_similarity_split"]:
            made["llm_fixed"] = plot_llm_input(fixed["input_similarity_split"][name], out_dir, "llm_fixed",
                                               "Same documents, only he/she changed: does the LLM answer differently?")
    return {k: v for k, v in made.items() if v}


def main():
    parser = argparse.ArgumentParser(description="Draw the audit's charts from results.json (the baseline, and the fixes if any).")
    parser.add_argument("--results", default="bias_audit_results.json")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        made = make_all(json.load(f), args.out_dir)
    print(f"{len(made)} charts in {args.out_dir}")


if __name__ == "__main__":
    main()
