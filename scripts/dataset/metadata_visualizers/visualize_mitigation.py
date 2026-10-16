import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Phase F charts, numbered after the audit's 01-06; drawn from the tracked mitigation_results.json
# (scripts/eval/export_mitigation_results.py), so they need no logs.
RESULTS_PATH = "docs/bias_audit/results/mitigation_results.json"
OUTPUT_DIR = "docs/bias_audit/visualizations"

LABELS = {"baseline": "Baseline (and permutation)", "blind_query": "Blind query", "pcf": "PCF",
          "blind_index": "Blind index", "permutation": "Permutation", "leace_query": "LEACE on the query",
          "ft_ccd": "Fine-tune, CCD", "cda": "Fine-tune, CDA", "leace_case_gender": "LEACE on case gender",
          "cda_first_build": "Fine-tune, CDA (first build)"}
COLORS = {"baseline": "#2c3e50", "blind_query": "#27ae60", "pcf": "#16a085", "blind_index": "#2980b9",
          "permutation": "#8e44ad", "leace_query": "#d35400", "ft_ccd": "#c0392b", "cda": "#e67e22",
          "leace_case_gender": "#95a5a6", "cda_first_build": "#bdc3c7"}
# Label offsets (points) so the labels near zero flips do not overlap
OFFSETS = {"baseline": (-8, 8), "blind_query": (2, -20), "pcf": (34, -4), "blind_index": (8, 6),
           "leace_query": (8, -14), "ft_ccd": (8, 6), "cda": (8, -4), "leace_case_gender": (-10, 8),
           "cda_first_build": (8, -4)}
TRADEOFF_ARMS = ["baseline", "blind_query", "pcf", "blind_index", "leace_query", "ft_ccd", "cda",
                 "leace_case_gender", "cda_first_build"]


def save(fig, out_dir, name):
    path = f"{out_dir}/{name}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Bias removed (he/she top-case flips on the 300-pair swap set) against relevance kept (article precision@3)
def plot_tradeoff(arms, out_dir):
    fig, ax = plt.subplots(figsize=(11, 7))
    base = arms["baseline"]
    base_prec = base["r5_retrieval_cost"]["audit_article_precision_at_3"]
    for name in TRADEOFF_ARMS:
        b = arms[name]
        flip = b["r4_swap_set_300"]["top1_flip"]["all"]
        x, lo, hi = (100 * v for v in (flip["rate"], *flip["wilson_95ci"]))
        y = b["r5_retrieval_cost"]["audit_article_precision_at_3"]
        p = (b["r5_retrieval_cost"].get("article_precision_change") or {}).get("p_wilcoxon")
        costly = p is not None and p < 0.05
        documented = b["status"].startswith("documented")
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="none", ecolor=COLORS[name], alpha=0.5, capsize=3)
        ax.scatter(x, y, s=140, color="white" if documented else COLORS[name], edgecolors=COLORS[name],
                   linewidths=2, marker="D" if costly else "o", zorder=3)
        if name == "blind_query":
            continue
        label = "PCF and blind query (1 and 0 of 300)" if name == "pcf" else LABELS[name]
        ax.annotate(label, (x, y), xytext=OFFSETS[name], textcoords="offset points", fontsize=10,
                    ha="right" if OFFSETS[name][0] < 0 else "left", color=COLORS[name] if not documented else "#7f8c8d")
    ax.axhline(base_prec, color="#2c3e50", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(-2, 45)
    ax.set_xlabel("He/she swap changes the top case (% of 300 pairs, 95% CI)")
    ax.set_ylabel("Article precision@3 on the audit questions")
    ax.set_title("Phase F trade-off: bias removed vs relevance kept", fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], marker="o", color="w", markeredgecolor="#555", markerfacecolor="#555", markersize=10,
               label="precision not significantly lower than baseline"),
        Line2D([], [], marker="D", color="w", markeredgecolor="#555", markerfacecolor="#555", markersize=9,
               label="precision significantly lower (Wilcoxon p < 0.05)"),
        Line2D([], [], marker="o", color="w", markeredgecolor="#7f8c8d", markerfacecolor="white", markersize=10,
               label="documented build that failed (not a final fix)"),
        Line2D([], [], color="#2c3e50", linestyle="--", label="baseline precision")], loc="center right", fontsize=9)
    fig.text(0.5, -0.02, "Good fixes sit top left: few flips, precision at the baseline line. "
             "Combinations are in table 11 (they add nothing over their best single fix).",
             fontsize=9, color="#555", ha="center", va="top")
    save(fig, out_dir, "07_fixes_tradeoff.png")


# Whole system: how often the he and she answers cite different cases, per fix, against repeat-run noise
def plot_whole_system(arms, out_dir):
    names = sorted((n for n, b in arms.items() if b.get("whole_system")),
                   key=lambda n: arms[n]["whole_system"]["he_she_cited_set"]["rate"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        ws = arms[name]["whole_system"]
        m, noise = ws["he_she_cited_set"], ws["repeat_noise_cited"]
        rate, lo, hi = (100 * v for v in (m["rate"], *m["wilson_95ci"]))
        ax.barh(i, rate, color=COLORS[name], alpha=0.85, edgecolor="black")
        ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black", capsize=4)
        ax.plot([100 * noise["rate_mean"]] * 2, [i - 0.4, i + 0.4], color="black", linewidth=2.5)
        p = m.get("p_wilcoxon_vs_baseline")
        note = f"{m['rows_differ']}/{m['rows']} ({rate:.1f}%)" + (f", p = {p:.2g} vs baseline" if p is not None else "")
        ax.text(hi + 1, i, note, va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n].replace(" (and permutation)", "") for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("He and she answers cite different cases (% of 160 rows: 40 bases x 4 runs, 95% CI)")
    ax.set_title("Whole system: does the fix reach the LLM's answer?", fontweight="bold")
    ax.legend(handles=[Line2D([], [], color="black", linewidth=2.5,
                              label="repeat noise: the same question asked in two runs")], loc="lower right", fontsize=9)
    save(fig, out_dir, "08_fixes_whole_system.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        arms = json.load(f)["arms"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_tradeoff(arms, args.out_dir)
    plot_whole_system(arms, args.out_dir)


if __name__ == "__main__":
    main()
