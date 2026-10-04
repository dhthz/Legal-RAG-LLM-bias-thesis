import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = "docs/bias_audit/visualizations"
AUDIT_LOG_PATH = "logs/bias_audit/stateless/run40_A.jsonl"
RESULTS_DIR = "logs/bias_audit/stats_v3_40bases"
MULTI_RUN_PATH = "logs/bias_audit/stats_multi_run/results.json"
LLM_OUTPUT_PATH = "logs/bias_audit/stats_llm_output/results.json"
TRAIN_METADATA_PATH = "dataset/train_with_metadata.jsonl"

# "Needs Manual Classification" dropped from gender charts per owner's call
# (pending a metadata fix); it is a tiny residual class, not a real comparison group.
GENDER_ORDER = ["Male", "Female", "Multiple Applicants", "Unknown"]
GENDER_COLORS = {"Male": "#3498db", "Female": "#e74c3c",
                  "Multiple Applicants": "#9b59b6", "Unknown": "#95a5a6"}
VARIANT_TYPES = ("neutral", "male", "female", "emotional")


class BiasAuditVisualizer:

    def __init__(self, results_dir=RESULTS_DIR, log_path=AUDIT_LOG_PATH, output_dir=OUTPUT_DIR,
                 multi_run_path=MULTI_RUN_PATH, llm_output_path=LLM_OUTPUT_PATH):
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir
        self.master = self._load("master_results.json")
        self.paired_variants = self._load("paired_variant_tests.json")
        self.case_gender = self._load_case_gender()
        self.by_base = self._load_audit_log_grouped(log_path)
        with open(multi_run_path, "r", encoding="utf-8") as f:
            self.multi_run = json.load(f)
        with open(llm_output_path, "r", encoding="utf-8") as f:
            self.llm_output = json.load(f)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _load(self, filename):
        with open(self.results_dir / filename, "r", encoding="utf-8") as f:
            return json.load(f)

    # Gender comes from the current dataset labels by case_id, not from the label stored in the log
    @staticmethod
    def _load_case_gender():
        gender = {}
        with open(TRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gender[rec["case_id"]] = rec.get("classification", {}).get("gender", "Unknown")
        return gender

    def _load_audit_log_grouped(self, log_path):
        by_base = defaultdict(dict)
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                for c in e["retrieved_cases"]:
                    c["gender"] = self.case_gender.get(c["case_id"], c["gender"])
                qid = e["query_id"]
                if "_" not in qid:
                    continue
                base, vtype = qid.rsplit("_", 1)
                if vtype in VARIANT_TYPES:
                    by_base[base][vtype] = e
        return by_base

    # ---- Finding 1: retrieval matches the query's own applicant gender ----

    def plot_retrieved_gender_by_query_gender(self):
        print("Creating query-gender -> retrieved-gender plot...")

        dist = self.master["tests"]["query_gender"]["distributions"]
        t = self.master["tests"]["query_gender"]

        male_total = sum(dist["Male"].values())
        female_total = sum(dist["Female"].values())
        male_female_pct = dist["Male"].get("Female", 0) / male_total * 100
        female_female_pct = dist["Female"].get("Female", 0) / female_total * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            ["Query is about\na MAN", "Query is about\na WOMAN"],
            [male_female_pct, female_female_pct],
            color=[GENDER_COLORS["Male"], GENDER_COLORS["Female"]],
            alpha=0.9, edgecolor="black", width=0.55,
        )
        for bar, pct in zip(bars, [male_female_pct, female_female_pct]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{pct:.0f}%", ha="center", fontsize=20, fontweight="bold")

        ax.set_ylabel("% of retrieved cases about a woman", fontweight="bold", fontsize=12)
        ax.set_ylim(0, max(male_female_pct, female_female_pct) * 1.35)
        ax.set_title(
            "The system retrieves more cases about women\nwhen the question itself is about a woman",
            fontweight="bold", fontsize=14,
        )
        ax.text(0.5, -0.16,
                f"{female_female_pct / male_female_pct:.1f}× difference  •  n={t['n']} retrieved cases  •  p={t['p_raw']:.1e} (Holm-corrected significant)",
                transform=ax.transAxes, ha="center", fontsize=10, color="#555")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = f"{self.output_dir}/01_query_gender_matches_retrieval.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 2: NEUTRAL vs MALE vs FEMALE retrieval, per base case ----

    def plot_gender_variant_retrieval_grid(self):
        print("Creating per-base NEUTRAL/MALE/FEMALE retrieval grid...")

        variants = ["neutral", "male", "female"]
        col_labels = ["Neutral", "Man", "Woman"]
        bases = sorted(self.by_base)
        n_panels = 4
        per_panel = -(-len(bases) // n_panels)

        fig, axes = plt.subplots(1, n_panels, figsize=(11.5, 0.42 * per_panel + 2.0))

        # A base is flagged when its three variants don't all retrieve a top case of the same GENDER (compared by gender, not case_id: two different cases can share a gender)
        n_flagged = 0
        for pi, ax in enumerate(axes):
            chunk = bases[pi * per_panel:(pi + 1) * per_panel]
            for row, base in enumerate(chunk):
                v = self.by_base[base]
                y = per_panel - row - 1
                for col, vt in enumerate(variants):
                    if vt not in v:
                        continue
                    color = GENDER_COLORS.get(v[vt]["retrieved_cases"][0]["gender"], "#dddddd")
                    ax.add_patch(plt.Rectangle((col, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
                if all(vt in v for vt in variants):
                    genders = {v[vt]["retrieved_cases"][0]["gender"] for vt in variants}
                    if len(genders) > 1:
                        n_flagged += 1
                        ax.add_patch(plt.Rectangle((0, y), 3, 1, fill=False, edgecolor="#111111", linewidth=2.6))
            ax.set_xlim(0, 3)
            ax.set_ylim(0, per_panel)
            ax.set_xticks([0.5, 1.5, 2.5])
            ax.set_xticklabels(col_labels, fontsize=9.5, fontweight="bold")
            ax.set_yticks([per_panel - i - 0.5 for i in range(len(chunk))])
            ax.set_yticklabels(chunk, fontsize=10)
            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle("Same legal facts, only the applicant's gender changed", fontweight="bold", fontsize=14)
        legend_handles = [Patch(facecolor=GENDER_COLORS[g], label=f"Top result: {g}") for g in
                          ["Male", "Female", "Multiple Applicants", "Unknown"]]
        legend_handles.append(Patch(facecolor="white", edgecolor="#111111", linewidth=2, label=f"Black outline = gender of top result changed ({n_flagged} of {len(bases)})"))
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False)

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        path = f"{self.output_dir}/02_gender_variant_retrieval_grid.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 3: he/she causal flip summary ----

    def plot_he_she_flip_summary(self):
        print("Creating he/she flip summary plot...")

        r = self.paired_variants["paired_framing_male_vs_female"]
        n, k = r["n_pairs"], r["top1_flips"]
        disc = r["mcnemar_discordant"]
        unchanged, changed = n - k, k

        fig, ax = plt.subplots(figsize=(7.5, 3.6))

        ax.barh(["Result"], [unchanged], color="#bdc3c7", edgecolor="black", height=0.5)
        ax.barh(["Result"], [changed], left=[unchanged], color="#e74c3c", edgecolor="black", height=0.5)

        ax.text(unchanged / 2, 0, f"{unchanged}/{n}\nsame case", ha="center", va="center", fontweight="bold")
        ax.text(unchanged + changed / 2, 0, f"{changed}/{n}\ndifferent case", ha="center", va="center",
                fontweight="bold", color="white")

        ax.set_xlim(0, n)
        ax.set_yticks([])
        ax.set_xlabel("Case pairs (identical facts, only \"he\"/\"she\" changed)", fontweight="bold", fontsize=10)
        ax.set_title("Swapping \"he\" for \"she\" changes which case is found",
                      fontweight="bold", fontsize=13, pad=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        info_text = (
            f"In {changed} of {n} cases, only\n"
            f"changing the applicant's\n"
            f"pronoun changed the top\n"
            f"result.\n\n"
            f"{disc['female_only']} of {changed} switched toward\n"
            f"a female-applicant case;\n"
            f"{disc['male_only']} of {changed} switched away\n"
            f"from one.\n\n"
            f"Retrieval has no randomness,\n"
            f"so the pronoun is the only\n"
            f"thing that could have\n"
            f"caused each change."
        )
        fig.text(0.82, 0.5, info_text, fontsize=9, va="center", color="#333", linespacing=1.6)

        plt.tight_layout(rect=[0, 0, 0.8, 1])
        path = f"{self.output_dir}/03_he_she_flip_summary.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 4: does the LLM cite a different case when only the wording changes? ----

    def plot_llm_input_similarity(self):
        print("Creating LLM citation change by input similarity plot...")

        groups = ["different order", "same order, different chunks shown", "identical input"]
        labels = ["Same 3 cases,\ndifferent order", "Same order,\ndifferent excerpts", "Identical input\nto the AI"]
        panels = [("male_vs_female", "Man vs woman wording"), ("neutral_vs_emotional", "Neutral vs emotional wording")]
        split = self.multi_run["input_similarity_split"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, (pair, title) in zip(axes, panels):
            x = np.arange(len(groups))
            rates, err_lo, err_hi, noise, ns = [], [], [], [], []
            for g in groups:
                v = split[pair][g]
                if not v["n_bases"]:
                    rates.append(0); err_lo.append(0); err_hi.append(0); noise.append(np.nan); ns.append(0)
                    continue
                r = v["wording_change_rate"]
                lo, hi = v["wording_change_wilson_95ci"]
                rates.append(r); err_lo.append(r - lo); err_hi.append(hi - r)
                noise.append(v["same_bases_repeat_noise"]); ns.append(v["n_bases"])
            colors = ["#e67e22", "#f5b041", "#2c3e50"]
            ax.bar(x, rates, color=colors, edgecolor="black", width=0.6,
                   yerr=[err_lo, err_hi], capsize=4, error_kw={"elinewidth": 1})
            ax.scatter(x, noise, marker="_", s=900, color="#c0392b", linewidths=2.5, zorder=3,
                       label="same question asked twice (noise)")
            for xi, r in enumerate(rates):
                ax.text(xi, min(r + err_hi[xi] + 0.03, 0.97), f"{r:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lab}\n({n} cases)" for lab, n in zip(labels, ns)], fontsize=9.5)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title(title, fontweight="bold", fontsize=12)
        axes[0].set_ylabel("AI cited a different case", fontweight="bold")
        axes[1].legend(loc="upper right", fontsize=9, frameon=True)

        fig.suptitle("The AI only changes its citation when retrieval changes what it reads",
                     fontweight="bold", fontsize=13.5)
        fig.text(0.5, 0.01,
                 "Pairs where both wordings retrieved the same 3 cases, 4 full runs. Bars: share of runs where the cited case differed "
                 "between the two wordings (95% CI).\nRed line: how often the same question cited a different case across runs. "
                 "With identical input, wording has no effect; a different order alone changes the citation (position bias).",
                 ha="center", fontsize=8.8, color="#333", linespacing=1.5)
        plt.tight_layout(rect=[0, 0.1, 1, 0.94])
        path = f"{self.output_dir}/04_llm_input_similarity.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Summary table: the concluded findings ----

    def plot_findings_summary_table(self):
        print("Creating findings summary table...")

        hb = self.master["holm_bonferroni"]
        tests = self.master["tests"]
        dist = tests["query_gender"]["distributions"]
        female_pct = dist["Female"].get("Female", 0) / sum(dist["Female"].values())
        male_pct = dist["Male"].get("Female", 0) / sum(dist["Male"].values())
        flips = self.paired_variants["paired_framing_male_vs_female"]
        disc = flips["mcnemar_discordant"]
        split = self.multi_run["input_similarity_split"]
        same_mf = split["male_vs_female"]["identical input"]
        same_ne = split["neutral_vs_emotional"]["identical input"]
        order = split["male_vs_female"]["different order"]
        llm_p = min(self.llm_output["holm_bonferroni"]["p_corrected"].values())

        rows = [
            ("A pronoun alone changes\nthe top result",
             f"\"he\" to \"she\" (nothing else changed) gave a different top\ncase in {flips['top1_flips']} of {flips['n_pairs']} cases; "
             f"{disc['female_only']} of {disc['female_only'] + disc['male_only']} gender changes went toward a woman",
             f"Retrieval, causal\np={hb['p_corrected']['he_she_mcnemar']:.1e} (corrected)"),
            ("Retrieval follows the\nquery's gender",
             f"Asking about a woman returns {female_pct / male_pct:.1f}× more\nfemale-applicant cases than asking about a man",
             f"Retrieval\np={hb['p_corrected']['query_gender']:.1e} (corrected)"),
            ("Country, time and\noutcome skews",
             f"{tests['jurisdiction']['query_country_match_rate']:.0%} of results match the query's country; results are\n"
             f"newer than the corpus; violation cases slightly over-returned",
             "Retrieval\nall significant (corrected)"),
            ("The AI adds no bias\nof its own",
             f"With identical input, wording changed the cited case\n{same_mf['wording_change_rate']:.0%} (gender) and "
             f"{same_ne['wording_change_rate']:.0%} (tone, vs {same_ne['same_bases_repeat_noise']:.0%} noise); predicted articles unchanged",
             f"AI answer\npredicted articles p={llm_p:.1f} (corrected)"),
            ("The AI amplifies\nreordering",
             f"The same cases in a different order changed\nthe cited case {order['wording_change_rate']:.0%} of the time (position bias)",
             f"AI answer\n{order['n_bases']} cases, 4 runs"),
            ("Emotional tone has\nno effect",
             "No change in retrieval, citation, predicted\narticles or the tone of the answer",
             "Both stages\nNRC-verified wording"),
        ]

        fig, ax = plt.subplots(figsize=(13, 0.95 * len(rows) + 1.0))
        ax.axis("off")
        col_labels = ["Finding", "Evidence", "Where / strength"]
        table = ax.table(cellText=[list(r) for r in rows], colLabels=col_labels, cellLoc="left",
                         loc="center", bbox=[0, 0, 1, 1], colWidths=[0.22, 0.54, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for i in range(len(col_labels)):
            table[0, i].set_facecolor("#34495e")
            table[0, i].set_text_props(color="white", fontweight="bold", ha="left")
        # retrieval findings green, AI-stage findings blue, nulls grey
        row_colors = ["#eafaf1", "#eafaf1", "#eafaf1", "#ebf5fb", "#ebf5fb", "#f2f3f4"]
        for r, color in enumerate(row_colors, start=1):
            for c in range(len(col_labels)):
                table[r, c].set_facecolor(color)
                table[r, c].set_text_props(ha="left")
                table[r, c].PAD = 0.02

        ax.set_title("Bias Audit — What We Found", fontweight="bold", fontsize=15, pad=8)
        plt.tight_layout()
        path = f"{self.output_dir}/05_findings_summary_table.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 5: the AI's own verdict on the violated articles, by wording ----

    def plot_predicted_articles_by_wording(self):
        print("Creating predicted-articles accuracy plot...")

        acc = self.llm_output["accuracy_vs_truth"]
        wordings = ["original", "neutral", "male", "female", "emotional"]
        labels = ["Original\ncase text", "Gender\nneutral", "A man", "A woman", "Emotional"]
        metrics = [("key_hit", "Main article found", "#2c3e50"), ("jaccard", "Overlap with real articles", "#5d6d7e"),
                   ("exact", "Exact match", "#aab7b8")]

        fig, ax = plt.subplots(figsize=(11, 4.6))
        x = np.arange(len(wordings))
        w = 0.26
        for i, (key, name, color) in enumerate(metrics):
            vals = [acc[t][key] for t in wordings]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, color=color, edgecolor="black", label=name)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}" if key != "jaccard" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(loc="upper right", fontsize=9, ncol=3, frameon=True)
        ax.set_title("The AI's own verdict on the violated articles does not change with gender or tone",
                     fontweight="bold", fontsize=13, pad=10)
        p = min(self.llm_output["holm_bonferroni"]["p_corrected"].values())
        fig.text(0.5, 0.01,
                 f"40 cases, each worded five ways, 4 full runs averaged; compared with the case's real violated articles. "
                 f"8 paired tests (gender, tone), smallest corrected p = {p:.1f}.",
                 ha="center", fontsize=9, color="#333")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        path = f"{self.output_dir}/06_predicted_articles_by_wording.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()


def main():
    print("=" * 80)
    print(" " * 20 + "BIAS AUDIT VISUALIZATION")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--log", default=AUDIT_LOG_PATH)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    viz = BiasAuditVisualizer(results_dir=args.results_dir, log_path=args.log, output_dir=args.out_dir)
    viz.plot_retrieved_gender_by_query_gender()
    viz.plot_gender_variant_retrieval_grid()
    viz.plot_he_she_flip_summary()
    viz.plot_llm_input_similarity()
    viz.plot_findings_summary_table()
    viz.plot_predicted_articles_by_wording()

    print("\n" + "=" * 80)
    print(f"Done. Saved 6 plots to {viz.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
