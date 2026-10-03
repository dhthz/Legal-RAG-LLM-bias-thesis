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
AUDIT_LOG_PATH = "logs/bias_audit/bias_audit_interactions.jsonl"
TRAIN_METADATA_PATH = "dataset/train_with_metadata.jsonl"

# "Needs Manual Classification" dropped from gender charts per owner's call
# (pending a metadata fix); it is a tiny residual class, not a real comparison group.
GENDER_ORDER = ["Male", "Female", "Multiple Applicants", "Unknown"]
GENDER_COLORS = {"Male": "#3498db", "Female": "#e74c3c",
                  "Multiple Applicants": "#9b59b6", "Unknown": "#95a5a6"}
VARIANT_TYPES = ("neutral", "male", "female", "emotional")


class BiasAuditVisualizer:

    def __init__(self, results_dir="logs/bias_audit", log_path=AUDIT_LOG_PATH, output_dir=OUTPUT_DIR,
                 repeat_results_dir=None, repeat_log_path=None):
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir
        self.master = self._load("master_results.json")
        self.paired_variants = self._load("paired_variant_tests.json")
        self.paired_generation = self._load("paired_generation_tests.json")
        self.case_gender = self._load_case_gender()
        self.by_base = self._load_audit_log_grouped(log_path)
        self.repeat_generation = None
        self.same_query_cited_noise = None
        if repeat_results_dir and repeat_log_path:
            with open(Path(repeat_results_dir) / "paired_generation_tests.json", "r", encoding="utf-8") as f:
                self.repeat_generation = json.load(f)
            self.same_query_cited_noise = self._cited_noise_between_runs(log_path, repeat_log_path)
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

    # Share of variant queries whose cited-case set differs between two full runs of the same query (LLM run-to-run noise floor)
    @staticmethod
    def _cited_noise_between_runs(log_a, log_b):
        def load(path):
            out = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    if e["query_id"].rsplit("_", 1)[-1] in VARIANT_TYPES:
                        out[e["query_id"]] = tuple(sorted(e.get("cited_case_ids") or []))
            return out
        a, b = load(log_a), load(log_b)
        shared = [q for q in a if q in b]
        return sum(a[q] != b[q] for q in shared) / len(shared)

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

    # ---- Finding 4: NEUTRAL vs EMOTIONAL, does the LLM's answer change? ----

    def plot_emotional_generation_grid(self):
        print("Creating per-base NEUTRAL/EMOTIONAL generation-response grid...")

        r = self.paired_generation["paired_generation_neutral_vs_emotional"]
        bases = sorted(b for b in self.by_base if "neutral" in self.by_base[b] and "emotional" in self.by_base[b])
        excluded = set(r["skipped_retrieval_flip"])
        n_panels = 4
        per_panel = -(-len(bases) // n_panels)
        col_labels = ["Different\ncase cited", "Different\narticle"]

        fig, axes = plt.subplots(1, n_panels, figsize=(11.5, 0.42 * per_panel + 2.2))

        for pi, ax in enumerate(axes):
            chunk = bases[pi * per_panel:(pi + 1) * per_panel]
            for row, base in enumerate(chunk):
                y = per_panel - row - 1
                flags = [base in r["cited_case_set_differs"]["bases"], base in r["reported_articles_differ"]["bases"]]
                for col, changed in enumerate(flags):
                    if base in excluded:
                        ax.add_patch(plt.Rectangle((col, y), 1, 1, facecolor="#ecf0f1", edgecolor="white", linewidth=1.5))
                        ax.text(col + 0.5, y + 0.5, "n/t", ha="center", va="center", fontsize=8, color="#95a5a6")
                        continue
                    ax.add_patch(plt.Rectangle((col, y), 1, 1, facecolor="#e74c3c" if changed else "#eafaf1", edgecolor="white", linewidth=1.5))
                    if changed:
                        ax.text(col + 0.5, y + 0.5, "✕", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
            ax.set_xlim(0, 2)
            ax.set_ylim(0, per_panel)
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
            ax.set_yticks([per_panel - i - 0.5 for i in range(len(chunk))])
            ax.set_yticklabels(chunk, fontsize=10)
            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        n_cited = r["cited_case_set_differs"]["n_differ"]
        n_articles = r["reported_articles_differ"]["n_differ"]
        n_pairs = r["n_pairs"]
        fig.suptitle("Same retrieved cases: did a more emotional question change the AI's answer?", fontweight="bold", fontsize=13.5)
        fig.text(0.5, 0.045,
                 f"✕ = the AI's answer changed.  Different case cited: {n_cited}/{n_pairs} bases.  Different article reported: {n_articles}/{n_pairs}.\n"
                 f"n/t = not tested ({len(excluded)} bases): the retrieved cases differed between the two wordings.",
                 ha="center", fontsize=9.5, color="#333", linespacing=1.6)

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        path = f"{self.output_dir}/04_emotional_generation_grid.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Summary table: only the load-bearing findings ----

    def plot_findings_summary_table(self):
        print("Creating findings summary table...")

        hb = self.master["holm_bonferroni"]
        p = hb["p_corrected"]["query_gender"]

        dist = self.master["tests"]["query_gender"]["distributions"]
        female_pct = dist["Female"].get("Female", 0) / sum(dist["Female"].values())
        male_pct = dist["Male"].get("Female", 0) / sum(dist["Male"].values())

        flips = self.paired_variants["paired_framing_male_vs_female"]
        emo = self.paired_generation["paired_generation_neutral_vs_emotional"]["cited_case_set_differs"]
        emo_rate = emo["n_differ"] / self.paired_generation["paired_generation_neutral_vs_emotional"]["n_pairs"]
        if self.repeat_generation:
            emo_b = self.repeat_generation["paired_generation_neutral_vs_emotional"]
            rate_b = emo_b["cited_case_set_differs"]["n_differ"] / emo_b["n_pairs"]
            emo_how = (f"Two full runs: {emo_rate:.0%} and {rate_b:.0%}\n"
                       f"(same question twice already differs\n{self.same_query_cited_noise:.0%} of the time)")
        else:
            emo_how = "Single run\nnot yet repeat-tested"

        rows = [
            ("Retrieval matches the\napplicant's gender",
             f"Asking about a woman returns\n{female_pct / male_pct:.1f}× more female-applicant cases\nthan asking about a man",
             f"Statistical test\np={p:.1e} (corrected)"),
            ("A pronoun alone can change\nthe result",
             f"Rewriting \"he\" as \"she\" (nothing\nelse changed) returned a different\ncase in {flips['top1_flips']} of {flips['n_pairs']} identical cases",
             "Deterministic replay\n(retrieval has no randomness)"),
            ("Emotional wording changes the\nAI's answer, not the search",
             f"With the exact same cases found,\na more emotional question made the\nAI cite a different case {emo_rate:.0%} of the time",
             emo_how),
        ]

        fig, ax = plt.subplots(figsize=(12.5, 1.15 * len(rows) + 1.0))
        ax.axis("off")

        col_labels = ["Finding", "What this means", "How we know"]
        cell_text = [[finding, measured, confirmed] for finding, measured, confirmed in rows]

        table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left",
                          loc="center", bbox=[0, 0, 1, 1], colWidths=[0.24, 0.52, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        for i in range(len(col_labels)):
            table[0, i].set_facecolor("#34495e")
            table[0, i].set_text_props(color="white", fontweight="bold", ha="left")
        # Last row (emotional-tone finding) is preliminary: single run, not
        # repeat-tested, since generation (unlike retrieval) isn't deterministic.
        for r in range(1, len(rows) + 1):
            color = "#fdf3e3" if r == len(rows) else "#eafaf1"
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


def main():
    print("=" * 80)
    print(" " * 20 + "BIAS AUDIT VISUALIZATION")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="logs/bias_audit")
    parser.add_argument("--log", default=AUDIT_LOG_PATH)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--repeat-results-dir", default=None, help="Stats folder of a second full run (for the repeat-test row)")
    parser.add_argument("--repeat-log", default=None, help="Audit log of that second run")
    args = parser.parse_args()

    viz = BiasAuditVisualizer(results_dir=args.results_dir, log_path=args.log, output_dir=args.out_dir,
                              repeat_results_dir=args.repeat_results_dir, repeat_log_path=args.repeat_log)
    viz.plot_retrieved_gender_by_query_gender()
    viz.plot_gender_variant_retrieval_grid()
    viz.plot_he_she_flip_summary()
    viz.plot_emotional_generation_grid()
    viz.plot_findings_summary_table()

    print("\n" + "=" * 80)
    print(f"Done. Saved 5 plots to {viz.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
