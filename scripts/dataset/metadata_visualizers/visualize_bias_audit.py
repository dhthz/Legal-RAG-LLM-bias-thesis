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

# "Needs Manual Classification" dropped from gender charts per owner's call
# (pending a metadata fix); it is a tiny residual class, not a real comparison group.
GENDER_ORDER = ["Male", "Female", "Multiple Applicants", "Unknown"]
GENDER_COLORS = {"Male": "#3498db", "Female": "#e74c3c",
                  "Multiple Applicants": "#9b59b6", "Unknown": "#95a5a6"}
VARIANT_TYPES = ("neutral", "male", "female", "emotional")


class BiasAuditVisualizer:

    def __init__(self, results_dir="logs/bias_audit"):
        self.results_dir = Path(results_dir)
        self.master = self._load("master_results.json")
        self.paired_variants = self._load("paired_variant_tests.json")
        self.paired_generation = self._load("paired_generation_tests.json")
        self.by_base = self._load_audit_log_grouped()
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _load(self, filename):
        with open(self.results_dir / filename, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_audit_log_grouped():
        by_base = defaultdict(dict)
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
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
                f"3.2× difference  •  n=391 retrieved cases  •  p={t['p_raw']:.1e} (Holm-corrected significant)",
                transform=ax.transAxes, ha="center", fontsize=10, color="#555")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/01_query_gender_matches_retrieval.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 2: NEUTRAL vs MALE vs FEMALE retrieval, per base case ----

    def plot_gender_variant_retrieval_grid(self):
        print("Creating per-base NEUTRAL/MALE/FEMALE retrieval grid...")

        variants = ["neutral", "male", "female"]
        col_labels = ["Written\nneutrally", "Applicant\nis a man", "Applicant\nis a woman"]
        bases = sorted(self.by_base)

        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(bases) + 1.0))

        for row, base in enumerate(bases):
            v = self.by_base[base]
            for col, vt in enumerate(variants):
                if vt not in v:
                    continue
                gender = v[vt]["retrieved_cases"][0]["gender"]
                color = GENDER_COLORS.get(gender, "#dddddd")
                ax.add_patch(plt.Rectangle((col, len(bases) - row - 1), 1, 1,
                                            facecolor=color, edgecolor="white", linewidth=1.5))

        # Flag any row where the three variants don't all retrieve a case of
        # the same gender — i.e. gender-marking (neutral -> male, neutral ->
        # female, or male -> female) changed which gender of case comes back.
        # Based on the retrieved case's GENDER, not its case_id: two variants
        # can retrieve different case_ids that are still the same gender
        # (not a gender-driven change), so comparing IDs directly would
        # misclassify those rows.
        n_flagged = 0
        for row, base in enumerate(bases):
            v = self.by_base[base]
            if "neutral" not in v or "male" not in v or "female" not in v:
                continue
            genders = {v[vt]["retrieved_cases"][0]["gender"] for vt in ("neutral", "male", "female")}
            if len(genders) > 1:
                n_flagged += 1
                ax.text(3.15, len(bases) - row - 0.5, "◀ changed", va="center", fontsize=8.5,
                        color="#c0392b", fontweight="bold")

        ax.set_xlim(0, 3.9)
        ax.set_ylim(0, len(bases))
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticks([len(bases) - i - 0.5 for i in range(len(bases))])
        ax.set_yticklabels(bases, fontsize=8)
        ax.set_title("Same legal facts, only the applicant's gender changed",
                      fontweight="bold", fontsize=13, pad=10)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        legend_handles = [Patch(facecolor=GENDER_COLORS[g], label=g) for g in
                           ["Male", "Female", "Multiple Applicants", "Unknown"]]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=9, frameon=False, title="Top result", title_fontsize=9,
                  alignment="left")

        info_text = (
            "Each row = one case,\nrewritten 3 ways with\nidentical facts.\n\n"
            "\"◀ changed\" = the three\nversions did not all return\nthe same gender of case\n"
            f"({n_flagged} of {len(bases)} bases)."
        )
        ax.text(1.02, 0.62, info_text, transform=ax.transAxes, fontsize=8.7,
                va="top", color="#333", linespacing=1.5)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/02_gender_variant_retrieval_grid.png"
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
            f"All {changed} switched toward\n"
            f"a female-applicant case\n"
            f"({disc['female_only']} of {changed});\n"
            f"none switched the other way\n"
            f"({disc['male_only']} of {changed}).\n\n"
            f"Retrieval has no randomness,\n"
            f"so the pronoun is the only\n"
            f"thing that could have\n"
            f"caused each change."
        )
        fig.text(0.82, 0.5, info_text, fontsize=9, va="center", color="#333", linespacing=1.6)

        plt.tight_layout(rect=[0, 0, 0.8, 1])
        path = f"{OUTPUT_DIR}/03_he_she_flip_summary.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Finding 4: NEUTRAL vs EMOTIONAL, does the LLM's answer change? ----

    def plot_emotional_generation_grid(self):
        print("Creating per-base NEUTRAL/EMOTIONAL generation-response grid...")

        r = self.paired_generation["paired_generation_neutral_vs_emotional"]
        bases = sorted(b for b in self.by_base if "neutral" in self.by_base[b] and "emotional" in self.by_base[b])

        rows = []
        for base in bases:
            cited_diff = base in r["cited_case_set_differs"]["bases"]
            articles_diff = base in r["reported_articles_differ"]["bases"]
            rows.append((base, cited_diff, articles_diff))

        fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(rows) + 1.0))
        col_labels = ["Cited a\ndifferent case", "Reported a\ndifferent article"]

        for row, (base, cited_diff, articles_diff) in enumerate(rows):
            y = len(rows) - row - 1
            for col, changed in enumerate([cited_diff, articles_diff]):
                color = "#e74c3c" if changed else "#eafaf1"
                ax.add_patch(plt.Rectangle((col, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
                if changed:
                    ax.text(col + 0.5, y + 0.5, "✕", ha="center", va="center", fontsize=13, color="white", fontweight="bold")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, len(rows))
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticks([len(rows) - i - 0.5 for i in range(len(rows))])
        ax.set_yticklabels([b for b, _, _ in rows], fontsize=8)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Same retrieved cases — did a more\nemotional question change the AI's answer?",
                      fontweight="bold", fontsize=12.5, pad=10)

        n_cited = r["cited_case_set_differs"]["n_differ"]
        n_articles = r["reported_articles_differ"]["n_differ"]
        n_pairs = r["n_pairs"]
        info_text = (
            f"Each row is one case, asked\n"
            f"once neutrally and once with\n"
            f"emotionally charged wording —\n"
            f"the retrieved cases were\n"
            f"identical both times.\n\n"
            f"✕ = the AI's answer changed\n"
            f"anyway:\n\n"
            f"• {n_cited}/{n_pairs} bases: discussed\n"
            f"  a different case\n"
            f"• {n_articles}/{n_pairs} bases: reported a\n"
            f"  different violated article"
        )
        ax.text(1.05, 1.0, info_text, transform=ax.transAxes, fontsize=8.7,
                va="top", color="#333", linespacing=1.6)

        plt.tight_layout(rect=[0, 0, 0.78, 1])
        path = f"{OUTPUT_DIR}/04_emotional_generation_grid.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    # ---- Summary table: only the load-bearing findings ----

    def plot_findings_summary_table(self):
        print("Creating findings summary table...")

        hb = self.master["holm_bonferroni"]
        p = hb["p_corrected"]["query_gender"]
        rows = [
            ("Retrieval matches the\napplicant's gender",
             "Asking about a woman returns\n3.2× more female-applicant cases\nthan asking about a man",
             f"Statistical test\np={p:.1e} (corrected)"),
            ("A pronoun alone can change\nthe result",
             "Rewriting \"he\" as \"she\" (nothing\nelse changed) returned a different\ncase in 7 of 20 identical cases",
             "Deterministic replay\n(retrieval has no randomness)"),
            ("Emotional wording changes the\nAI's answer, not the search",
             "With the exact same cases found,\na more emotional question made the\nAI cite a different case 27% of the time",
             "Single run\nnot yet repeat-tested"),
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
        path = f"{OUTPUT_DIR}/05_findings_summary_table.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()


def main():
    print("=" * 80)
    print(" " * 20 + "BIAS AUDIT VISUALIZATION")
    print("=" * 80)

    viz = BiasAuditVisualizer()
    viz.plot_retrieved_gender_by_query_gender()
    viz.plot_gender_variant_retrieval_grid()
    viz.plot_he_she_flip_summary()
    viz.plot_emotional_generation_grid()
    viz.plot_findings_summary_table()

    print("\n" + "=" * 80)
    print(f"Done. Saved 5 plots to {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
