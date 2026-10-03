import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = "docs/llm_queries/visualizations"
GENDER_COLORS = {"Male": "#3498db", "Female": "#e74c3c",
                  "Multiple Applicants": "#9b59b6", "Unknown": "#95a5a6"}


class LLMQueriesVisualizer:

    def __init__(self, queries_path, variants_path):
        self.queries = self.load_jsonl(queries_path)
        self.variants = self.load_jsonl(variants_path)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def load_jsonl(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def plot_gender_by_article(self):
        print("Creating gender-by-article stratification plot...")

        articles = sorted(set(r["article"] for r in self.queries),
                           key=lambda a: -sum(1 for r in self.queries if r["article"] == a))
        genders = ["Male", "Female", "Multiple Applicants", "Unknown"]

        counts = {a: Counter(r["gender"] for r in self.queries if r["article"] == a)
                  for a in articles}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Bias Audit Query Set: Gender Stratification by Article (n=175)",
                     fontsize=15, fontweight="bold")

        bottom = np.zeros(len(articles))
        for g in genders:
            vals = [counts[a][g] for a in articles]
            ax1.bar([f"Art {a}" for a in articles], vals, bottom=bottom, label=g,
                    color=GENDER_COLORS[g], alpha=0.85, edgecolor="black")
            bottom += vals
        ax1.set_ylabel("Number of queries", fontweight="bold")
        ax1.set_title("Counts", fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=20)

        bottom = np.zeros(len(articles))
        for g in genders:
            totals = [sum(counts[a].values()) for a in articles]
            vals = [counts[a][g] / t * 100 if t else 0 for a, t in zip(articles, totals)]
            ax2.bar([f"Art {a}" for a in articles], vals, bottom=bottom, label=g,
                    color=GENDER_COLORS[g], alpha=0.85, edgecolor="black")
            bottom += vals
        ax2.set_ylabel("Percentage (%)", fontweight="bold")
        ax2.set_title("Proportions", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.tick_params(axis="x", rotation=20)

        fig.text(0.5, -0.02,
                  "Article share is corpus-proportional (train_with_metadata.jsonl), not equal-per-article;\n"
                  "gender quotas are global, spread across articles by real per-article availability (arXiv:2312.04745).",
                  ha="center", fontsize=9, style="italic")

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/01_gender_by_article.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    def plot_global_gender_quotas(self):
        print("Creating global gender quota achievement plot...")

        quotas = {"Female": 60, "Multiple Applicants": 40, "Unknown": 15}
        counts = Counter(r["gender"] for r in self.queries)

        classes = ["Male", "Female", "Multiple Applicants", "Unknown"]
        achieved = [counts.get(c, 0) for c in classes]
        target = [None, quotas["Female"], quotas["Multiple Applicants"], quotas["Unknown"]]

        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(len(classes))
        bars = ax.bar(x, achieved, color=[GENDER_COLORS[c] for c in classes],
                      alpha=0.85, edgecolor="black", width=0.55, label="Achieved")

        for i, (t, c) in enumerate(zip(target, classes)):
            if t is not None:
                ax.plot([i - 0.3, i + 0.3], [t, t], color="black", linewidth=2.5,
                        linestyle="--", zorder=5,
                        label="Quota target" if i == 1 else None)

        for bar, val in zip(bars, achieved):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}",
                    ha="center", va="bottom", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Number of queries", fontweight="bold")
        ax.set_title("Global Gender Quotas: Target vs. Achieved (n=175)\n"
                      "Equal Male/Female allocation follows arXiv:2312.04745's power argument",
                      fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)



        plt.tight_layout()
        path = f"{OUTPUT_DIR}/02_gender_quota_achievement.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    def plot_country_diversity(self):
        print("Creating country diversity plot...")

        counts = Counter(r["country"] for r in self.queries)
        top = counts.most_common(12)
        labels = [c for c, _ in top] + (["Other"] if len(counts) > 12 else [])
        values = [v for _, v in top]
        if len(counts) > 12:
            values.append(sum(v for _, v in counts.most_common()[12:]))

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(labels)))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       alpha=0.9, edgecolor="black")
        ax.set_xlabel("Number of queries", fontweight="bold")
        ax.set_title(f"Country Diversity in Bias Audit Query Set\n"
                     f"({len(counts)} distinct countries across 175 queries)",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {val}",
                    va="center", fontsize=9)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/03_country_diversity.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    def plot_query_length_distribution(self):
        print("Creating query length distribution plot...")

        word_counts = [len(r["query_text"].split()) for r in self.queries]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(word_counts, bins=30, color="steelblue", alpha=0.75, edgecolor="black")
        mean_v, med_v = np.mean(word_counts), np.median(word_counts)
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_v:.0f}")
        ax.axvline(med_v, color="green", linestyle="--", linewidth=2, label=f"Median: {med_v:.0f}")
        ax.axvline(6000, color="black", linestyle=":", linewidth=2,
                   label="Truncation cap: 6000")

        ax.set_xlabel("Query length (words)", fontweight="bold")
        ax.set_ylabel("Number of queries", fontweight="bold")
        n_capped = sum(1 for w in word_counts if w >= 6000)
        ax.set_title(f"Query Length Distribution (n=175)\n"
                     f"No sanitization: full case facts kept, names/details intact "
                     f"({n_capped} of 175 hit the truncation cap)",
                     fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/04_query_length_distribution.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()

    def plot_variant_manipulation_check(self):
        print("Creating variant-subset manipulation-check plot...")

        by_base = {}
        for r in self.variants:
            by_base.setdefault(r["base_query_id"], {})[r["variant_type"]] = r

        deltas = []
        growth = []
        for qid, v in by_base.items():
            n = v["neutral"]["sentiment_info"]["nrc_emotional_intensity"]
            e = v["emotional"]["sentiment_info"]["nrc_emotional_intensity"]
            deltas.append((qid, (e / n - 1) * 100 if n else float("inf")))
            wn = len(v["neutral"]["query_text"].split())
            we = len(v["emotional"]["query_text"].split())
            growth.append((we - wn) / wn * 100)

        deltas.sort(key=lambda x: x[1])

        n_bases = len(by_base)
        n_pass = sum(1 for _, d in deltas if d >= 6.0)
        n_drift = sum(1 for g in growth if abs(g) > 3.0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 0.3 * n_bases + 2.5))
        fig.suptitle(f"Query-Variant Subset: EMOTIONAL vs. NEUTRAL Manipulation Check "
                     f"({n_bases} hand-authored bases)", fontsize=14, fontweight="bold")

        qids = [q for q, _ in deltas]
        vals = [d for _, d in deltas]
        colors = ["#2ecc71" if d >= 6.0 else "#e74c3c" for d in vals]
        bars = ax1.barh(qids, vals, color=colors, alpha=0.85, edgecolor="black")
        ax1.axvline(6.0, color="black", linestyle="--", linewidth=2,
                    label="6% acceptance threshold")
        ax1.set_xlabel("NRC emotional intensity delta (%)", fontweight="bold")
        ax1.set_title(f"Per-base intensity gain\n({n_pass} of {n_bases} pass the pre-registered gate)",
                      fontweight="bold")
        ax1.legend(loc="lower right")
        ax1.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {val:+.1f}%",
                     va="center", fontsize=8)
        ax1.tick_params(axis="y", labelsize=8)

        ax2.scatter(vals, growth, s=70, color="#9b59b6", alpha=0.8, edgecolor="black")
        ax2.axhline(0, color="gray", linestyle="-", linewidth=1)
        ax2.axvline(6.0, color="black", linestyle="--", linewidth=1.5,
                    label="6% threshold")
        ax2.set_xlabel("NRC emotional intensity delta (%)", fontweight="bold")
        ax2.set_ylabel("Word-count growth, EMOTIONAL vs. NEUTRAL (%)", fontweight="bold")
        ax2.set_title(f"Intensity gain vs. length growth\n"
                      f"({n_bases - n_drift} of {n_bases} bases within 3% word-count drift; the rest are the earlier batch)",
                      fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = f"{OUTPUT_DIR}/05_variant_manipulation_check.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()


def main():
    print("=" * 80)
    print(" " * 15 + "PHASE D QUERY SET VISUALIZATION (llm_queries)")
    print("=" * 80)

    queries_path = "dataset/eval/final_llm_queries.jsonl"
    variants_path = "dataset/eval/final_llm_queries_variants.jsonl"

    viz = LLMQueriesVisualizer(queries_path, variants_path)
    viz.plot_gender_by_article()
    viz.plot_global_gender_quotas()
    viz.plot_country_diversity()
    viz.plot_query_length_distribution()
    viz.plot_variant_manipulation_check()

    print("\n" + "=" * 80)
    print(f"Done. Saved 5 plots to {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
