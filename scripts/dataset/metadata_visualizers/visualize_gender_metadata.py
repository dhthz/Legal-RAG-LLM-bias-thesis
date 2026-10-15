import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

DATASET_PATH = "dataset/train_with_metadata.jsonl"
# The labels before the 209 "Needs Manual Classification" cases were labelled by hand (kept locally)
BEFORE_PATH = "dataset/train_with_metadata.jsonl.pre_gender_corrections.bak"
VALIDATION_PATH = "docs/bias_audit/results/tables/10_gender_label_validation.csv"
COUNTS_PATH = "docs/metadata_reports/gender_labels.csv"
OUTPUT_DIR = "visualizations/gender_analysis"
LABELS = ["Male", "Female", "Multiple Applicants", "Unknown", "Needs Manual Classification"]
GENDER_COLORS = {"Male": "#3498db", "Female": "#e74c3c", "Multiple Applicants": "#9b59b6", "Unknown": "#95a5a6"}


class GenderVisualizer:

    def __init__(self, dataset_path):
        self.cases = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                case = json.loads(line)
                self.cases.append({
                    'gender': case['classification']['gender'],
                    'confidence': case['classification'].get('confidence'),
                    'violated_articles': case.get('violated_articles', []),
                    'judgment_date': case.get('judgment_date')
                })
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Loaded {len(self.cases)} cases from {dataset_path}")

    def save(self, fig, name):
        path = f"{OUTPUT_DIR}/{name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    # Final label counts, the counts before the hand labelling, and how many were labelled by hand
    def write_counts(self):
        final = Counter(c['gender'] for c in self.cases)
        hand = Counter(c['gender'] for c in self.cases if c['confidence'] == 'human')
        before = None
        if os.path.exists(BEFORE_PATH):
            with open(BEFORE_PATH, 'r', encoding='utf-8') as f:
                before = Counter(json.loads(line)['classification']['gender'] for line in f)
        total = sum(final.values())
        with open(COUNTS_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["label", "before_hand_labelling", "labelled_by_hand", "final", "final_share"])
            for label in LABELS:
                writer.writerow([label, before[label] if before else "", hand[label], final[label],
                                 round(final[label] / total, 4)])
            writer.writerow(["total", sum(before.values()) if before else "", sum(hand.values()), total, 1.0])
        print(f"Saved: {COUNTS_PATH}")
        return final, before

    def plot_distribution(self, final, before):
        labels = [l for l in LABELS if final[l] or (before and before[l])]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 6))
        if before:
            ax.bar(x - 0.2, [before[l] for l in labels], 0.4, color="#d5d8dc", edgecolor='black',
                   label="AI labels (before hand labelling)")
        bars = ax.bar(x + (0.2 if before else 0), [final[l] for l in labels], 0.4 if before else 0.6,
                      color=[GENDER_COLORS.get(l, "#95a5a6") for l in labels], edgecolor='black', label="Final labels")
        total = sum(final.values())
        for bar, label in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{final[label]}\n({final[label] / total:.1%})",
                    ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel("Number of cases")
        ax.set_title(f"Applicant gender in the {total:,} indexed judgments", fontweight='bold')
        ax.legend()
        self.save(fig, "gender_distribution.png")

    # Accuracy of the AI labels against the hand check (300 cases), with 95% Wilson intervals
    def plot_validation(self):
        with open(VALIDATION_PATH, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        names = [r['ai_label'].replace(" (reweighted to corpus shares)", "\n(reweighted)") for r in rows]
        acc = np.array([float(r['accuracy']) for r in rows]) * 100
        lo = acc - np.array([float(r['ci95_lo']) for r in rows]) * 100
        hi = np.array([float(r['ci95_hi']) for r in rows]) * 100 - acc
        colors = [GENDER_COLORS.get(r['ai_label'], "#34495e") for r in rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, acc, yerr=[lo, hi], capsize=6, color=colors, edgecolor='black', alpha=0.85)
        for i, (r, a) in enumerate(zip(rows, acc)):
            label = f"{a:.1f}%" + (f"\n{r['correct']}/{r['n_checked']}" if r['correct'] else "")
            ax.text(i, 2, label, ha='center', va='bottom', color='white', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Labels confirmed by hand (%)")
        ax.set_title("Gender labels checked by hand (100 per label)", fontweight='bold')
        self.save(fig, "gender_label_validation.png")

    def plot_by_article(self):
        article_gender = defaultdict(Counter)
        for case in self.cases:
            for article in case['violated_articles']:
                article_gender[article][case['gender']] += 1
        articles = sorted((a for a in article_gender if sum(article_gender[a].values()) >= 50),
                          key=lambda a: sum(article_gender[a].values()), reverse=True)[:12]
        genders = list(GENDER_COLORS)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle("Applicant gender by violated article (top 12 articles)", fontsize=16, fontweight='bold')
        bottom_counts, bottom_pct = np.zeros(len(articles)), np.zeros(len(articles))
        for gender in genders:
            counts = np.array([article_gender[a][gender] for a in articles])
            pct = np.array([article_gender[a][gender] / sum(article_gender[a].values()) * 100 for a in articles])
            ax1.bar(articles, counts, bottom=bottom_counts, label=gender, color=GENDER_COLORS[gender], edgecolor='black')
            ax2.bar(articles, pct, bottom=bottom_pct, label=gender, color=GENDER_COLORS[gender], edgecolor='black')
            bottom_counts += counts
            bottom_pct += pct
        ax1.set_ylabel("Number of cases")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 100)
        for ax in (ax1, ax2):
            ax.set_xlabel("Article")
            ax.legend(title="Gender")
        plt.tight_layout()
        self.save(fig, "gender_by_article.png")

    def plot_temporal_trends(self):
        yearly = defaultdict(Counter)
        for case in self.cases:
            if case['judgment_date']:
                yearly[int(case['judgment_date'][:4])][case['gender']] += 1
        years = sorted(yearly)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle("Applicant gender over time", fontsize=16, fontweight='bold')
        for gender in ["Male", "Female", "Multiple Applicants"]:
            counts = [yearly[y][gender] for y in years]
            shares = [yearly[y][gender] / sum(yearly[y].values()) * 100 for y in years]
            ax1.plot(years, counts, marker='o', markersize=4, linewidth=2, label=gender, color=GENDER_COLORS[gender])
            ax2.plot(years, shares, marker='o', markersize=4, linewidth=2, label=gender, color=GENDER_COLORS[gender])
        ax1.set_ylabel("Number of cases")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 100)
        for ax in (ax1, ax2):
            ax.set_xlabel("Judgment year")
            ax.legend()
        plt.tight_layout()
        self.save(fig, "gender_temporal_trends.png")


def main():
    visualizer = GenderVisualizer(DATASET_PATH)
    final, before = visualizer.write_counts()
    visualizer.plot_distribution(final, before)
    visualizer.plot_validation()
    visualizer.plot_by_article()
    visualizer.plot_temporal_trends()


if __name__ == "__main__":
    main()
