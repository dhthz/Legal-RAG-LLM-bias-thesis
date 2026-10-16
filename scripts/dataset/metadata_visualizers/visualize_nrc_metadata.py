import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# NRC emotion scores of the indexed cases (sentiment_extractor.py, NRC Emotion Lexicon, Mohammad & Turney 2013):
# each emotion's share of all NRC affect tags in the case facts; intensity = the four summed, capped at 1.
# Read from the released metadata, so no dataset is needed. Summary for the corpus and by applicant gender.
METADATA_PATH = "docs/data/case_metadata.csv"
SUMMARY_PATH = "docs/metadata_reports/nrc_emotion.csv"
OUTPUT_DIR = "docs/dataset/visualizations/sentiment_analysis"
SCORES = {"nrc_fear": "Fear", "nrc_anger": "Anger", "nrc_sadness": "Sadness", "nrc_disgust": "Disgust",
          "nrc_emotional_intensity": "Intensity (sum)"}
GROUPS = ["All cases", "Male", "Female", "Multiple Applicants", "Unknown"]
GROUP_COLORS = {"All cases": "#34495e", "Male": "#3498db", "Female": "#e74c3c", "Multiple Applicants": "#9b59b6",
                "Unknown": "#95a5a6"}


def load_groups():
    with open(METADATA_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {"All cases": rows}
    for g in GROUPS[1:]:
        groups[g] = [r for r in rows if r["gender"] == g]
    return {g: {s: np.array([float(r[s]) for r in rs]) for s in SCORES} for g, rs in groups.items()}


def write_summary(groups):
    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "cases", "score", "mean", "sd", "median"])
        for g, scores in groups.items():
            for s, values in scores.items():
                writer.writerow([g, len(values), s, f"{values.mean():.4f}", f"{values.std(ddof=1):.4f}",
                                 f"{np.median(values):.4f}"])
    print(f"Saved: {SUMMARY_PATH}")


def plot(groups):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(SCORES))
    width = 0.8 / len(GROUPS)
    for i, g in enumerate(GROUPS):
        values = groups[g]
        means = [values[s].mean() for s in SCORES]
        sems = [1.96 * values[s].std(ddof=1) / np.sqrt(len(values[s])) for s in SCORES]
        ax.bar(x + (i - (len(GROUPS) - 1) / 2) * width, means, width, yerr=sems, capsize=2,
               color=GROUP_COLORS[g], edgecolor="black", linewidth=0.5,
               label=f"{g} ({len(values['nrc_fear']):,})")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORES.values())
    ax.set_ylabel("Share of the NRC emotion tags in the facts (mean, 95% CI)")
    ax.set_title("NRC emotion scores of the indexed cases, by applicant gender", fontweight="bold")
    ax.legend(title="Applicant gender (cases)", fontsize=9)
    path = f"{OUTPUT_DIR}/emotion_composition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    groups = load_groups()
    write_summary(groups)
    plot(groups)


if __name__ == "__main__":
    main()
