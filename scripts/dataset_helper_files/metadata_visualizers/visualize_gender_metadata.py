import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import scipy.stats as stats
from datetime import datetime


# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class GenderVisualizer:
    """Creates visualizations for gender metadata analysis"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cases = []
        self.load_data()

    def load_data(self):
        """Load gender classification data from dataset"""
        print(f"Loading data from {self.dataset_path}...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    if 'classification' in case:
                        self.cases.append({
                            'case_id': case['case_id'],
                            'gender': case['classification'].get('gender', 'Unknown'),
                            'confidence': case['classification'].get('confidence', 'Unknown'),
                            'violated_articles': case.get('violated_articles', []),
                            'judgment_date': case.get('judgment_date', None)
                        })
                except:
                    continue

        print(f"Loaded {len(self.cases)} cases with gender classification\n")

    def plot_gender_distribution(self):
        """Plot overall gender distribution"""
        print("Creating gender distribution plot...")

        gender_counts = Counter(c['gender'] for c in self.cases)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Gender Distribution in ECHR Cases', fontsize=16, fontweight='bold')

        # Pie chart
        colors = ['#3498db', '#e74c3c', '#95a5a6', '#9b59b6']
        labels = list(gender_counts.keys())
        sizes = list(gender_counts.values())
        percentages = [f'{label}\n{size} ({size/sum(sizes)*100:.1f}%)'
                      for label, size in zip(labels, sizes)]

        explode = [0.05] * len(labels)  # Slight separation
        ax1.pie(sizes, labels=percentages, colors=colors, explode=explode,
               shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Gender Distribution (Proportions)', fontsize=14, fontweight='bold')

        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Cases', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Gender Classification', fontweight='bold', fontsize=12)
        ax2.set_title('Gender Distribution (Counts)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/sum(sizes)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        output_path = 'visualizations/gender_distribution.png'
        Path('visualizations').mkdir(exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_confidence_analysis(self):
        """Analyze confidence levels by gender"""
        print("Creating confidence analysis plot...")

        # Group by gender and confidence
        gender_confidence = defaultdict(lambda: Counter())

        for case in self.cases:
            gender = case['gender']
            confidence = case['confidence']
            gender_confidence[gender][confidence] += 1

        # Prepare data
        genders = ['Male', 'Female', 'Multiple Applicants', 'Unknown']
        confidences = ['High', 'Medium', 'Low', 'N/A']
        colors_conf = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c', 'N/A': '#95a5a6'}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Gender Classification Confidence Analysis', fontsize=16, fontweight='bold')

        # Stacked bar chart
        bottom = np.zeros(len(genders))
        for conf in confidences:
            counts = [gender_confidence[g][conf] for g in genders]
            ax1.bar(genders, counts, bottom=bottom, label=conf,
                   color=colors_conf[conf], alpha=0.8, edgecolor='black')
            bottom += counts

        ax1.set_ylabel('Number of Cases', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Gender', fontweight='bold', fontsize=12)
        ax1.set_title('Confidence Levels by Gender (Stacked)', fontsize=14, fontweight='bold')
        ax1.legend(title='Confidence', fontsize=10, title_fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=15)

        # Percentage stacked bar chart
        for gender in genders:
            total = sum(gender_confidence[gender].values())
            if total > 0:
                for conf in confidences:
                    gender_confidence[gender][conf] = (gender_confidence[gender][conf] / total) * 100

        bottom = np.zeros(len(genders))
        for conf in confidences:
            percentages = [gender_confidence[g][conf] for g in genders]
            ax2.bar(genders, percentages, bottom=bottom, label=conf,
                   color=colors_conf[conf], alpha=0.8, edgecolor='black')
            bottom += percentages

        ax2.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Gender', fontweight='bold', fontsize=12)
        ax2.set_title('Confidence Levels by Gender (Normalized)', fontsize=14, fontweight='bold')
        ax2.legend(title='Confidence', fontsize=10, title_fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=15)
        ax2.set_ylim([0, 100])

        plt.tight_layout()
        output_path = 'visualizations/gender_confidence_analysis.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_gender_by_article(self):
        """Analyze gender distribution by violated article using Chi-square tests"""
        print("Creating gender by article analysis...")

        # Group by article
        article_gender = defaultdict(lambda: Counter())

        for case in self.cases:
            for article in case['violated_articles']:
                article_gender[article][case['gender']] += 1

        # Filter articles with enough samples
        article_gender = {k: v for k, v in article_gender.items()
                         if sum(v.values()) >= 50}

        # Sort by total cases
        sorted_articles = sorted(article_gender.keys(),
                                key=lambda a: sum(article_gender[a].values()),
                                reverse=True)[:12]  # Top 12 articles

        # Prepare data for plotting
        genders = ['Male', 'Female', 'Multiple Applicants', 'Unknown']
        colors = ['#3498db', '#e74c3c', '#9b59b6', '#95a5a6']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Gender Distribution by Violated Article (Top 12 Articles)',
                    fontsize=16, fontweight='bold')

        # Stacked bar chart
        bottom = np.zeros(len(sorted_articles))
        for gender, color in zip(genders, colors):
            counts = [article_gender[art][gender] for art in sorted_articles]
            ax1.bar(sorted_articles, counts, bottom=bottom, label=gender,
                   color=color, alpha=0.8, edgecolor='black')
            bottom += counts

        ax1.set_ylabel('Number of Cases', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Article', fontweight='bold', fontsize=12)
        ax1.set_title('Gender Distribution by Article (Counts)', fontsize=14, fontweight='bold')
        ax1.legend(title='Gender', fontsize=10, title_fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Percentage stacked bar
        article_gender_pct = {}
        for article in sorted_articles:
            total = sum(article_gender[article].values())
            article_gender_pct[article] = {g: (article_gender[article][g] / total) * 100
                                          for g in genders}

        bottom = np.zeros(len(sorted_articles))
        for gender, color in zip(genders, colors):
            percentages = [article_gender_pct[art][gender] for art in sorted_articles]
            ax2.bar(sorted_articles, percentages, bottom=bottom, label=gender,
                   color=color, alpha=0.8, edgecolor='black')
            bottom += percentages

        ax2.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Article', fontweight='bold', fontsize=12)
        ax2.set_title('Gender Distribution by Article (Proportions)', fontsize=14, fontweight='bold')
        ax2.legend(title='Gender', fontsize=10, title_fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim([0, 100])

        plt.tight_layout()
        output_path = 'visualizations/gender_by_article.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

        # Chi-square test
        print("\n  📊 Chi-square Tests (testing if gender distribution differs by article):")
        # Compare each article to overall distribution
        overall_gender = Counter(c['gender'] for c in self.cases)
        overall_male_pct = overall_gender['Male'] / sum(overall_gender.values())

        for article in sorted_articles[:5]:  # Top 5
            observed_male = article_gender[article]['Male']
            total = sum(article_gender[article].values())
            expected_male = overall_male_pct * total

            # Chi-square for male proportion
            chi2 = ((observed_male - expected_male) ** 2) / expected_male
            # Simple approximation (should use proper contingency table)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"     Article {article:10} Male: {observed_male/total:.1%} vs {overall_male_pct:.1%} baseline, χ²={chi2:.2f}, p={p_value:.4f} {sig}")

    def plot_temporal_trends(self):
        """Analyze gender representation over time"""
        print("Creating temporal trend analysis...")

        # Parse dates and group by year
        yearly_gender = defaultdict(lambda: Counter())

        for case in self.cases:
            if case['judgment_date']:
                try:
                    year = int(case['judgment_date'][:4])
                    if 1960 <= year <= 2030:  # Reasonable range
                        yearly_gender[year][case['gender']] += 1
                except:
                    continue

        # Sort years
        years = sorted(yearly_gender.keys())

        # Prepare data
        genders = ['Male', 'Female', 'Multiple Applicants']
        colors = ['#3498db', '#e74c3c', '#9b59b6']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Temporal Trends in Gender Representation (ECHR Cases)',
                    fontsize=16, fontweight='bold')

        # Absolute counts
        for gender, color in zip(genders, colors):
            counts = [yearly_gender[y][gender] for y in years]
            ax1.plot(years, counts, marker='o', linewidth=2, label=gender,
                    color=color, markersize=4)

        ax1.set_ylabel('Number of Cases', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Judgment Year', fontweight='bold', fontsize=12)
        ax1.set_title('Gender Representation Over Time (Counts)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # Proportions
        for gender, color in zip(genders, colors):
            proportions = []
            for y in years:
                total = sum(yearly_gender[y].values())
                prop = (yearly_gender[y][gender] / total * 100) if total > 0 else 0
                proportions.append(prop)
            ax2.plot(years, proportions, marker='o', linewidth=2, label=gender,
                    color=color, markersize=4)

        ax2.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Judgment Year', fontweight='bold', fontsize=12)
        ax2.set_title('Gender Representation Over Time (Proportions)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 100])

        plt.tight_layout()
        output_path = 'visualizations/gender_temporal_trends.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_bias_indicators(self):
        """Plot potential bias indicators for RAG evaluation"""
        print("Creating bias indicator plots...")

        gender_counts = Counter(c['gender'] for c in self.cases)
        total = sum(gender_counts.values())

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Bias Indicators for RAG/LLM Evaluation',
                    fontsize=16, fontweight='bold')

        # 1. Gender imbalance visualization
        genders = ['Male', 'Female', 'Multiple\nApplicants', 'Unknown']
        counts = [gender_counts['Male'], gender_counts['Female'],
                 gender_counts['Multiple Applicants'], gender_counts['Unknown']]
        colors = ['#3498db', '#e74c3c', '#9b59b6', '#95a5a6']

        baseline = total / 4  # Perfect balance
        bars = ax1.bar(genders, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.axhline(baseline, color='red', linestyle='--', linewidth=2,
                   label=f'Perfect Balance ({baseline:.0f})')

        ax1.set_ylabel('Number of Cases', fontweight='bold')
        ax1.set_xlabel('Gender', fontweight='bold')
        ax1.set_title('Gender Imbalance\n(Potential Representation Bias)', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add percentages
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count/total*100:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

        # 2. Male vs Female ratio by confidence
        male_conf = Counter(c['confidence'] for c in self.cases if c['gender'] == 'Male')
        female_conf = Counter(c['confidence'] for c in self.cases if c['gender'] == 'Female')

        confidences = ['High', 'Medium', 'Low']
        male_counts = [male_conf[c] for c in confidences]
        female_counts = [female_conf[c] for c in confidences]

        x = np.arange(len(confidences))
        width = 0.35

        ax2.bar(x - width/2, male_counts, width, label='Male', color='#3498db', alpha=0.8)
        ax2.bar(x + width/2, female_counts, width, label='Female', color='#e74c3c', alpha=0.8)

        ax2.set_ylabel('Number of Cases', fontweight='bold')
        ax2.set_xlabel('Confidence Level', fontweight='bold')
        ax2.set_title('Male vs Female by Confidence\n(Potential Classification Bias)',
                     fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(confidences)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Article-specific gender skew
        # Identify most male-skewed and female-skewed articles
        article_male_ratio = {}
        for article, gender_dist in article_gender.items() if 'article_gender' in dir() else {}:
            total_art = sum(gender_dist.values())
            if total_art >= 50:
                male_pct = gender_dist['Male'] / total_art
                article_male_ratio[article] = male_pct

        if article_male_ratio:
            sorted_by_male = sorted(article_male_ratio.items(), key=lambda x: x[1])
            top_female = sorted_by_male[:5]  # Most female-skewed
            top_male = sorted_by_male[-5:]   # Most male-skewed

            articles_plot = [a[0] for a in top_female] + [a[0] for a in top_male]
            ratios_plot = [a[1]*100 for a in top_female] + [a[1]*100 for a in top_male]
            colors_plot = ['#e74c3c']*5 + ['#3498db']*5

            ax3.barh(articles_plot, ratios_plot, color=colors_plot, alpha=0.7, edgecolor='black')
            ax3.axvline(50, color='black', linestyle='--', linewidth=2, label='Gender Parity')
            ax3.set_xlabel('Male Percentage (%)', fontweight='bold')
            ax3.set_ylabel('Article', fontweight='bold')
            ax3.set_title('Articles with Strongest Gender Skew\n(Potential Article-Gender Bias)',
                         fontweight='bold')
            ax3.legend()
            ax3.grid(axis='x', alpha=0.3)

        # 4. Confidence distribution pie
        conf_counts = Counter(c['confidence'] for c in self.cases)
        labels_conf = list(conf_counts.keys())
        sizes_conf = list(conf_counts.values())
        colors_conf_pie = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c', 'N/A': '#95a5a6'}
        colors_pie = [colors_conf_pie[l] for l in labels_conf]

        ax4.pie(sizes_conf, labels=labels_conf, autopct='%1.1f%%', colors=colors_pie,
               shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax4.set_title('Overall Classification Confidence\n(Potential Uncertainty Indicator)',
                     fontweight='bold')

        plt.tight_layout()
        output_path = 'visualizations/gender_bias_indicators.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def generate_statistical_report(self):
        """Generate statistical test results"""
        print("\nGenerating statistical analysis report...")

        report = []
        report.append("=" * 80)
        report.append("GENDER METADATA - STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)

        # 1. Overall distribution
        gender_counts = Counter(c['gender'] for c in self.cases)
        total = sum(gender_counts.values())

        report.append("\n1. OVERALL GENDER DISTRIBUTION")
        report.append("-" * 80)
        for gender, count in gender_counts.most_common():
            pct = (count / total) * 100
            report.append(f"  {gender:20} {count:5} ({pct:5.2f}%)")

        # 2. Chi-square goodness of fit (test against uniform distribution)
        observed = list(gender_counts.values())
        expected = [total / len(observed)] * len(observed)
        chi2, p = stats.chisquare(observed, expected)

        report.append("\n2. CHI-SQUARE TEST: Gender Balance")
        report.append("-" * 80)
        report.append(f"  H0: Gender is uniformly distributed")
        report.append(f"  χ² = {chi2:.3f}, p = {p:.4f}")
        if p < 0.001:
            report.append(f"  → HIGHLY SIGNIFICANT imbalance (p < 0.001)")
        else:
            report.append(f"  → Not significant")

        # 3. Male to Female ratio
        male = gender_counts['Male']
        female = gender_counts['Female']
        ratio = male / female if female > 0 else float('inf')

        report.append("\n3. MALE TO FEMALE RATIO")
        report.append("-" * 80)
        report.append(f"  Male cases:   {male}")
        report.append(f"  Female cases: {female}")
        report.append(f"  Ratio:        {ratio:.2f}:1")
        if ratio > 1.5:
            report.append(f"  → MALE-SKEWED dataset (ratio > 1.5)")
        elif ratio < 0.67:
            report.append(f"  → FEMALE-SKEWED dataset (ratio < 0.67)")
        else:
            report.append(f"  → Relatively balanced")

        # Save report
        report_text = "\n".join(report)
        output_path = 'visualizations/gender_statistical_report.txt'
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"  ✓ Saved: {output_path}")

        # Print to console
        print("\n" + report_text)


def main():
    """Run complete gender visualization pipeline"""

    print("=" * 80)
    print(" " * 20 + "GENDER METADATA VISUALIZATION")
    print("=" * 80)
    print("\nTechniques used:")
    print("  • Distribution analysis (pie charts, bar charts)")
    print("  • Confidence level analysis (stacked bars)")
    print("  • Article-specific analysis (chi-square tests)")
    print("  • Temporal trend analysis (time series)")
    print("  • Bias indicators (imbalance metrics)")
    print("=" * 80 + "\n")

    dataset_path = "dataset/train_with_metadata.jsonl"

    visualizer = GenderVisualizer(dataset_path)

    # Generate all visualizations
    visualizer.plot_gender_distribution()
    visualizer.plot_confidence_analysis()
    visualizer.plot_gender_by_article()
    visualizer.plot_temporal_trends()
    visualizer.plot_bias_indicators()
    visualizer.generate_statistical_report()

    print("\n" + "=" * 80)
    print("✅ GENDER VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files in visualizations/:")
    print("  1. gender_distribution.png - Overall gender breakdown")
    print("  2. gender_confidence_analysis.png - Confidence levels by gender")
    print("  3. gender_by_article.png - Article-specific patterns (chi-square)")
    print("  4. gender_temporal_trends.png - Changes over time")
    print("  5. gender_bias_indicators.png - Potential bias sources for RAG")
    print("  6. gender_statistical_report.txt - Statistical test results")
    print("\nKey findings for thesis:")
    print("  → Use gender distribution to show dataset composition")
    print("  → Use article analysis to identify gender-article associations")
    print("  → Use bias indicators to justify gender-aware RAG testing")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Store article_gender globally for bias indicators
    global article_gender
    article_gender = {}

    main()
