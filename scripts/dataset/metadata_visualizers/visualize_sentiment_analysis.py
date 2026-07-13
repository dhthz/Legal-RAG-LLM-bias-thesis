
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import scipy.stats as stats


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


class SentimentVisualizer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cases = []
        self.sentiment_data = []
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.dataset_path}...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    self.cases.append(case)

                    if 'sentiment_info' in case:
                        sentiment_info = case['sentiment_info'].copy()
                        sentiment_info['case_id'] = case['case_id']
                        sentiment_info['violated_articles'] = case.get('violated_articles', [])
                        self.sentiment_data.append(sentiment_info)
                except:
                    continue

        print(f"Loaded {len(self.sentiment_data)} cases with sentiment metadata\n")

    def plot_distributions(self):
        print("Creating distribution plots...")

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Sentiment Metadata Distributions', fontsize=16, fontweight='bold')

        metrics = [
            ('nrc_emotional_intensity', 'Emotional Intensity', '%'),
            ('nrc_fear', 'Fear', '%'),
            ('nrc_anger', 'Anger', '%'),
            ('nrc_sadness', 'Sadness', '%'),
            ('nrc_disgust', 'Disgust', '%'),
            ('passive_voice_ratio', 'Passive Voice Ratio', '%'),
            ('victim_language_count', 'Victim Language Count', 'count'),
            ('perpetrator_mentions', 'Perpetrator Mentions', 'count'),
            ('word_count', 'Word Count', 'words')
        ]

        for idx, (metric, label, unit) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [s[metric] for s in self.sentiment_data if metric in s]

            if not values:
                continue

            # Histogram with KDE
            ax.hist(values, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True)

            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            # Mean line
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

            # Formatting
            if unit == '%':
                ax.set_xlabel(f'{label} ({unit})')
            else:
                ax.set_xlabel(f'{label}')

            ax.set_ylabel('Density')
            ax.set_title(label, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = 'visualizations/sentiment_distributions.png'
        Path('visualizations').mkdir(exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_correlation_heatmap(self):
        print("Creating correlation heatmap...")

        # Extract key metrics
        metrics = [
            'nrc_emotional_intensity',
            'nrc_fear',
            'nrc_anger',
            'nrc_sadness',
            'nrc_disgust',
            'passive_voice_ratio',
            'victim_language_count',
            'perpetrator_mentions'
        ]

        # Build correlation matrix
        data_matrix = []
        for metric in metrics:
            values = [s[metric] for s in self.sentiment_data if metric in s]
            data_matrix.append(values)

        data_matrix = np.array(data_matrix).T
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Nice labels
        labels = [
            'Emotional\nIntensity',
            'Fear',
            'Anger',
            'Sadness',
            'Disgust',
            'Passive\nVoice',
            'Victim\nLanguage',
            'Perpetrator\nMentions'
        ]

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1,
                    xticklabels=labels, yticklabels=labels,
                    square=True, linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})

        plt.title('Sentiment Metrics Correlation Matrix\n(Pearson Correlation Coefficients)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = 'visualizations/sentiment_correlation_heatmap.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

        # Print significant correlations
        print("\n  📊 Significant Correlations (|r| > 0.3):")
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                r = corr_matrix[i, j]
                if abs(r) > 0.3:
                    print(f"     {metrics[i]} ↔ {metrics[j]}: r = {r:.3f}")

    def plot_article_comparison(self):
        print("Creating article comparison plots...")

        # Group by article
        article_data = defaultdict(lambda: defaultdict(list))

        for s in self.sentiment_data:
            for article in s['violated_articles']:
                article_data[article]['emotional_intensity'].append(s['nrc_emotional_intensity'])
                article_data[article]['fear'].append(s['nrc_fear'])
                article_data[article]['anger'].append(s['nrc_anger'])
                article_data[article]['passive_voice'].append(s['passive_voice_ratio'])

        # Filter articles with enough samples (min 30)
        article_data = {k: v for k, v in article_data.items()
                       if len(v['emotional_intensity']) >= 30}

        # Sort by average emotional intensity
        sorted_articles = sorted(article_data.keys(),
                                key=lambda a: np.mean(article_data[a]['emotional_intensity']),
                                reverse=True)

        # Top 10 articles
        top_articles = sorted_articles[:10]

        # Create boxplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sentiment Patterns by Violated Article (Top 10 Articles by Case Count)',
                    fontsize=16, fontweight='bold')

        metrics = [
            ('emotional_intensity', 'Emotional Intensity', axes[0, 0]),
            ('fear', 'Fear', axes[0, 1]),
            ('anger', 'Anger', axes[1, 0]),
            ('passive_voice', 'Passive Voice Ratio', axes[1, 1])
        ]

        for metric_key, metric_label, ax in metrics:
            data_for_plot = [article_data[art][metric_key] for art in top_articles]
            labels_for_plot = [f"Art. {art}\n(n={len(article_data[art][metric_key])})"
                              for art in top_articles]

            bp = ax.boxplot(data_for_plot, labels=labels_for_plot,
                           patch_artist=True, notch=True, showmeans=True)

            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_articles)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(metric_label, fontweight='bold')
            ax.set_xlabel('Article', fontweight='bold')
            ax.set_title(f'{metric_label} by Article', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            # ANOVA test
            f_stat, p_value = stats.f_oneway(*data_for_plot)
            ax.text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=9)

        plt.tight_layout()
        output_path = 'visualizations/sentiment_by_article.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

        # Statistical test summary
        print("\n  📊 ANOVA Results (testing if articles differ):")
        for metric_key, metric_label, _ in metrics:
            data_for_test = [article_data[art][metric_key] for art in top_articles]
            f_stat, p_value = stats.f_oneway(*data_for_test)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"     {metric_label:25} F={f_stat:6.2f}, p={p_value:.4f} {sig}")

    def plot_emotion_composition(self):
        print("Creating emotion composition plot...")

        emotions = ['nrc_fear', 'nrc_anger', 'nrc_sadness', 'nrc_disgust']
        emotion_labels = ['Fear', 'Anger', 'Sadness', 'Disgust']

        # Calculate means
        emotion_means = [np.mean([s[e] for s in self.sentiment_data]) for e in emotions]

        # Pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart
        colors = ['#ff9999', '#ff6666', '#9999ff', '#66ff99']
        explode = (0.1, 0, 0, 0)  # Explode fear (most prominent)

        ax1.pie(emotion_means, labels=emotion_labels, autopct='%1.1f%%',
               colors=colors, explode=explode, shadow=True, startangle=90)
        ax1.set_title('Average Emotion Composition\n(Proportion of Emotional Words)',
                     fontsize=14, fontweight='bold')

        # Bar chart with error bars
        emotion_stds = [np.std([s[e] for s in self.sentiment_data]) for e in emotions]

        bars = ax2.bar(emotion_labels, emotion_means, yerr=emotion_stds,
                      color=colors, alpha=0.7, edgecolor='black', capsize=5)

        ax2.set_ylabel('Proportion of Words', fontweight='bold')
        ax2.set_xlabel('Emotion', fontweight='bold')
        ax2.set_title('Emotion Prevalence with Standard Deviation',
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, emotion_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        output_path = 'visualizations/emotion_composition.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_bias_indicators(self):
        print("Creating bias indicator plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Potential Bias Indicators for RAG/LLM Evaluation',
                    fontsize=16, fontweight='bold')

        # 1. Emotional intensity distribution (quartiles)
        ax = axes[0, 0]
        emotional_intensity = [s['nrc_emotional_intensity'] for s in self.sentiment_data]
        q1, q2, q3 = np.percentile(emotional_intensity, [25, 50, 75])

        categories = ['Low\n(0-25%)', 'Medium-Low\n(25-50%)', 'Medium-High\n(50-75%)', 'High\n(75-100%)']
        counts = [
            sum(1 for x in emotional_intensity if x <= q1),
            sum(1 for x in emotional_intensity if q1 < x <= q2),
            sum(1 for x in emotional_intensity if q2 < x <= q3),
            sum(1 for x in emotional_intensity if x > q3)
        ]

        bars = ax.bar(categories, counts, color=['#90ee90', '#ffd700', '#ffa500', '#ff6347'],
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Cases', fontweight='bold')
        ax.set_xlabel('Emotional Intensity Quartile', fontweight='bold')
        ax.set_title('Cases by Emotional Intensity\n(Potential High-Emotion Bias)', fontweight='bold')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/len(emotional_intensity)*100:.1f}%)',
                   ha='center', va='bottom')

        # 2. Passive voice distribution
        ax = axes[0, 1]
        passive_voice = [s['passive_voice_ratio'] for s in self.sentiment_data]

        ax.scatter(range(len(passive_voice)), sorted(passive_voice),
                  alpha=0.5, s=10, color='steelblue')
        ax.axhline(np.mean(passive_voice), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(passive_voice):.2%}')
        ax.axhline(np.median(passive_voice), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(passive_voice):.2%}')

        ax.set_ylabel('Passive Voice Ratio', fontweight='bold')
        ax.set_xlabel('Cases (sorted)', fontweight='bold')
        ax.set_title('Passive Voice Distribution\n(Potential Formal/Procedural Bias)',
                    fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Victim vs Perpetrator language
        ax = axes[1, 0]
        victim_counts = [s['victim_language_count'] for s in self.sentiment_data]
        perp_counts = [s['perpetrator_mentions'] for s in self.sentiment_data]

        ax.scatter(victim_counts, perp_counts, alpha=0.5, s=20, color='purple')
        ax.plot([0, max(victim_counts)], [0, max(victim_counts)], 'r--',
               linewidth=2, label='Equal mentions line')

        ax.set_xlabel('Victim Language Count', fontweight='bold')
        ax.set_ylabel('Perpetrator Mentions', fontweight='bold')
        ax.set_title('Victim vs Perpetrator Language\n(Potential Framing Bias)',
                    fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Calculate ratio
        ratio = np.mean(victim_counts) / np.mean(perp_counts)
        ax.text(0.95, 0.05, f'Avg Victim:Perp Ratio\n{ratio:.2f}:1',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')

        # 4. Joint distribution: Emotion vs Passive Voice
        ax = axes[1, 1]

        hexbin = ax.hexbin(emotional_intensity, passive_voice,
                          gridsize=30, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel('Emotional Intensity', fontweight='bold')
        ax.set_ylabel('Passive Voice Ratio', fontweight='bold')
        ax.set_title('Emotional Intensity vs Passive Voice\n(Correlation Test)',
                    fontweight='bold')

        # Correlation
        r, p = stats.pearsonr(emotional_intensity, passive_voice)
        ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np = {p:.4f}',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')

        plt.colorbar(hexbin, ax=ax, label='Case Count')

        plt.tight_layout()
        output_path = 'visualizations/bias_indicators.png'
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def generate_statistical_report(self):
        print("\nGenerating statistical analysis report...")

        report = []
        report.append("=" * 80)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)

        # 1. Normality tests (Shapiro-Wilk)
        report.append("\n1. NORMALITY TESTS (Shapiro-Wilk)")
        report.append("-" * 80)
        metrics = ['nrc_emotional_intensity', 'nrc_fear', 'passive_voice_ratio']
        for metric in metrics:
            values = [s[metric] for s in self.sentiment_data]
            stat, p = stats.shapiro(values[:5000])  # Limit to 5000 for computational efficiency
            dist = "Normal" if p > 0.05 else "Non-normal"
            report.append(f"  {metric:30} W={stat:.4f}, p={p:.4f} → {dist}")

        # 2. T-tests: High vs Low emotion cases
        report.append("\n2. T-TESTS: High vs Low Emotional Intensity Cases")
        report.append("-" * 80)
        emotional_intensity = [s['nrc_emotional_intensity'] for s in self.sentiment_data]
        median_emotion = np.median(emotional_intensity)

        high_emotion_passive = [s['passive_voice_ratio'] for s in self.sentiment_data
                               if s['nrc_emotional_intensity'] > median_emotion]
        low_emotion_passive = [s['passive_voice_ratio'] for s in self.sentiment_data
                              if s['nrc_emotional_intensity'] <= median_emotion]

        t_stat, p_value = stats.ttest_ind(high_emotion_passive, low_emotion_passive)
        report.append(f"  Passive voice: High vs Low emotion")
        report.append(f"    High emotion mean: {np.mean(high_emotion_passive):.2%}")
        report.append(f"    Low emotion mean:  {np.mean(low_emotion_passive):.2%}")
        report.append(f"    t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.001:
            report.append(f"    → HIGHLY SIGNIFICANT difference (p < 0.001)")
        elif p_value < 0.05:
            report.append(f"    → Significant difference (p < 0.05)")
        else:
            report.append(f"    → No significant difference")

        # 3. Effect sizes (Cohen's d)
        report.append("\n3. EFFECT SIZES (Cohen's d)")
        report.append("-" * 80)
        mean_high = np.mean(high_emotion_passive)
        mean_low = np.mean(low_emotion_passive)
        std_pooled = np.sqrt((np.var(high_emotion_passive) + np.var(low_emotion_passive)) / 2)
        cohens_d = (mean_high - mean_low) / std_pooled
        report.append(f"  Passive voice difference (High vs Low emotion):")
        report.append(f"    Cohen's d = {cohens_d:.3f}")
        if abs(cohens_d) < 0.2:
            report.append(f"    → Small effect")
        elif abs(cohens_d) < 0.5:
            report.append(f"    → Medium effect")
        else:
            report.append(f"    → Large effect")

        # Save report
        report_text = "\n".join(report)
        output_path = 'visualizations/statistical_analysis_report.txt'
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"  ✓ Saved: {output_path}")

        # Print to console
        print("\n" + report_text)


def main():

    print("=" * 80)
    print(" " * 20 + "SENTIMENT METADATA VISUALIZATION")
    print("=" * 80)
    print("\nTechniques used:")
    print("  • Distribution analysis (histograms + KDE)")
    print("  • Correlation analysis (Pearson correlation)")
    print("  • Article comparison (ANOVA)")
    print("  • Statistical tests (t-tests, normality tests)")
    print("  • Bias indicators (quartiles, ratios)")
    print("=" * 80 + "\n")

    dataset_path = "dataset/train_with_metadata.jsonl"

    visualizer = SentimentVisualizer(dataset_path)

    # Generate all visualizations
    visualizer.plot_distributions()
    visualizer.plot_correlation_heatmap()
    visualizer.plot_article_comparison()
    visualizer.plot_emotion_composition()
    visualizer.plot_bias_indicators()
    visualizer.generate_statistical_report()

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files in visualizations/:")
    print("  1. sentiment_distributions.png - Distribution of all metrics")
    print("  2. sentiment_correlation_heatmap.png - Correlation matrix")
    print("  3. sentiment_by_article.png - Article-specific patterns (ANOVA)")
    print("  4. emotion_composition.png - Overall emotion breakdown")
    print("  5. bias_indicators.png - Potential bias sources for RAG")
    print("  6. statistical_analysis_report.txt - Statistical test results")
    print("\nKey findings for thesis:")
    print("  → Use correlation heatmap to explain metric relationships")
    print("  → Use article comparison to show emotional variance by violation type")
    print("  → Use bias indicators to justify need for bias testing in RAG")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()