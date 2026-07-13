#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import itertools

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATASET_PATH = "dataset/train.jsonl"
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    print("Loading data from train.jsonl...")

    data = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            case = json.loads(line)
            data.append(case)

    print(f"✅ Loaded {len(data)} cases")
    return data

def plot_article_distribution(data):
    print("\n1. Generating violated articles distribution...")

    # Count violated articles
    article_counts = Counter()
    for case in data:
        violated_articles = case.get('violated_articles', [])
        for article in violated_articles:
            article_counts[article] += 1

    # Get top 20 articles
    top_articles = article_counts.most_common(20)
    articles = [art for art, _ in top_articles]
    counts = [count for _, count in top_articles]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top 20 violated articles bar chart
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(articles)))
    bars = ax1.barh(range(len(articles)), counts, color=colors, edgecolor='black', linewidth=1.5)

    ax1.set_yticks(range(len(articles)))
    ax1.set_yticklabels(articles)
    ax1.set_xlabel('Number of Cases', fontsize=12)
    ax1.set_title('Top 20 Most Violated Articles', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(count + 20, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=9)

    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    sorted_counts = sorted(article_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    cumsum_pct = (cumsum / cumsum[-1]) * 100

    ax2.plot(range(1, len(cumsum)+1), cumsum_pct, 'b-', linewidth=2)
    ax2.axhline(y=80, color='r', linestyle='--', linewidth=2, label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% threshold')

    # Find how many articles account for 80% and 90%
    idx_80 = np.where(cumsum_pct >= 80)[0][0] + 1
    idx_90 = np.where(cumsum_pct >= 90)[0][0] + 1

    ax2.axvline(x=idx_80, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=idx_90, color='orange', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Number of Articles (ranked)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title(f'Cumulative Article Distribution\n({idx_80} articles = 80%, {idx_90} articles = 90%)',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Pie chart for top 10
    ax3 = axes[1, 0]
    top_10 = article_counts.most_common(10)
    top_10_labels = [art for art, _ in top_10]
    top_10_counts = [count for _, count in top_10]
    other_count = sum(article_counts.values()) - sum(top_10_counts)

    if other_count > 0:
        top_10_labels.append('Others')
        top_10_counts.append(other_count)

    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_10_labels)))
    wedges, texts, autotexts = ax3.pie(top_10_counts, labels=top_10_labels, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    ax3.set_title('Top 10 Violated Articles (Proportion)', fontsize=14, fontweight='bold')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_violations = sum(article_counts.values())
    unique_articles = len(article_counts)
    avg_violations_per_article = total_violations / unique_articles

    summary_text = f"""
    VIOLATED ARTICLES STATISTICS
    {'─' * 50}

    Total cases:                    {len(data):,}
    Total violation instances:      {total_violations:,}
    Unique articles violated:       {unique_articles}

    Avg violations per article:     {avg_violations_per_article:.1f}
    Avg articles per case:          {total_violations/len(data):.2f}

    Top 5 Most Violated Articles:
    """

    for i, (art, count) in enumerate(article_counts.most_common(5), 1):
        pct = (count / len(data)) * 100
        summary_text += f"\n      {i}. {art:15} {count:5,} cases ({pct:5.1f}%)"

    summary_text += f"""

    Coverage Analysis:
      • Top 10 articles cover:      {sum(c for _, c in article_counts.most_common(10))/total_violations*100:.1f}%
      • Top 20 articles cover:      {sum(c for _, c in article_counts.most_common(20))/total_violations*100:.1f}%
      • {idx_80} articles needed for 80% coverage
      • {idx_90} articles needed for 90% coverage
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "article_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_temporal_trends(data):
    print("\n2. Generating temporal trends...")

    # Extract judgment dates
    years = []
    for case in data:
        judgment_date = case.get('judgment_date')
        if judgment_date:
            try:
                year = int(judgment_date.split('-')[0])
                if 1950 <= year <= 2030:  # Sanity check
                    years.append(year)
            except:
                continue

    year_counts = Counter(years)
    sorted_years = sorted(year_counts.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cases per year
    ax1 = axes[0, 0]
    counts_by_year = [year_counts[year] for year in sorted_years]

    ax1.bar(sorted_years, counts_by_year, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Judgment Year', fontsize=12)
    ax1.set_ylabel('Number of Cases', fontsize=12)
    ax1.set_title('Cases by Judgment Year', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add trend line
    z = np.polyfit(sorted_years, counts_by_year, 2)
    p = np.poly1d(z)
    ax1.plot(sorted_years, p(sorted_years), "r--", alpha=0.8, linewidth=2, label='Trend (polynomial)')
    ax1.legend()

    # 2. Cumulative cases over time
    ax2 = axes[0, 1]
    cumsum = np.cumsum(counts_by_year)

    ax2.plot(sorted_years, cumsum, 'b-', linewidth=2, marker='o', markersize=3)
    ax2.fill_between(sorted_years, cumsum, alpha=0.3)
    ax2.set_xlabel('Judgment Year', fontsize=12)
    ax2.set_ylabel('Cumulative Cases', fontsize=12)
    ax2.set_title(f'Cumulative Cases Over Time (Total: {len(years):,})',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Decade breakdown
    ax3 = axes[1, 0]

    def get_decade(year):
        return (year // 10) * 10

    decade_counts = Counter([get_decade(year) for year in years])
    decades = sorted(decade_counts.keys())
    decade_labels = [f'{d}s' for d in decades]
    decade_values = [decade_counts[d] for d in decades]

    colors_decade = plt.cm.viridis(np.linspace(0.2, 0.9, len(decades)))
    bars = ax3.bar(decade_labels, decade_values, color=colors_decade,
                   edgecolor='black', linewidth=1.5)

    ax3.set_xlabel('Decade', fontsize=12)
    ax3.set_ylabel('Number of Cases', fontsize=12)
    ax3.set_title('Cases by Decade', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, decade_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontsize=10)

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    years_array = np.array(years)
    summary_text = f"""
    TEMPORAL DISTRIBUTION STATISTICS
    {'─' * 50}

    Total cases with dates:         {len(years):,}
    Date range:                     {min(years)} - {max(years)}
    Span:                           {max(years) - min(years)} years

    Cases per year:
      Mean:                         {np.mean(counts_by_year):.1f}
      Median:                       {np.median(counts_by_year):.1f}
      Std Dev:                      {np.std(counts_by_year):.1f}
      Max:                          {max(counts_by_year)} ({sorted_years[counts_by_year.index(max(counts_by_year))]})
      Min:                          {min(counts_by_year)} ({sorted_years[counts_by_year.index(min(counts_by_year))]})

    Decade with most cases:         {max(decade_counts, key=decade_counts.get)}s ({max(decade_counts.values())} cases)

    Quartiles:
      Q1 (25th percentile):         {int(np.percentile(years_array, 25))}
      Q2 (50th percentile):         {int(np.percentile(years_array, 50))}
      Q3 (75th percentile):         {int(np.percentile(years_array, 75))}

    Data Quality:
      • {len(years)/len(data)*100:.1f}% of cases have judgment dates
      • Coverage spans {len(sorted_years)} different years
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "temporal_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_defendant_countries(data):
    print("\n3. Generating defendant countries distribution...")

    # Count defendants
    defendant_counts = Counter()
    for case in data:
        defendants = case.get('defendants', [])
        for defendant in defendants:
            defendant_counts[defendant] += 1

    # Get top 20 countries
    top_20 = defendant_counts.most_common(20)
    countries = [c for c, _ in top_20]
    counts = [cnt for _, cnt in top_20]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top 20 defendant countries
    ax1 = axes[0, 0]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(countries)))
    bars = ax1.barh(range(len(countries)), counts, color=colors,
                    edgecolor='black', linewidth=1.5)

    ax1.set_yticks(range(len(countries)))
    ax1.set_yticklabels(countries)
    ax1.set_xlabel('Number of Cases', fontsize=12)
    ax1.set_title('Top 20 Defendant Countries', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = (count / len(data)) * 100
        ax1.text(count + 10, bar.get_y() + bar.get_height()/2,
                f'{count:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)

    # 2. Geographic distribution pie chart
    ax2 = axes[0, 1]

    # Group by region (simplified)
    def get_region(country):
        eastern_europe = ['RUSSIA', 'UKRAINE', 'POLAND', 'ROMANIA', 'BULGARIA',
                         'CZECH REPUBLIC', 'SLOVAKIA', 'HUNGARY', 'CROATIA', 'SERBIA',
                         'LITHUANIA', 'LATVIA', 'ESTONIA', 'MOLDOVA', 'ALBANIA']
        western_europe = ['UNITED KINGDOM', 'FRANCE', 'GERMANY', 'ITALY', 'SPAIN',
                         'NETHERLANDS', 'BELGIUM', 'SWITZERLAND', 'AUSTRIA', 'PORTUGAL',
                         'GREECE', 'SWEDEN', 'NORWAY', 'DENMARK', 'FINLAND', 'IRELAND']

        if country in eastern_europe:
            return 'Eastern Europe'
        elif country in western_europe:
            return 'Western Europe'
        else:
            return 'Other'

    region_counts = Counter()
    for case in data:
        defendants = case.get('defendants', [])
        for defendant in defendants:
            region = get_region(defendant)
            region_counts[region] += 1

    regions = list(region_counts.keys())
    region_values = list(region_counts.values())
    colors_region = ['#3498db', '#e74c3c', '#95a5a6']

    wedges, texts, autotexts = ax2.pie(region_values, labels=regions, autopct='%1.1f%%',
                                         colors=colors_region[:len(regions)], startangle=90,
                                         textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('Regional Distribution', fontsize=14, fontweight='bold')

    # 3. Pareto chart (80/20 rule)
    ax3 = axes[1, 0]

    sorted_counts = sorted(defendant_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    cumsum_pct = (cumsum / cumsum[-1]) * 100

    ax3_twin = ax3.twinx()

    bars = ax3.bar(range(len(sorted_counts)), sorted_counts, color='steelblue',
                   edgecolor='black', alpha=0.7, label='Case Count')
    line = ax3_twin.plot(range(len(cumsum_pct)), cumsum_pct, 'r-', linewidth=2,
                         marker='o', markersize=4, label='Cumulative %')
    ax3_twin.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.7)

    ax3.set_xlabel('Country Rank', fontsize=12)
    ax3.set_ylabel('Number of Cases', fontsize=12, color='steelblue')
    ax3_twin.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='red')
    ax3.set_title('Pareto Analysis (80/20 Rule)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_defendant_instances = sum(defendant_counts.values())
    unique_defendants = len(defendant_counts)

    summary_text = f"""
    DEFENDANT COUNTRIES STATISTICS
    {'─' * 50}

    Total cases:                    {len(data):,}
    Total defendant instances:      {total_defendant_instances:,}
    Unique countries:               {unique_defendants}

    Top 5 Defendant Countries:
    """

    for i, (country, count) in enumerate(defendant_counts.most_common(5), 1):
        pct = (count / len(data)) * 100
        summary_text += f"\n      {i}. {country:20} {count:5,} ({pct:5.1f}%)"

    # Find concentration
    top_10_pct = sum(c for _, c in defendant_counts.most_common(10)) / total_defendant_instances * 100
    top_20_pct = sum(c for _, c in defendant_counts.most_common(20)) / total_defendant_instances * 100

    summary_text += f"""

    Concentration:
      • Top 10 countries:           {top_10_pct:.1f}%
      • Top 20 countries:           {top_20_pct:.1f}%

    Regional Distribution:
    """

    for region, count in region_counts.most_common():
        pct = (count / total_defendant_instances) * 100
        summary_text += f"\n      • {region:20} {count:5,} ({pct:5.1f}%)"

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "defendant_countries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_case_outcomes(data):
    print("\n4. Generating case outcomes analysis...")

    # Count outcomes
    violated_count = 0
    non_violated_count = 0
    violations_per_case = []

    for case in data:
        violated_articles = case.get('violated_articles', [])
        num_violations = len(violated_articles)
        violations_per_case.append(num_violations)

        if num_violations > 0:
            violated_count += 1
        else:
            non_violated_count += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Outcome pie chart
    ax1 = axes[0, 0]
    labels = ['At Least One Violation', 'No Violations']
    sizes = [violated_count, non_violated_count]
    colors = ['#e74c3c', '#2ecc71']
    explode = (0.1, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                         autopct='%1.1f%%', colors=colors, startangle=90,
                                         textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title(f'Case Outcomes\n({len(data):,} total cases)',
                  fontsize=14, fontweight='bold')

    # 2. Distribution of violations per case
    ax2 = axes[0, 1]
    violation_counts = Counter(violations_per_case)
    max_violations = max(violations_per_case)

    x_vals = list(range(0, max_violations + 1))
    y_vals = [violation_counts.get(i, 0) for i in x_vals]

    bars = ax2.bar(x_vals, y_vals, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Number of Violated Articles per Case', fontsize=12)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Distribution of Violations per Case', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight zero violations
    if len(bars) > 0:
        bars[0].set_color('#2ecc71')
        bars[0].set_alpha(0.8)

    # 3. Box plot of violations
    ax3 = axes[1, 0]
    violations_only = [v for v in violations_per_case if v > 0]

    if len(violations_only) > 0:
        bp = ax3.boxplot(violations_only, vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][0].set_edgecolor('black')
        bp['boxes'][0].set_linewidth(2)

        q1, median, q3 = np.percentile(violations_only, [25, 50, 75])
        stats_text = f'Min: {min(violations_only)}\nQ1: {q1:.1f}\nMedian: {median:.1f}\nQ3: {q3:.1f}\nMax: {max(violations_only)}\nMean: {np.mean(violations_only):.2f}'
        ax3.text(1.35, median, stats_text, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_ylabel('Number of Violated Articles', fontsize=12)
        ax3.set_title('Violation Count Distribution\n(Cases with violations only)',
                      fontsize=14, fontweight='bold')
        ax3.set_xticklabels(['Violated Cases'])
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No violation data available', ha='center', va='center')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    violations_array = np.array(violations_per_case)
    summary_text = f"""
    CASE OUTCOMES STATISTICS
    {'─' * 50}

    Total cases:                    {len(data):,}
    Cases with violations:          {violated_count:,} ({violated_count/len(data)*100:.1f}%)
    Cases without violations:       {non_violated_count:,} ({non_violated_count/len(data)*100:.1f}%)

    Violations per Case:
      Mean:                         {np.mean(violations_array):.2f}
      Median:                       {np.median(violations_array):.1f}
      Std Dev:                      {np.std(violations_array):.2f}
      Min:                          {min(violations_array)}
      Max:                          {max(violations_array)}

    For cases WITH violations:
      Mean violations:              {np.mean(violations_only) if violations_only else 0:.2f}
      Median violations:            {np.median(violations_only) if violations_only else 0:.1f}
      Most common count:            {Counter(violations_only).most_common(1)[0] if violations_only else (0, 0)}

    Violation Distribution:
    """

    for num_viol in sorted(set(violations_per_case))[:8]:  # Show first 8
        count = violation_counts[num_viol]
        pct = (count / len(data)) * 100
        summary_text += f"\n      {num_viol} violations:              {count:5,} ({pct:5.1f}%)"

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "case_outcomes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_article_cooccurrence(data):
    print("\n5. Generating article co-occurrence analysis...")

    # Build co-occurrence matrix for top 15 articles
    article_counts = Counter()
    for case in data:
        violated_articles = case.get('violated_articles', [])
        for article in violated_articles:
            article_counts[article] += 1

    top_15_articles = [art for art, _ in article_counts.most_common(15)]

    # Co-occurrence matrix
    cooccurrence = defaultdict(lambda: defaultdict(int))

    for case in data:
        violated_articles = case.get('violated_articles', [])
        # Filter to top 15
        violated_top = [art for art in violated_articles if art in top_15_articles]

        # Count co-occurrences
        for art1, art2 in itertools.combinations(violated_top, 2):
            cooccurrence[art1][art2] += 1
            cooccurrence[art2][art1] += 1  # Symmetric

    # Create matrix
    matrix = np.zeros((len(top_15_articles), len(top_15_articles)))
    for i, art1 in enumerate(top_15_articles):
        for j, art2 in enumerate(top_15_articles):
            if i != j:
                matrix[i, j] = cooccurrence[art1][art2]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 1. Heatmap
    ax1 = axes[0]
    im = ax1.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax1.set_xticks(range(len(top_15_articles)))
    ax1.set_yticks(range(len(top_15_articles)))
    ax1.set_xticklabels(top_15_articles, rotation=45, ha='right')
    ax1.set_yticklabels(top_15_articles)

    # Add values
    for i in range(len(top_15_articles)):
        for j in range(len(top_15_articles)):
            if matrix[i, j] > 0:
                text = ax1.text(j, i, int(matrix[i, j]),
                               ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white",
                               fontsize=8)

    ax1.set_title('Article Co-occurrence Matrix (Top 15 Articles)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Number of Co-occurrences')

    # 2. Top co-occurrence pairs
    ax2 = axes[1]

    # Find top pairs
    pairs = []
    for art1 in top_15_articles:
        for art2 in top_15_articles:
            if art1 < art2:  # Avoid duplicates
                count = cooccurrence[art1][art2]
                if count > 0:
                    pairs.append((f"{art1} + {art2}", count))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:15]

    pair_labels = [p for p, _ in top_pairs]
    pair_counts = [c for _, c in top_pairs]

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(pair_labels)))
    bars = ax2.barh(range(len(pair_labels)), pair_counts, color=colors,
                    edgecolor='black', linewidth=1.5)

    ax2.set_yticks(range(len(pair_labels)))
    ax2.set_yticklabels(pair_labels, fontsize=9)
    ax2.set_xlabel('Number of Cases', fontsize=12)
    ax2.set_title('Top 15 Article Co-occurrence Pairs', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, count in zip(bars, pair_counts):
        ax2.text(count + 5, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "article_cooccurrence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def generate_dataset_report(data):
    print("\n6. Generating dataset overview report...")

    # Collect statistics
    article_counts = Counter()
    defendant_counts = Counter()
    years = []
    violations_per_case = []

    for case in data:
        violated_articles = case.get('violated_articles', [])
        violations_per_case.append(len(violated_articles))

        for article in violated_articles:
            article_counts[article] += 1

        defendants = case.get('defendants', [])
        for defendant in defendants:
            defendant_counts[defendant] += 1

        judgment_date = case.get('judgment_date')
        if judgment_date:
            try:
                year = int(judgment_date.split('-')[0])
                if 1950 <= year <= 2030:
                    years.append(year)
            except:
                pass

    violations_array = np.array(violations_per_case)
    violated_count = sum(1 for v in violations_per_case if v > 0)

    report = f"""
{'=' * 70}
ECHR DATASET OVERVIEW REPORT
{'=' * 70}

Dataset: {DATASET_PATH}
Generated: {Path(__file__).name}

{'─' * 70}
DATASET COMPOSITION
{'─' * 70}

Total cases:                     {len(data):,}
Cases with violations:           {violated_count:,} ({violated_count/len(data)*100:.1f}%)
Cases without violations:        {len(data) - violated_count:,} ({(len(data) - violated_count)/len(data)*100:.1f}%)

{'─' * 70}
VIOLATED ARTICLES
{'─' * 70}

Unique articles violated:        {len(article_counts)}
Total violation instances:       {sum(article_counts.values()):,}
Avg violations per case:         {np.mean(violations_array):.2f}
Avg articles violated per case:  {sum(article_counts.values())/len(data):.2f}

Top 10 Most Violated Articles:
"""

    for i, (art, count) in enumerate(article_counts.most_common(10), 1):
        pct = (count / len(data)) * 100
        report += f"  {i:2}. {art:20} {count:6,} cases ({pct:5.1f}%)\n"

    report += f"""
{'─' * 70}
DEFENDANT COUNTRIES
{'─' * 70}

Unique defendant countries:      {len(defendant_counts)}
Total defendant instances:       {sum(defendant_counts.values()):,}

Top 10 Defendant Countries:
"""

    for i, (country, count) in enumerate(defendant_counts.most_common(10), 1):
        pct = (count / len(data)) * 100
        report += f"  {i:2}. {country:25} {count:6,} cases ({pct:5.1f}%)\n"

    report += f"""
{'─' * 70}
TEMPORAL DISTRIBUTION
{'─' * 70}

Cases with judgment dates:       {len(years):,} ({len(years)/len(data)*100:.1f}%)
Date range:                      {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}
Span:                            {max(years) - min(years) if years else 0} years

Judgment years:
  Mean:                          {int(np.mean(years)) if years else 'N/A'}
  Median:                        {int(np.median(years)) if years else 'N/A'}
  Std Dev:                       {np.std(years):.1f if years else 'N/A'}

{'─' * 70}
DATA QUALITY INDICATORS
{'─' * 70}

Field Availability:
  • judgment_date:               {len(years)/len(data)*100:.1f}% present
  • violated_articles:           {sum(1 for c in data if c.get('violated_articles'))/len(data)*100:.1f}% present
  • defendants:                  {sum(1 for c in data if c.get('defendants'))/len(data)*100:.1f}% present
  • applicants:                  {sum(1 for c in data if c.get('applicants'))/len(data)*100:.1f}% present
  • facts:                       {sum(1 for c in data if c.get('facts'))/len(data)*100:.1f}% present

Dataset Characteristics:
  • Average case complexity:     {np.mean(violations_array):.2f} violated articles
  • Article diversity:           {len(article_counts)} unique articles
  • Geographic coverage:         {len(defendant_counts)} countries
  • Temporal coverage:           {max(years) - min(years) if years else 0} years

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

    output_path = OUTPUT_DIR / "dataset_overview_report.txt"
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"   ✅ Saved: {output_path}")

def main():
    print("=" * 70)
    print("DATASET OVERVIEW VISUALIZATION")
    print("=" * 70)

    # Load data
    data = load_data()

    # Generate all visualizations
    plot_article_distribution(data)
    plot_temporal_trends(data)
    plot_defendant_countries(data)
    plot_case_outcomes(data)
    plot_article_cooccurrence(data)
    generate_dataset_report(data)

    print("\n" + "=" * 70)
    print("✅ All dataset visualizations completed!")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  1. article_distribution.png")
    print("  2. temporal_trends.png")
    print("  3. defendant_countries.png")
    print("  4. case_outcomes.png")
    print("  5. article_cooccurrence.png")
    print("  6. dataset_overview_report.txt")

if __name__ == "__main__":
    main()