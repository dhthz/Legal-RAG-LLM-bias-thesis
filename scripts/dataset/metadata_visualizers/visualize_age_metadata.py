#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATASET_PATH = "dataset/train_with_metadata.jsonl"
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    print("Loading data from train_with_metadata.jsonl...")

    data = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            case = json.loads(line)
            data.append(case)

    print(f"✅ Loaded {len(data)} cases")
    return data

def plot_age_distribution(data):
    print("\n1. Generating age distribution plots...")

    # Extract ages
    ages = []
    age_sources = []
    for case in data:
        age_info = case.get('age_info', {})
        if age_info and age_info.get('age_at_judgment'):
            ages.append(age_info['age_at_judgment'])
            age_sources.append(age_info.get('age_source', 'unknown'))

    ages = np.array(ages)

    # Define age groups
    def categorize_age(age):
        if age < 18:
            return 'Minor (<18)'
        elif age < 30:
            return 'Young Adult (18-29)'
        elif age < 50:
            return 'Adult (30-49)'
        elif age < 65:
            return 'Middle-aged (50-64)'
        else:
            return 'Senior (65+)'

    age_groups = [categorize_age(age) for age in ages]
    age_group_counts = Counter(age_groups)

    # Count by source
    source_counts = Counter(age_sources)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Age histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(ages, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Add KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ages)
    x_range = np.linspace(ages.min(), ages.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    ax1.axvline(np.median(ages), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ages):.1f}')
    ax1.axvline(np.mean(ages), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ages):.1f}')

    ax1.set_xlabel('Age at Judgment', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Age Distribution (n={len(ages):,})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Age groups bar chart
    ax2 = axes[0, 1]
    age_group_order = ['Minor (<18)', 'Young Adult (18-29)', 'Adult (30-49)',
                       'Middle-aged (50-64)', 'Senior (65+)']
    counts = [age_group_counts.get(group, 0) for group in age_group_order]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax2.bar(range(len(age_group_order)), counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(age_group_order)))
    ax2.set_xticklabels(age_group_order, rotation=45, ha='right')
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Age Groups Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(ages)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    # 3. Age source breakdown
    ax3 = axes[1, 0]
    source_labels = list(source_counts.keys())
    source_values = list(source_counts.values())
    colors_source = ['#3498db', '#e67e22', '#95a5a6']

    wedges, texts, autotexts = ax3.pie(source_values, labels=source_labels, autopct='%1.1f%%',
                                         colors=colors_source, startangle=90,
                                         textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax3.set_title('Age Information Source', fontsize=14, fontweight='bold')

    # 4. Box plot with quartiles
    ax4 = axes[1, 1]
    bp = ax4.boxplot(ages, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(2)

    # Add statistical annotations
    q1, median, q3 = np.percentile(ages, [25, 50, 75])
    iqr = q3 - q1

    stats_text = f'Min: {ages.min()}\nQ1: {q1:.1f}\nMedian: {median:.1f}\nQ3: {q3:.1f}\nMax: {ages.max()}\nIQR: {iqr:.1f}\nStd: {np.std(ages):.1f}'
    ax4.text(1.35, np.median(ages), stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_ylabel('Age at Judgment', fontsize=12)
    ax4.set_title('Age Distribution Statistics', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(['All Cases'])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_age_by_article(data):
    print("\n2. Generating age by article analysis...")

    # Extract age data by article
    article_ages = defaultdict(list)

    for case in data:
        age_info = case.get('age_info', {})
        age = age_info.get('age_at_judgment') if age_info else None

        if age:
            violated_articles = case.get('violated_articles', [])
            for article in violated_articles:
                article_ages[article].append(age)

    # Get top 12 most common articles
    article_counts = {art: len(ages) for art, ages in article_ages.items()}
    top_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    top_article_names = [art for art, _ in top_articles]

    # Prepare data for ANOVA
    age_groups = [article_ages[art] for art in top_article_names]

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*age_groups)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 1. Boxplot comparison
    ax1 = axes[0]
    positions = range(len(top_article_names))
    bp = ax1.boxplot(age_groups, positions=positions, patch_artist=True, widths=0.6)

    # Color boxes by median age
    medians = [np.median(ages) for ages in age_groups]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(medians)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(top_article_names, rotation=45, ha='right')
    ax1.set_ylabel('Age at Judgment', fontsize=12)
    ax1.set_title(f'Age Distribution by Article (ANOVA: F={f_stat:.2f}, p={p_value:.2e})',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample sizes
    for i, ages in enumerate(age_groups):
        ax1.text(i, ax1.get_ylim()[1], f'n={len(ages)}',
                ha='center', va='bottom', fontsize=8, rotation=45)

    # 2. Mean age with confidence intervals
    ax2 = axes[1]
    means = [np.mean(ages) for ages in age_groups]
    stds = [np.std(ages) for ages in age_groups]
    ns = [len(ages) for ages in age_groups]
    sems = [std / np.sqrt(n) for std, n in zip(stds, ns)]
    ci_95 = [1.96 * sem for sem in sems]

    bars = ax2.bar(positions, means, yerr=ci_95, capsize=5,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(top_article_names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Age (±95% CI)', fontsize=12)
    ax2.set_title('Mean Age by Article with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax2.axhline(y=np.mean([item for sublist in age_groups for item in sublist]),
                color='red', linestyle='--', linewidth=2, label='Overall Mean')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, mean, ci) in enumerate(zip(bars, means, ci_95)):
        ax2.text(bar.get_x() + bar.get_width()/2., mean + ci,
                f'{mean:.1f}±{ci:.1f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_by_article.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

    return f_stat, p_value, top_article_names, age_groups

def plot_age_by_gender(data):
    print("\n3. Generating age by gender cross-tabulation...")

    # Extract age and gender data
    male_ages = []
    female_ages = []

    for case in data:
        age_info = case.get('age_info', {})
        age = age_info.get('age_at_judgment') if age_info else None

        classification = case.get('classification', {})
        gender = classification.get('gender')

        if age and gender:
            if gender == 'male' or gender == 'Male':
                male_ages.append(age)
            elif gender == 'female' or gender == 'Female':
                female_ages.append(age)

    # Check if we have sufficient data
    if len(male_ages) == 0 and len(female_ages) == 0:
        print("   ⚠️  Skipping age by gender analysis - no gender+age data available")
        return None, None, None

    if len(male_ages) == 0 or len(female_ages) == 0:
        print(f"   ⚠️  Warning: Insufficient data (male={len(male_ages)}, female={len(female_ages)})")
        print("   ⚠️  Skipping statistical tests and some plots")
        t_stat, p_value, cohens_d = None, None, None
    else:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(male_ages, female_ages)
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(male_ages)**2 + np.std(female_ages)**2) / 2)
        cohens_d = (np.mean(male_ages) - np.mean(female_ages)) / pooled_std if pooled_std > 0 else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overlapping histograms
    ax1 = axes[0, 0]
    if len(male_ages) > 0:
        ax1.hist(male_ages, bins=30, alpha=0.6, label=f'Male (n={len(male_ages):,})',
                 color='#3498db', density=True, edgecolor='black')
        ax1.axvline(np.mean(male_ages), color='blue', linestyle='--', linewidth=2, alpha=0.7)

    if len(female_ages) > 0:
        ax1.hist(female_ages, bins=30, alpha=0.6, label=f'Female (n={len(female_ages):,})',
                 color='#e74c3c', density=True, edgecolor='black')
        ax1.axvline(np.mean(female_ages), color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Age at Judgment', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)

    if t_stat is not None:
        ax1.set_title(f'Age Distribution by Gender\n(t={t_stat:.2f}, p={p_value:.4f}, d={cohens_d:.3f})',
                      fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Age Distribution by Gender\n(Insufficient data for statistical tests)',
                      fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plots side by side
    ax2 = axes[0, 1]
    plot_data = []
    plot_labels = []

    if len(male_ages) > 0:
        plot_data.append(male_ages)
        plot_labels.append('Male')
    if len(female_ages) > 0:
        plot_data.append(female_ages)
        plot_labels.append('Female')

    if len(plot_data) > 0:
        bp = ax2.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, widths=0.5)

        colors = ['#3498db', '#e74c3c']
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i] if i < len(colors) else '#95a5a6')

    ax2.set_ylabel('Age at Judgment', fontsize=12)
    ax2.set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistical annotations
    if len(male_ages) > 0:
        male_stats = f'Mean: {np.mean(male_ages):.1f}\nMedian: {np.median(male_ages):.1f}\nStd: {np.std(male_ages):.1f}'
        ax2.text(0.7, np.median(male_ages), male_stats, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    if len(female_ages) > 0:
        x_pos = 1.7 if len(male_ages) > 0 else 0.7
        female_stats = f'Mean: {np.mean(female_ages):.1f}\nMedian: {np.median(female_ages):.1f}\nStd: {np.std(female_ages):.1f}'
        ax2.text(x_pos, np.median(female_ages), female_stats, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # 3. Age group breakdown by gender
    ax3 = axes[1, 0]

    def categorize_age(age):
        if age < 18:
            return 'Minor'
        elif age < 30:
            return 'Young'
        elif age < 50:
            return 'Adult'
        elif age < 65:
            return 'Middle'
        else:
            return 'Senior'

    male_groups = Counter([categorize_age(age) for age in male_ages])
    female_groups = Counter([categorize_age(age) for age in female_ages])

    age_group_order = ['Minor', 'Young', 'Adult', 'Middle', 'Senior']
    male_counts = [male_groups.get(group, 0) for group in age_group_order]
    female_counts = [female_groups.get(group, 0) for group in age_group_order]

    x = np.arange(len(age_group_order))
    width = 0.35

    bars1 = ax3.bar(x - width/2, male_counts, width, label='Male', color='#3498db', edgecolor='black')
    bars2 = ax3.bar(x + width/2, female_counts, width, label='Female', color='#e74c3c', edgecolor='black')

    ax3.set_xlabel('Age Group', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Age Groups by Gender', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(age_group_order)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Violin plot or summary text
    ax4 = axes[1, 1]

    if len(plot_data) > 0 and all(len(d) > 1 for d in plot_data):
        # Only create violin plot if we have sufficient data
        positions = list(range(1, len(plot_data) + 1))
        parts = ax4.violinplot(plot_data, positions=positions,
                               showmeans=True, showmedians=True)

        colors_violin = ['#3498db', '#e74c3c']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_violin[i] if i < len(colors_violin) else '#95a5a6')
            pc.set_alpha(0.6)

        ax4.set_xticks(positions)
        ax4.set_xticklabels(plot_labels)
        ax4.set_ylabel('Age at Judgment', fontsize=12)
        ax4.set_title('Age Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        # Show summary text if insufficient data
        ax4.axis('off')
        summary = f"""
        AGE BY GENDER SUMMARY
        {'─' * 40}

        Male cases with age:     {len(male_ages):,}
        Female cases with age:   {len(female_ages):,}

        Note: Insufficient data for violin plot
        (need at least 2 data points per gender)
        """
        ax4.text(0.5, 0.5, summary, fontsize=11, family='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_by_gender.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

    return t_stat, p_value, cohens_d

def plot_temporal_trends(data):
    print("\n4. Generating temporal trends...")

    # Extract judgment years and ages
    year_ages = defaultdict(list)

    for case in data:
        age_info = case.get('age_info', {})
        age = age_info.get('age_at_judgment') if age_info else None

        judgment_date = case.get('judgment_date')

        if age and judgment_date:
            try:
                year = int(judgment_date.split('-')[0])
                if 1950 <= year <= 2030:  # Sanity check
                    year_ages[year].append(age)
            except:
                continue

    # Calculate statistics by year
    years = sorted(year_ages.keys())
    mean_ages = [np.mean(year_ages[year]) for year in years]
    median_ages = [np.median(year_ages[year]) for year in years]
    counts = [len(year_ages[year]) for year in years]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. Mean age over time with trend line
    ax1 = axes[0]
    ax1.plot(years, mean_ages, 'o-', color='#3498db', linewidth=2, markersize=4, label='Mean Age')
    ax1.plot(years, median_ages, 's-', color='#e67e22', linewidth=2, markersize=4, alpha=0.7, label='Median Age')

    # Add trend line
    z = np.polyfit(years, mean_ages, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')

    ax1.set_xlabel('Judgment Year', fontsize=12)
    ax1.set_ylabel('Age at Judgment', fontsize=12)
    ax1.set_title('Average Age Trends Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Number of cases with age info over time
    ax2 = axes[1]
    ax2.bar(years, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Judgment Year', fontsize=12)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Cases with Age Information Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Age distribution heatmap over time (5-year bins)
    ax3 = axes[2]

    # Create 5-year bins
    year_bins = list(range(min(years), max(years) + 5, 5))
    age_bins = list(range(0, 101, 10))

    heatmap_data = np.zeros((len(age_bins)-1, len(year_bins)-1))

    for case in data:
        age_info = case.get('age_info', {})
        age = age_info.get('age_at_judgment') if age_info else None
        judgment_date = case.get('judgment_date')

        if age and judgment_date:
            try:
                year = int(judgment_date.split('-')[0])
                if min(years) <= year <= max(years):
                    year_idx = np.digitize([year], year_bins)[0] - 1
                    age_idx = np.digitize([age], age_bins)[0] - 1

                    if 0 <= year_idx < len(year_bins)-1 and 0 <= age_idx < len(age_bins)-1:
                        heatmap_data[age_idx, year_idx] += 1
            except:
                continue

    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax3.set_xticks(range(len(year_bins)-1))
    ax3.set_xticklabels([f'{year_bins[i]}-{year_bins[i+1]}' for i in range(len(year_bins)-1)], rotation=45, ha='right')
    ax3.set_yticks(range(len(age_bins)-1))
    ax3.set_yticklabels([f'{age_bins[i]}-{age_bins[i+1]}' for i in range(len(age_bins)-1)])

    ax3.set_xlabel('Judgment Year (5-year bins)', fontsize=12)
    ax3.set_ylabel('Age Groups', fontsize=12)
    ax3.set_title('Age Distribution Heatmap Over Time', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax3, label='Number of Cases')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_temporal_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_age_availability(data):
    print("\n5. Generating age availability analysis...")

    total_cases = len(data)
    cases_with_age = 0
    cases_with_birth_year = 0
    age_sources = Counter()

    # Age availability by article
    article_age_availability = defaultdict(lambda: {'total': 0, 'with_age': 0})

    for case in data:
        age_info = case.get('age_info', {})
        has_age = bool(age_info and age_info.get('age_at_judgment'))
        has_birth = bool(age_info and age_info.get('birth_year'))

        if has_age:
            cases_with_age += 1
            age_sources[age_info.get('age_source', 'unknown')] += 1

        if has_birth:
            cases_with_birth_year += 1

        # Track by article
        violated_articles = case.get('violated_articles', [])
        for article in violated_articles:
            article_age_availability[article]['total'] += 1
            if has_age:
                article_age_availability[article]['with_age'] += 1

    # Calculate availability percentages
    age_availability_pct = (cases_with_age / total_cases) * 100
    birth_availability_pct = (cases_with_birth_year / total_cases) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall availability pie chart
    ax1 = axes[0, 0]
    labels = ['With Age Info', 'Without Age Info']
    sizes = [cases_with_age, total_cases - cases_with_age]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title(f'Age Information Availability\n({cases_with_age:,} / {total_cases:,} cases)',
                  fontsize=14, fontweight='bold')

    # 2. Age source breakdown
    ax2 = axes[0, 1]
    source_names = list(age_sources.keys())
    source_counts = list(age_sources.values())

    bars = ax2.bar(source_names, source_counts, color=['#3498db', '#e67e22', '#95a5a6'],
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Age Information Sources', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/cases_with_age*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # 3. Availability by article (top 15)
    ax3 = axes[1, 0]

    # Sort by total cases
    sorted_articles = sorted(article_age_availability.items(),
                            key=lambda x: x[1]['total'], reverse=True)[:15]

    article_names = [art for art, _ in sorted_articles]
    availability_pcts = [(stats['with_age'] / stats['total'] * 100)
                         for _, stats in sorted_articles]
    total_counts = [stats['total'] for _, stats in sorted_articles]

    bars = ax3.barh(range(len(article_names)), availability_pcts,
                    color=plt.cm.RdYlGn(np.array(availability_pcts)/100),
                    edgecolor='black', linewidth=1.5)

    ax3.set_yticks(range(len(article_names)))
    ax3.set_yticklabels(article_names)
    ax3.set_xlabel('Age Availability (%)', fontsize=12)
    ax3.set_title('Age Information Availability by Article (Top 15)', fontsize=14, fontweight='bold')
    ax3.axvline(x=age_availability_pct, color='red', linestyle='--', linewidth=2,
                label=f'Overall: {age_availability_pct:.1f}%')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # Add counts
    for i, (bar, pct, count) in enumerate(zip(bars, availability_pcts, total_counts)):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                f'{pct:.1f}% (n={count})',
                ha='left', va='center', fontsize=8)

    # 4. Missing data summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    AGE METADATA AVAILABILITY SUMMARY
    {'─' * 50}

    Total Cases:                {total_cases:,}
    Cases with Age:             {cases_with_age:,} ({age_availability_pct:.2f}%)
    Cases without Age:          {total_cases - cases_with_age:,} ({100 - age_availability_pct:.2f}%)

    Cases with Birth Year:      {cases_with_birth_year:,} ({birth_availability_pct:.2f}%)

    Age Information Sources:
    """

    for source, count in age_sources.most_common():
        pct = (count / cases_with_age) * 100
        summary_text += f"  • {source.capitalize():15} {count:5,} ({pct:5.1f}%)\n"

    summary_text += f"""

    Data Quality Notes:
    • {age_availability_pct:.1f}% of cases have age information
    • Age is primarily obtained from: {age_sources.most_common(1)[0][0]}
    • Missing age data may introduce bias in age-based analyses
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_availability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def plot_bias_indicators(data):
    print("\n6. Generating bias indicators...")

    # Extract ages
    ages = []
    for case in data:
        age_info = case.get('age_info', {})
        if age_info and age_info.get('age_at_judgment'):
            ages.append(age_info['age_at_judgment'])

    ages = np.array(ages)

    # Define quartiles
    q1, median, q3 = np.percentile(ages, [25, 50, 75])

    # Categorize into quartiles
    def get_quartile(age):
        if age <= q1:
            return 'Q1 (Youngest)'
        elif age <= median:
            return 'Q2'
        elif age <= q3:
            return 'Q3'
        else:
            return 'Q4 (Oldest)'

    quartile_counts = Counter([get_quartile(age) for age in ages])

    # Age distribution skewness
    from scipy.stats import skew, kurtosis
    age_skewness = skew(ages)
    age_kurtosis = kurtosis(ages)

    # Outlier detection (IQR method)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ages[(ages < lower_bound) | (ages > upper_bound)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Quartile distribution
    ax1 = axes[0, 0]
    quartile_order = ['Q1 (Youngest)', 'Q2', 'Q3', 'Q4 (Oldest)']
    counts = [quartile_counts.get(q, 0) for q in quartile_order]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    bars = ax1.bar(quartile_order, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Cases', fontsize=12)
    ax1.set_title('Age Quartile Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add threshold labels
    for i, (bar, count, q_label) in enumerate(zip(bars, counts, quartile_order)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\n({count/len(ages)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

        # Add age range
        if i == 0:
            range_text = f'≤{q1:.0f}'
        elif i == 1:
            range_text = f'{q1:.0f}-{median:.0f}'
        elif i == 2:
            range_text = f'{median:.0f}-{q3:.0f}'
        else:
            range_text = f'>{q3:.0f}'

        ax1.text(bar.get_x() + bar.get_width()/2., -height*0.1,
                range_text, ha='center', va='top', fontsize=9, style='italic')

    # 2. Distribution shape analysis
    ax2 = axes[0, 1]
    ax2.hist(ages, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = np.mean(ages), np.std(ages)
    x = np.linspace(ages.min(), ages.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')

    ax2.set_xlabel('Age at Judgment', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Distribution Shape\n(Skewness: {age_skewness:.3f}, Kurtosis: {age_kurtosis:.3f})',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Interpret skewness
    if abs(age_skewness) < 0.5:
        skew_interp = "Approximately symmetric"
    elif age_skewness > 0:
        skew_interp = "Right-skewed (younger bias)"
    else:
        skew_interp = "Left-skewed (older bias)"

    ax2.text(0.05, 0.95, skew_interp, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Outlier analysis
    ax3 = axes[1, 0]

    # Show all data points with outliers highlighted
    normal_ages = ages[(ages >= lower_bound) & (ages <= upper_bound)]

    ax3.scatter(range(len(normal_ages)), sorted(normal_ages),
               alpha=0.5, s=10, color='steelblue', label='Normal Range')

    if len(outliers) > 0:
        outlier_positions = [i for i, age in enumerate(sorted(ages)) if age in outliers]
        ax3.scatter(outlier_positions, sorted(outliers),
                   alpha=0.8, s=50, color='red', marker='x', linewidths=2, label='Outliers')

    ax3.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Lower Bound')
    ax3.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Upper Bound')

    ax3.set_xlabel('Case Index (sorted by age)', fontsize=12)
    ax3.set_ylabel('Age at Judgment', fontsize=12)
    ax3.set_title(f'Outlier Detection (IQR Method)\n{len(outliers)} outliers ({len(outliers)/len(ages)*100:.2f}%)',
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Bias summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    bias_text = f"""
    AGE BIAS INDICATORS
    {'─' * 50}

    Distribution Statistics:
      Mean:                     {np.mean(ages):.2f} years
      Median:                   {np.median(ages):.2f} years
      Mode:                     {stats.mode(ages.astype(int), keepdims=True).mode[0]} years
      Std Dev:                  {np.std(ages):.2f} years

    Quartiles:
      Q1 (25th percentile):     {q1:.2f} years
      Q2 (50th percentile):     {median:.2f} years
      Q3 (75th percentile):     {q3:.2f} years
      IQR:                      {iqr:.2f} years

    Shape Analysis:
      Skewness:                 {age_skewness:.3f} ({skew_interp})
      Kurtosis:                 {age_kurtosis:.3f}
      Range:                    {ages.min():.0f} - {ages.max():.0f} years

    Outliers:
      Count:                    {len(outliers)} ({len(outliers)/len(ages)*100:.2f}%)
      Lower bound:              {lower_bound:.2f} years
      Upper bound:              {upper_bound:.2f} years

    Potential Bias Concerns:
      • Skewness indicates {skew_interp.lower()}
      • {len(outliers)/len(ages)*100:.1f}% outliers may affect analyses
      • Age range spans {ages.max() - ages.min():.0f} years
    """

    ax4.text(0.1, 0.5, bias_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "age_bias_indicators.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def generate_statistical_report(data):
    print("\n7. Generating statistical report...")

    # Extract all ages
    ages = []
    for case in data:
        age_info = case.get('age_info', {})
        if age_info and age_info.get('age_at_judgment'):
            ages.append(age_info['age_at_judgment'])

    ages = np.array(ages)

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(ages) if len(ages) < 5000 else (None, None)

    # Calculate comprehensive statistics
    report = f"""
{'=' * 70}
AGE METADATA STATISTICAL ANALYSIS REPORT
{'=' * 70}

Dataset: {DATASET_PATH}
Generated: {Path(__file__).name}

{'─' * 70}
SAMPLE STATISTICS
{'─' * 70}

Total cases analyzed:        {len(data):,}
Cases with age info:         {len(ages):,} ({len(ages)/len(data)*100:.2f}%)
Cases without age info:      {len(data) - len(ages):,} ({(len(data) - len(ages))/len(data)*100:.2f}%)

{'─' * 70}
DESCRIPTIVE STATISTICS
{'─' * 70}

Central Tendency:
  Mean:                      {np.mean(ages):.2f} years
  Median:                    {np.median(ages):.2f} years
  Mode:                      {stats.mode(ages.astype(int), keepdims=True).mode[0]} years

Dispersion:
  Standard Deviation:        {np.std(ages):.2f} years
  Variance:                  {np.var(ages):.2f}
  Range:                     {ages.max() - ages.min():.0f} years ({ages.min():.0f} - {ages.max():.0f})
  IQR:                       {np.percentile(ages, 75) - np.percentile(ages, 25):.2f} years

Quartiles:
  Q1 (25th percentile):      {np.percentile(ages, 25):.2f} years
  Q2 (50th percentile):      {np.percentile(ages, 50):.2f} years
  Q3 (75th percentile):      {np.percentile(ages, 75):.2f} years

Percentiles:
  5th percentile:            {np.percentile(ages, 5):.2f} years
  10th percentile:           {np.percentile(ages, 10):.2f} years
  90th percentile:           {np.percentile(ages, 90):.2f} years
  95th percentile:           {np.percentile(ages, 95):.2f} years

{'─' * 70}
DISTRIBUTION SHAPE
{'─' * 70}

Skewness:                    {stats.skew(ages):.4f}
  Interpretation:            {'Right-skewed (younger bias)' if stats.skew(ages) > 0 else 'Left-skewed (older bias)' if stats.skew(ages) < -0.5 else 'Approximately symmetric'}

Kurtosis:                    {stats.kurtosis(ages):.4f}
  Interpretation:            {'Leptokurtic (heavy tails)' if stats.kurtosis(ages) > 0 else 'Platykurtic (light tails)'}

"""

    if shapiro_stat is not None:
        report += f"""Normality Test (Shapiro-Wilk):
  Statistic:                 {shapiro_stat:.4f}
  p-value:                   {shapiro_p:.4e}
  Result:                    {'Data is NOT normally distributed (p < 0.05)' if shapiro_p < 0.05 else 'Data is approximately normally distributed (p ≥ 0.05)'}

"""

    report += f"""{'─' * 70}
AGE GROUPS BREAKDOWN
{'─' * 70}

"""

    def categorize_age(age):
        if age < 18:
            return 'Minor (<18)'
        elif age < 30:
            return 'Young Adult (18-29)'
        elif age < 50:
            return 'Adult (30-49)'
        elif age < 65:
            return 'Middle-aged (50-64)'
        else:
            return 'Senior (65+)'

    age_groups = Counter([categorize_age(age) for age in ages])
    for group in ['Minor (<18)', 'Young Adult (18-29)', 'Adult (30-49)', 'Middle-aged (50-64)', 'Senior (65+)']:
        count = age_groups.get(group, 0)
        pct = (count / len(ages)) * 100
        report += f"  {group:25} {count:6,} ({pct:5.2f}%)\n"

    report += f"""
{'─' * 70}
OUTLIER ANALYSIS (IQR METHOD)
{'─' * 70}

"""

    q1, q3 = np.percentile(ages, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ages[(ages < lower_bound) | (ages > upper_bound)]

    report += f"""Lower bound:                 {lower_bound:.2f} years
Upper bound:                 {upper_bound:.2f} years
Number of outliers:          {len(outliers)} ({len(outliers)/len(ages)*100:.2f}%)
"""

    if len(outliers) > 0:
        report += f"""Outlier ages:                {', '.join([str(int(age)) for age in sorted(outliers)])}
"""

    report += f"""
{'─' * 70}
BIAS ASSESSMENT
{'─' * 70}

Age Distribution Characteristics:
  • {'Right-skewed distribution suggests younger applicants are overrepresented' if stats.skew(ages) > 0.5 else 'Left-skewed distribution suggests older applicants are overrepresented' if stats.skew(ages) < -0.5 else 'Approximately symmetric age distribution'}
  • Mean age ({np.mean(ages):.1f}) {'>' if np.mean(ages) > np.median(ages) else '<' if np.mean(ages) < np.median(ages) else '≈'} Median age ({np.median(ages):.1f})
  • {len(outliers)} outliers detected ({len(outliers)/len(ages)*100:.2f}% of cases)

Missing Data Pattern:
  • {(len(data) - len(ages))/len(data)*100:.1f}% of cases lack age information
  • Missing age data may introduce selection bias

Potential RAG System Implications:
  • Age-based retrieval may favor {'younger' if stats.skew(ages) > 0 else 'older'} applicants
  • Outlier ages may affect similarity scoring
  • Missing age data affects {(len(data) - len(ages))/len(data)*100:.1f}% of cases

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

    output_path = OUTPUT_DIR / "age_statistical_report.txt"
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"   ✅ Saved: {output_path}")

def main():
    print("=" * 70)
    print("AGE METADATA VISUALIZATION")
    print("=" * 70)

    # Load data
    data = load_data()

    # Generate all visualizations
    plot_age_distribution(data)
    plot_age_by_article(data)
    plot_age_by_gender(data)
    plot_temporal_trends(data)
    plot_age_availability(data)
    plot_bias_indicators(data)
    generate_statistical_report(data)

    print("\n" + "=" * 70)
    print("✅ All age visualizations completed!")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  1. age_distribution.png")
    print("  2. age_by_article.png")
    print("  3. age_by_gender.png")
    print("  4. age_temporal_trends.png")
    print("  5. age_availability.png")
    print("  6. age_bias_indicators.png")
    print("  7. age_statistical_report.txt")

if __name__ == "__main__":
    main()
