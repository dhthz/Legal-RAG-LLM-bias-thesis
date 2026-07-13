import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path


def load_sentiment_data(dataset_path):
    print(f"Loading sentiment data from {dataset_path}...")

    cases = []
    sentiment_data = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                case = json.loads(line)
                cases.append(case)

                if 'sentiment_info' in case:
                    sentiment_info = case['sentiment_info'].copy()
                    sentiment_info['case_id'] = case['case_id']
                    sentiment_info['violated_articles'] = case.get('violated_articles', [])
                    sentiment_data.append(sentiment_info)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipped line {line_num} due to JSON error: {e}")
                continue

    print(f"Loaded {len(sentiment_data)} cases with sentiment metadata\n")
    return cases, sentiment_data


def compute_basic_statistics(sentiment_data):

    metrics = [
        'word_count', 'emotional_word_count', 'nrc_fear', 'nrc_anger',
        'nrc_sadness', 'nrc_disgust', 'nrc_emotional_intensity',
        'sentence_count', 'passive_voice_count', 'passive_voice_ratio',
        'first_person_pronouns', 'third_person_pronouns',
        'perpetrator_mentions', 'victim_language_count'
    ]

    stats = {}

    for metric in metrics:
        values = [s[metric] for s in sentiment_data if metric in s]

        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
            }

    return stats


def analyze_by_article(sentiment_data):

    article_sentiment = defaultdict(list)

    for s in sentiment_data:
        for article in s.get('violated_articles', []):
            article_sentiment[article].append(s)

    # Compute statistics for each article
    article_stats = {}

    for article, sentiments in article_sentiment.items():
        if len(sentiments) < 10:  # Skip articles with too few cases
            continue

        article_stats[article] = {
            'count': len(sentiments),
            'avg_emotional_intensity': np.mean([s['nrc_emotional_intensity'] for s in sentiments]),
            'avg_fear': np.mean([s['nrc_fear'] for s in sentiments]),
            'avg_anger': np.mean([s['nrc_anger'] for s in sentiments]),
            'avg_sadness': np.mean([s['nrc_sadness'] for s in sentiments]),
            'avg_disgust': np.mean([s['nrc_disgust'] for s in sentiments]),
            'avg_passive_voice': np.mean([s['passive_voice_ratio'] for s in sentiments]),
            'avg_victim_language': np.mean([s['victim_language_count'] for s in sentiments]),
            'avg_perpetrator_mentions': np.mean([s['perpetrator_mentions'] for s in sentiments]),
        }

    return article_stats


def identify_outliers(sentiment_data):

    # Sort by emotional intensity
    sorted_by_intensity = sorted(sentiment_data, key=lambda x: x['nrc_emotional_intensity'], reverse=True)

    outliers = {
        'highest_intensity': sorted_by_intensity[:10],
        'lowest_intensity': sorted_by_intensity[-10:],
    }

    # High fear cases
    sorted_by_fear = sorted(sentiment_data, key=lambda x: x['nrc_fear'], reverse=True)
    outliers['highest_fear'] = sorted_by_fear[:10]

    # High anger cases
    sorted_by_anger = sorted(sentiment_data, key=lambda x: x['nrc_anger'], reverse=True)
    outliers['highest_anger'] = sorted_by_anger[:10]

    # High passive voice
    sorted_by_passive = sorted(sentiment_data, key=lambda x: x['passive_voice_ratio'], reverse=True)
    outliers['highest_passive_voice'] = sorted_by_passive[:10]

    return outliers


def compute_correlations(sentiment_data):

    # Extract key metrics
    emotional_intensity = [s['nrc_emotional_intensity'] for s in sentiment_data]
    passive_voice = [s['passive_voice_ratio'] for s in sentiment_data]
    victim_language = [s['victim_language_count'] for s in sentiment_data]
    perpetrator_mentions = [s['perpetrator_mentions'] for s in sentiment_data]
    word_count = [s['word_count'] for s in sentiment_data]

    correlations = {
        'emotional_intensity_vs_passive_voice': np.corrcoef(emotional_intensity, passive_voice)[0, 1],
        'emotional_intensity_vs_victim_language': np.corrcoef(emotional_intensity, victim_language)[0, 1],
        'victim_language_vs_perpetrator_mentions': np.corrcoef(victim_language, perpetrator_mentions)[0, 1],
        'word_count_vs_emotional_intensity': np.corrcoef(word_count, emotional_intensity)[0, 1],
        'passive_voice_vs_victim_language': np.corrcoef(passive_voice, victim_language)[0, 1],
    }

    return correlations


def generate_report(stats, article_stats, outliers, correlations, output_path):

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "SENTIMENT METADATA ANALYSIS REPORT\n")
        f.write(" " * 15 + "ECHR Legal Cases - Emotional & Linguistic Patterns\n")
        f.write("=" * 80 + "\n\n")

        # 1. BASIC STATISTICS
        f.write("=" * 80 + "\n")
        f.write("1. BASIC STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for metric, vals in stats.items():
            f.write(f"{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Mean:       {vals['mean']:.4f}\n")
            f.write(f"  Median:     {vals['median']:.4f}\n")
            f.write(f"  Std Dev:    {vals['std']:.4f}\n")
            f.write(f"  Min:        {vals['min']:.4f}\n")
            f.write(f"  Max:        {vals['max']:.4f}\n")
            f.write(f"  Q1 (25%):   {vals['q25']:.4f}\n")
            f.write(f"  Q3 (75%):   {vals['q75']:.4f}\n")
            f.write(f"  Range:      {vals['max'] - vals['min']:.4f}\n\n")

        # 2. EMOTIONAL INTENSITY INSIGHTS
        f.write("=" * 80 + "\n")
        f.write("2. EMOTIONAL INTENSITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Average Emotional Intensity: {stats['nrc_emotional_intensity']['mean']:.2%}\n")
        f.write(f"  - This represents the proportion of emotionally charged words\n")
        f.write(f"  - Range: {stats['nrc_emotional_intensity']['min']:.2%} to {stats['nrc_emotional_intensity']['max']:.2%}\n\n")

        f.write("Emotion Breakdown (Average Proportions):\n")
        f.write(f"  Fear:      {stats['nrc_fear']['mean']:.2%}  (range: {stats['nrc_fear']['min']:.2%} - {stats['nrc_fear']['max']:.2%})\n")
        f.write(f"  Sadness:   {stats['nrc_sadness']['mean']:.2%}  (range: {stats['nrc_sadness']['min']:.2%} - {stats['nrc_sadness']['max']:.2%})\n")
        f.write(f"  Anger:     {stats['nrc_anger']['mean']:.2%}  (range: {stats['nrc_anger']['min']:.2%} - {stats['nrc_anger']['max']:.2%})\n")
        f.write(f"  Disgust:   {stats['nrc_disgust']['mean']:.2%}  (range: {stats['nrc_disgust']['min']:.2%} - {stats['nrc_disgust']['max']:.2%})\n\n")

        # Determine dominant emotion
        emotion_avgs = {
            'Fear': stats['nrc_fear']['mean'],
            'Sadness': stats['nrc_sadness']['mean'],
            'Anger': stats['nrc_anger']['mean'],
            'Disgust': stats['nrc_disgust']['mean'],
        }
        dominant_emotion = max(emotion_avgs, key=emotion_avgs.get)
        f.write(f"Dominant Emotion: {dominant_emotion} ({emotion_avgs[dominant_emotion]:.2%})\n\n")

        # 3. LINGUISTIC FEATURES
        f.write("=" * 80 + "\n")
        f.write("3. LINGUISTIC FEATURES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Average Case Length: {stats['word_count']['mean']:.0f} words ({stats['sentence_count']['mean']:.0f} sentences)\n")
        f.write(f"  - Shortest case: {stats['word_count']['min']:.0f} words\n")
        f.write(f"  - Longest case:  {stats['word_count']['max']:.0f} words\n\n")

        f.write(f"Passive Voice Usage: {stats['passive_voice_ratio']['mean']:.2%} of sentences\n")
        f.write(f"  - This indicates legal/formal writing style\n")
        f.write(f"  - Range: {stats['passive_voice_ratio']['min']:.2%} to {stats['passive_voice_ratio']['max']:.2%}\n\n")

        f.write("Pronoun Usage (Average per case):\n")
        f.write(f"  First-person (I, we):   {stats['first_person_pronouns']['mean']:.1f}\n")
        f.write(f"  Third-person (he, she): {stats['third_person_pronouns']['mean']:.1f}\n")
        f.write(f"  Ratio (1st:3rd):        1:{stats['third_person_pronouns']['mean']/max(stats['first_person_pronouns']['mean'], 1):.1f}\n\n")

        f.write("Actor Mentions (Average per case):\n")
        f.write(f"  Victim language:       {stats['victim_language_count']['mean']:.1f} instances\n")
        f.write(f"  Perpetrator mentions:  {stats['perpetrator_mentions']['mean']:.1f} instances\n")
        f.write(f"  Victim:Perpetrator:    {stats['victim_language_count']['mean']:.1f}:{stats['perpetrator_mentions']['mean']:.1f}\n\n")

        # 4. ARTICLE-SPECIFIC PATTERNS
        f.write("=" * 80 + "\n")
        f.write("4. SENTIMENT PATTERNS BY VIOLATED ARTICLE\n")
        f.write("=" * 80 + "\n\n")

        # Sort articles by emotional intensity
        sorted_articles = sorted(article_stats.items(), key=lambda x: x[1]['avg_emotional_intensity'], reverse=True)

        f.write(f"{'Article':<12} {'Cases':<8} {'Emot.Int.':<12} {'Fear':<10} {'Anger':<10} {'Sadness':<10} {'Passive%':<10}\n")
        f.write("-" * 80 + "\n")

        for article, stats_dict in sorted_articles:
            f.write(f"{article:<12} {stats_dict['count']:<8} "
                   f"{stats_dict['avg_emotional_intensity']:<12.2%} "
                   f"{stats_dict['avg_fear']:<10.2%} "
                   f"{stats_dict['avg_anger']:<10.2%} "
                   f"{stats_dict['avg_sadness']:<10.2%} "
                   f"{stats_dict['avg_passive_voice']:<10.2%}\n")

        f.write("\n")
        f.write("Key Insights:\n")
        highest_emotion_article = sorted_articles[0]
        lowest_emotion_article = sorted_articles[-1]
        f.write(f"  - Highest emotional intensity: Article {highest_emotion_article[0]} ({highest_emotion_article[1]['avg_emotional_intensity']:.2%})\n")
        f.write(f"  - Lowest emotional intensity:  Article {lowest_emotion_article[0]} ({lowest_emotion_article[1]['avg_emotional_intensity']:.2%})\n\n")

        # Find article with most fear
        fear_sorted = sorted(article_stats.items(), key=lambda x: x[1]['avg_fear'], reverse=True)
        f.write(f"  - Highest fear:  Article {fear_sorted[0][0]} ({fear_sorted[0][1]['avg_fear']:.2%})\n")

        # Find article with most anger
        anger_sorted = sorted(article_stats.items(), key=lambda x: x[1]['avg_anger'], reverse=True)
        f.write(f"  - Highest anger: Article {anger_sorted[0][0]} ({anger_sorted[0][1]['avg_anger']:.2%})\n\n")

        # 5. CORRELATIONS
        f.write("=" * 80 + "\n")
        f.write("5. CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Pearson Correlation Coefficients:\n\n")
        for corr_name, corr_value in correlations.items():
            corr_display = corr_name.replace('_', ' ').title()
            strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
            direction = "Positive" if corr_value > 0 else "Negative"
            f.write(f"  {corr_display}:\n")
            f.write(f"    r = {corr_value:+.4f} ({strength} {direction} correlation)\n\n")

        f.write("Interpretation:\n")
        if correlations['emotional_intensity_vs_passive_voice'] < 0:
            f.write("  - Cases with higher emotional intensity tend to use LESS passive voice\n")
            f.write("    (more direct, active descriptions of emotional events)\n\n")

        if correlations['victim_language_vs_perpetrator_mentions'] > 0.3:
            f.write("  - Cases mentioning perpetrators frequently also use victim language\n")
            f.write("    (indicating narratives focused on harm and agency)\n\n")

        # 6. OUTLIERS & EXTREME CASES
        f.write("=" * 80 + "\n")
        f.write("6. EXTREME CASES (POTENTIAL RETRIEVAL BIAS INDICATORS)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Top 10 Most Emotionally Intense Cases:\n")
        f.write("-" * 80 + "\n")
        for i, case in enumerate(outliers['highest_intensity'][:10], 1):
            f.write(f"{i:2}. {case['case_id']:<15} Intensity: {case['nrc_emotional_intensity']:.2%}  ")
            f.write(f"Fear: {case['nrc_fear']:.2%}  Articles: {', '.join(case['violated_articles'])}\n")

        f.write("\nTop 10 Highest Fear Cases:\n")
        f.write("-" * 80 + "\n")
        for i, case in enumerate(outliers['highest_fear'][:10], 1):
            f.write(f"{i:2}. {case['case_id']:<15} Fear: {case['nrc_fear']:.2%}  ")
            f.write(f"Intensity: {case['nrc_emotional_intensity']:.2%}  Articles: {', '.join(case['violated_articles'])}\n")

        f.write("\nTop 10 Highest Passive Voice Cases:\n")
        f.write("-" * 80 + "\n")
        for i, case in enumerate(outliers['highest_passive_voice'][:10], 1):
            f.write(f"{i:2}. {case['case_id']:<15} Passive: {case['passive_voice_ratio']:.2%}  ")
            f.write(f"Intensity: {case['nrc_emotional_intensity']:.2%}  Articles: {', '.join(case['violated_articles'])}\n")

        # 7. BIAS IMPLICATIONS
        f.write("\n" + "=" * 80 + "\n")
        f.write("7. POTENTIAL RETRIEVAL BIAS IMPLICATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Emotion-Based Retrieval Risks:\n\n")

        f.write("1. HIGH EMOTIONAL INTENSITY BIAS:\n")
        f.write(f"   - Cases in top 10% intensity: >={stats['nrc_emotional_intensity']['q75']:.2%}\n")
        f.write("   - Risk: Embeddings may prioritize emotionally charged language\n")
        f.write("   - Impact: Neutral but legally important cases may rank lower\n\n")

        f.write("2. PASSIVE VOICE VARIATION:\n")
        f.write(f"   - Range: {stats['passive_voice_ratio']['min']:.2%} to {stats['passive_voice_ratio']['max']:.2%}\n")
        f.write("   - Risk: High passive voice = more formal, potentially lower similarity\n")
        f.write("   - Impact: Procedural cases may be underretrieved vs victim narratives\n\n")

        f.write("3. VICTIM/PERPETRATOR LANGUAGE ASYMMETRY:\n")
        victim_perp_ratio = stats['victim_language_count']['mean'] / stats['perpetrator_mentions']['mean']
        f.write(f"   - Victim language is {victim_perp_ratio:.2f}x more frequent than perpetrator mentions\n")
        f.write("   - Risk: Queries using perpetrator framing may retrieve fewer results\n")
        f.write("   - Impact: Asymmetric representation in retrieval results\n\n")

        # 8. RECOMMENDATIONS
        f.write("=" * 80 + "\n")
        f.write("8. RECOMMENDATIONS FOR BIAS MITIGATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. RETRIEVAL EVALUATION:\n")
        f.write("   - Test queries across emotional intensity spectrum\n")
        f.write("   - Compare retrieval results for high vs low emotion cases\n")
        f.write("   - Measure if emotional intensity correlates with ranking\n\n")

        f.write("2. QUERY FORMULATION:\n")
        f.write("   - Use mix of emotional and neutral terminology\n")
        f.write("   - Test both active and passive voice queries\n")
        f.write("   - Balance victim and perpetrator language in test sets\n\n")

        f.write("3. METADATA-AWARE RERANKING:\n")
        f.write("   - Consider normalizing by emotional intensity\n")
        f.write("   - Weight passive voice cases appropriately\n")
        f.write("   - Ensure diverse emotional profiles in top-k results\n\n")

        f.write("4. BIAS METRICS TO TRACK:\n")
        f.write("   - Emotional intensity distribution in retrieved results\n")
        f.write("   - Passive voice ratio in top-k vs dataset average\n")
        f.write("   - Article distribution correlation with emotion scores\n\n")

        # 9. SUMMARY STATISTICS
        f.write("=" * 80 + "\n")
        f.write("9. SUMMARY KEY METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset Size:              {len([s for s in stats.values()][0] if stats else 0)} cases\n")
        f.write(f"Avg Emotional Intensity:   {stats['nrc_emotional_intensity']['mean']:.2%}\n")
        f.write(f"Avg Case Length:           {stats['word_count']['mean']:.0f} words\n")
        f.write(f"Avg Passive Voice:         {stats['passive_voice_ratio']['mean']:.2%}\n")
        f.write(f"Dominant Emotion:          {dominant_emotion}\n")
        f.write(f"Victim:Perpetrator Ratio:  {victim_perp_ratio:.2f}:1\n")
        f.write(f"Most Emotional Article:    {highest_emotion_article[0]}\n")
        f.write(f"Least Emotional Article:   {lowest_emotion_article[0]}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main():

    dataset_path = "dataset/train_with_metadata.jsonl"
    output_path = "SENTIMENT_ANALYSIS_REPORT.txt"

    print("\n" + "=" * 80)
    print(" " * 25 + "SENTIMENT METADATA ANALYZER")
    print("=" * 80 + "\n")

    # Load data
    cases, sentiment_data = load_sentiment_data(dataset_path)

    # Compute statistics
    print("Computing basic statistics...")
    stats = compute_basic_statistics(sentiment_data)

    print("Analyzing by article...")
    article_stats = analyze_by_article(sentiment_data)

    print("Identifying outliers...")
    outliers = identify_outliers(sentiment_data)

    print("Computing correlations...")
    correlations = compute_correlations(sentiment_data)

    # Generate report
    print(f"\nGenerating report: {output_path}")
    generate_report(stats, article_stats, outliers, correlations, output_path)

    print(f"\n✅ Report generated successfully!")
    print(f"📄 Output: {output_path}\n")

    # Print quick summary
    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Cases analyzed:          {len(sentiment_data)}")
    print(f"Avg emotional intensity: {stats['nrc_emotional_intensity']['mean']:.2%}")
    print(f"Avg passive voice:       {stats['passive_voice_ratio']['mean']:.2%}")
    print(f"Articles analyzed:       {len(article_stats)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
