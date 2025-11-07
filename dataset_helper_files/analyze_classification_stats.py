import json
import os



def analyze_classification_files(results_dir: str):
    # Initialize counters
    total_cases = 0

    # Classification counts
    classifications = {
        "Male": {"total": 0, "High": 0, "Medium": 0, "Low": 0, "N/A": 0},
        "Female": {"total": 0, "High": 0, "Medium": 0, "Low": 0, "N/A": 0},
        "Unknown": {"total": 0, "High": 0, "Medium": 0, "Low": 0, "N/A": 0},
        "Multiple Applicants": {"total": 0, "High": 0, "Medium": 0, "Low": 0, "N/A": 0},
        "Needs Manual Classification": {"total": 0, "High": 0, "Medium": 0, "Low": 0, "N/A": 0}
    }

    # Find all result files
    result_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('gender_classifications_results_') and f.endswith('.jsonl')
    ])

    print(f"Found {len(result_files)} result files to process")
    print("=" * 70)

    # Process each file
    for result_file in result_files:
        file_path = os.path.join(results_dir, result_file)
        print(f"\nProcessing: {result_file}")

        file_cases = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line)

                    # Extract classification and confidence
                    classification = result.get('classification', {})
                    gender = classification.get('gender', 'Unknown')
                    confidence = classification.get('confidence', 'N/A')

                    # Update counters
                    if gender in classifications:
                        classifications[gender]["total"] += 1
                        if confidence in classifications[gender]:
                            classifications[gender][confidence] += 1
                        else:
                            classifications[gender]["N/A"] += 1

                    file_cases += 1
                    total_cases += 1

                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON error on line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error on line {line_num}: {e}")

        print(f"  Processed {file_cases} cases from this file")

    print("\n" + "=" * 70)
    print(f"TOTAL CASES PROCESSED: {total_cases}")
    print("=" * 70)

    # Print detailed statistics
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION STATISTICS")
    print("=" * 70)

    for category in ["Male", "Female", "Unknown", "Multiple Applicants", "Needs Manual Classification"]:
        stats = classifications[category]
        total = stats["total"]

        if total == 0:
            continue

        print(f"\n{category.upper()}")
        print("-" * 70)
        print(f"  Total: {total:,} cases ({total/total_cases*100:.2f}% of all cases)")

        if stats["High"] > 0:
            print(f"    High Confidence:   {stats['High']:,} ({stats['High']/total*100:.2f}% of {category})")
        if stats["Medium"] > 0:
            print(f"    Medium Confidence: {stats['Medium']:,} ({stats['Medium']/total*100:.2f}% of {category})")
        if stats["Low"] > 0:
            print(f"    Low Confidence:    {stats['Low']:,} ({stats['Low']/total*100:.2f}% of {category})")
        if stats["N/A"] > 0:
            print(f"    N/A Confidence:    {stats['N/A']:,} ({stats['N/A']/total*100:.2f}% of {category})")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Category':<30} {'Total':>10} {'% of All':>10} {'High':>10} {'Medium':>10}")
    print("-" * 70)

    for category in ["Male", "Female", "Unknown", "Multiple Applicants", "Needs Manual Classification"]:
        stats = classifications[category]
        total = stats["total"]
        pct_all = (total/total_cases*100) if total_cases > 0 else 0
        high = stats["High"]
        medium = stats["Medium"]

        print(f"{category:<30} {total:>10,} {pct_all:>9.2f}% {high:>10,} {medium:>10,}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {total_cases:>10,} {100.0:>9.2f}%")

    # High/Medium confidence breakdown by category
    print("\n" + "=" * 70)
    print("CONFIDENCE LEVEL PERCENTAGES (within each category)")
    print("=" * 70)
    print(f"{'Category':<30} {'High %':>12} {'Medium %':>12}")
    print("-" * 70)

    for category in ["Male", "Female", "Unknown", "Multiple Applicants", "Needs Manual Classification"]:
        stats = classifications[category]
        total = stats["total"]

        if total > 0:
            high_pct = (stats["High"]/total*100)
            medium_pct = (stats["Medium"]/total*100)
            print(f"{category:<30} {high_pct:>11.2f}% {medium_pct:>11.2f}%")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    male_total = classifications["Male"]["total"]
    female_total = classifications["Female"]["total"]
    unknown_total = classifications["Unknown"]["total"]
    multiple_total = classifications["Multiple Applicants"]["total"]
    needs_manual_total = classifications["Needs Manual Classification"]["total"]

    gendered_total = male_total + female_total
    high_confidence_gendered = classifications["Male"]["High"] + classifications["Female"]["High"]

    print(f"\n1. Gender Distribution:")
    print(f"   - Male cases: {male_total:,} ({male_total/total_cases*100:.2f}%)")
    print(f"   - Female cases: {female_total:,} ({female_total/total_cases*100:.2f}%)")
    print(f"   - Male/Female ratio: {male_total/female_total:.2f}:1" if female_total > 0 else "   - Male/Female ratio: N/A")

    print(f"\n2. Classification Certainty:")
    print(f"   - Cases with clear gender (Male or Female): {gendered_total:,} ({gendered_total/total_cases*100:.2f}%)")
    print(f"   - High confidence gendered cases: {high_confidence_gendered:,} ({high_confidence_gendered/gendered_total*100:.2f}% of gendered)" if gendered_total > 0 else "")
    print(f"   - Unknown: {unknown_total:,} ({unknown_total/total_cases*100:.2f}%)")
    print(f"   - Needs Manual Classification: {needs_manual_total:,} ({needs_manual_total/total_cases*100:.2f}%)")
    print(f"   - Multiple Applicants: {multiple_total:,} ({multiple_total/total_cases*100:.2f}%)")

    print(f"\n3. Data Quality:")
    uncertain = unknown_total + needs_manual_total
    print(f"   - Clear single-gender cases: {gendered_total:,} ({gendered_total/total_cases*100:.2f}%)")
    print(f"   - Uncertain/ambiguous cases: {uncertain:,} ({uncertain/total_cases*100:.2f}%)")
    print(f"   - Multiple applicant cases: {multiple_total:,} ({multiple_total/total_cases*100:.2f}%)")

    print("\n" + "=" * 70)

    return classifications, total_cases


if __name__ == "__main__":
    # File paths - relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "Metadata Extraction Files", "gender_classification_JSNOL_results")

    print("GENDER CLASSIFICATION ANALYSIS")
    print("=" * 70)
    print(f"Results Directory: {results_dir}\n")

    classifications, total = analyze_classification_files(results_dir)

    print(f"\nAnalysis complete. Total cases analyzed: {total:,}")

    if total != 9000:
        print(f"\n⚠ WARNING: Expected 9000 cases but found {total} cases!")
    else:
        print(f"\n✓ Successfully analyzed all 9000 cases!")
