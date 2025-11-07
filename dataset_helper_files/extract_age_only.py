import json
import os
import re
from datetime import datetime
from collections import Counter
from pathlib import Path


class AgeExtractor:
    """Extract age information from case facts."""

    def __init__(self):
        # Compile regex patterns for better performance
        # Birth year patterns
        self.birth_year_patterns = [
            re.compile(r'(?:applicant|he|she).*?born\s+(?:in|on)\s+.*?(\d{4})', re.IGNORECASE),
            re.compile(r'born\s+(?:in|on)\s+.*?(\d{4})', re.IGNORECASE),
            re.compile(r'(?:applicant|he|she).*?birth.*?(\d{4})', re.IGNORECASE),
        ]

        # Age patterns (direct age statement)
        self.age_patterns = [
            re.compile(r'(?:applicant|he|she).*?aged?\s+(\d+)', re.IGNORECASE),
            re.compile(r'at\s+the\s+age\s+of\s+(\d+)', re.IGNORECASE),
            re.compile(r'(?:applicant|he|she)\s+(?:is|was)\s+(\d+)\s+years\s+old', re.IGNORECASE),
        ]

    def extract_birth_year(self, facts, case_id):
        """
        Extract birth year from facts.
        Returns: int or None
        """
        # Check first 10 facts (usually biographical info is early)
        for fact in facts[:10]:
            # Skip very long facts (likely procedural)
            if len(fact) > 2000:
                continue

            for pattern in self.birth_year_patterns:
                matches = pattern.findall(fact)
                for match in matches:
                    try:
                        year = int(match)
                        # Sanity check: reasonable birth year range
                        if 1850 <= year <= 2020:
                            return year
                    except ValueError:
                        continue

        return None

    def extract_direct_age(self, facts, case_id):
        for fact in facts[:10]:
            if len(fact) > 2000:
                continue

            for pattern in self.age_patterns:
                matches = pattern.findall(fact)
                for match in matches:
                    try:
                        age = int(match)
                        # Sanity check: reasonable age range
                        if 0 <= age <= 120:
                            return age
                    except ValueError:
                        continue

        return None

    def calculate_age_at_judgment(self, birth_year, judgment_date):
        if birth_year is None or judgment_date is None:
            return None

        try:
            judgment_year = datetime.strptime(judgment_date, '%Y-%m-%d').year
            age = judgment_year - birth_year

            # Sanity check
            if 0 <= age <= 120:
                return age
        except (ValueError, TypeError):
            pass

        return None


def process_dataset(input_file, output_file):
    extractor = AgeExtractor()

    stats = {
        'total_processed': 0,
        'male_cases': 0,
        'female_cases': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'birth_years_found': 0,
        'direct_ages_found': 0,
        'calculated_ages': 0,
        'total_with_age': 0,
        'age_distribution': Counter(),
    }

    results = []

    print(f"Processing {input_file}...")
    print("=" * 80)

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                case = json.loads(line)

                # Extract metadata
                case_id = case.get('case_id')
                case_no = case.get('case_no')
                classification = case.get('classification', {})
                gender = classification.get('gender')
                confidence = classification.get('confidence')
                facts = case.get('facts', [])
                judgment_date = case.get('judgment_date')

                # Only process Male/Female cases (skip Unknown, Multiple Applicants, etc.)
                if gender not in ['Male', 'Female']:
                    continue

                # Only process High/Medium confidence cases
                if confidence not in ['High', 'Medium']:
                    continue

                stats['total_processed'] += 1
                if gender == 'Male':
                    stats['male_cases'] += 1
                else:
                    stats['female_cases'] += 1

                if confidence == 'High':
                    stats['high_confidence'] += 1
                else:
                    stats['medium_confidence'] += 1

                # Extract age information
                birth_year = extractor.extract_birth_year(facts, case_id)
                direct_age = extractor.extract_direct_age(facts, case_id)

                # Calculate age at judgment if birth year found
                calculated_age = extractor.calculate_age_at_judgment(birth_year, judgment_date)

                # Determine final age (prefer calculated over direct)
                final_age = calculated_age if calculated_age is not None else direct_age

                # Update statistics
                if birth_year:
                    stats['birth_years_found'] += 1
                if direct_age:
                    stats['direct_ages_found'] += 1
                if calculated_age:
                    stats['calculated_ages'] += 1
                if final_age:
                    stats['total_with_age'] += 1
                    # Group by decade for distribution
                    decade = (final_age // 10) * 10
                    stats['age_distribution'][f"{decade}s"] += 1

                # Create result entry
                result = {
                    'case_id': case_id,
                    'case_no': case_no,
                    'age_info': {
                        'birth_year': birth_year,
                        'age_at_judgment': final_age,
                        'age_source': 'calculated' if calculated_age else ('direct' if direct_age else None)
                    }
                }

                results.append(result)

                # Progress indicator
                if stats['total_processed'] % 500 == 0:
                    print(f"  Processed {stats['total_processed']} Male/Female cases...")

            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error on line {line_num}: {e}")
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")

    # Write results to JSONL
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

    return stats, results


def print_statistics(stats):
    """Print comprehensive statistics."""

    print("\n" + "=" * 80)
    print("AGE EXTRACTION STATISTICS")
    print("=" * 80)

    print(f"\nCases Processed (Male/Female with High/Medium confidence):")
    print(f"  Total: {stats['total_processed']:,}")
    print(f"    Male: {stats['male_cases']:,} ({stats['male_cases']/stats['total_processed']*100:.1f}%)")
    print(f"    Female: {stats['female_cases']:,} ({stats['female_cases']/stats['total_processed']*100:.1f}%)")
    print(f"  High Confidence: {stats['high_confidence']:,} ({stats['high_confidence']/stats['total_processed']*100:.1f}%)")
    print(f"  Medium Confidence: {stats['medium_confidence']:,} ({stats['medium_confidence']/stats['total_processed']*100:.1f}%)")

    print(f"\nAge Extraction Results:")
    print(f"  Birth years found: {stats['birth_years_found']:,} ({stats['birth_years_found']/stats['total_processed']*100:.1f}%)")
    print(f"  Direct ages found: {stats['direct_ages_found']:,} ({stats['direct_ages_found']/stats['total_processed']*100:.1f}%)")
    print(f"  Ages calculated from birth year: {stats['calculated_ages']:,} ({stats['calculated_ages']/stats['total_processed']*100:.1f}%)")
    print(f"  ")
    print(f"  ✓ TOTAL WITH AGE INFO: {stats['total_with_age']:,} ({stats['total_with_age']/stats['total_processed']*100:.1f}%)")
    print(f"  ✗ Cases without age: {stats['total_processed'] - stats['total_with_age']:,} ({(stats['total_processed'] - stats['total_with_age'])/stats['total_processed']*100:.1f}%)")

    print("\n" + "-" * 80)
    print("AGE DISTRIBUTION (by decade at time of judgment)")
    print("-" * 80)
    for age_range in sorted(stats['age_distribution'].keys(), key=lambda x: int(x[:-1])):
        count = stats['age_distribution'][age_range]
        pct = count / stats['total_processed'] * 100
        bar = '█' * int(pct * 2)  # Simple bar chart
        print(f"  {age_range:10s}: {count:4,} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 80)


def generate_summary_report(output_file, stats):
    """Generate a text summary report."""

    report_file = output_file.replace('.jsonl', '_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AGE EXTRACTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input: train_with_gender.jsonl\n")
        f.write(f"Output: {Path(output_file).name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Male/Female cases (High/Medium confidence): {stats['total_processed']:,}\n")
        f.write(f"  Male: {stats['male_cases']:,}\n")
        f.write(f"  Female: {stats['female_cases']:,}\n\n")

        f.write(f"Age information found: {stats['total_with_age']:,} ")
        f.write(f"({stats['total_with_age']/stats['total_processed']*100:.1f}%)\n")
        f.write(f"  Birth years extracted: {stats['birth_years_found']:,}\n")
        f.write(f"  Direct ages extracted: {stats['direct_ages_found']:,}\n")
        f.write(f"  Ages calculated: {stats['calculated_ages']:,}\n\n")

        f.write(f"Cases without age: {stats['total_processed'] - stats['total_with_age']:,} ")
        f.write(f"({(stats['total_processed'] - stats['total_with_age'])/stats['total_processed']*100:.1f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("AGE DISTRIBUTION (at time of judgment)\n")
        f.write("=" * 80 + "\n\n")

        for age_range in sorted(stats['age_distribution'].keys(), key=lambda x: int(x[:-1])):
            count = stats['age_distribution'][age_range]
            pct = count / stats['total_processed'] * 100
            f.write(f"{age_range:10s}: {count:4,} ({pct:5.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nDetailed report saved to: {report_file}")


def main():
    """Main execution."""

    # File paths - relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_file = os.path.join(project_root, "dataset", "train_with_gender.jsonl")
    output_file = os.path.join(project_root, "dataset", "age_extractions.jsonl")

    print("=" * 80)
    print("AGE EXTRACTION")
    print("=" * 80)
    print("\nExtracting age information for Male/Female cases only...")
    print("(High and Medium confidence classifications)\n")

    # Process dataset
    stats, results = process_dataset(input_file, output_file)

    # Print statistics
    print_statistics(stats)

    # Generate report
    generate_summary_report(output_file, stats)

    # Sample output
    print("\n" + "-" * 80)
    print("SAMPLE EXTRACTIONS (first 15 cases with age)")
    print("-" * 80)

    samples = [r for r in results if r['age_info']['age_at_judgment'] is not None][:15]

    for i, sample in enumerate(samples, 1):
        info = sample['age_info']
        print(f"\n{i}. Case: {sample['case_id']}")
        print(f"   Birth year: {info['birth_year']}")
        print(f"   Age at judgment: {info['age_at_judgment']} years ({info['age_source']})")

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")
    print(f"Total entries: {len(results):,}")
    print(f"Entries with age: {stats['total_with_age']:,} ({stats['total_with_age']/len(results)*100:.1f}%)")
    print(f"\nReady for merging with train_with_gender.jsonl using case_id and case_no")
    print("=" * 80)


if __name__ == "__main__":
    main()
