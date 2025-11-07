import json
import os


def load_classifications(results_dir):
    classifications = {}

    # Find all result files
    result_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('gender_classifications_results_') and f.endswith('.jsonl')
    ])

    print(f"Loading classifications from {len(result_files)} files...")

    total_loaded = 0
    duplicates = []

    for result_file in result_files:
        file_path = os.path.join(results_dir, result_file)
        print(f"  Reading: {result_file}")

        file_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line)

                    case_id = result.get('case_id')
                    case_no = result.get('case_no')
                    classification = result.get('classification', {})

                    if not case_id or not case_no:
                        print(f"    Warning: Missing case_id or case_no on line {line_num}")
                        continue

                    # Create composite key
                    key = (case_id, case_no)

                    # Check for duplicates
                    if key in classifications:
                        duplicates.append({
                            'case_id': case_id,
                            'case_no': case_no,
                            'file': result_file,
                            'line': line_num
                        })

                    classifications[key] = classification
                    file_count += 1
                    total_loaded += 1

                except json.JSONDecodeError as e:
                    print(f"    Error: JSON decode error on line {line_num}: {e}")
                except Exception as e:
                    print(f"    Error on line {line_num}: {e}")

        print(f"    Loaded {file_count} classifications")

    print(f"\nTotal classifications loaded: {total_loaded}")

    if duplicates:
        print(f"\n⚠ WARNING: Found {len(duplicates)} duplicate case_id/case_no combinations!")
        print("First 5 duplicates:")
        for dup in duplicates[:5]:
            print(f"  - {dup['case_id']} / {dup['case_no']} in {dup['file']} line {dup['line']}")

    return classifications, duplicates


def merge_with_train(train_file, classifications, output_file):
    print(f"\nMerging classifications with {train_file}...")

    merged_count = 0
    no_classification_found = []
    total_cases = 0

    with open(train_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                case = json.loads(line)
                total_cases += 1

                case_id = case.get('case_id')
                case_no = case.get('case_no')

                if not case_id or not case_no:
                    print(f"  Warning: Missing case_id or case_no on line {line_num} in train.jsonl")
                    no_classification_found.append({
                        'case_id': case_id or 'MISSING',
                        'case_no': case_no or 'MISSING',
                        'line': line_num,
                        'reason': 'Missing case_id or case_no in original file'
                    })
                    continue

                # Create composite key
                key = (case_id, case_no)

                # Find classification
                if key in classifications:
                    case['classification'] = classifications[key]
                    merged_count += 1
                else:
                    no_classification_found.append({
                        'case_id': case_id,
                        'case_no': case_no,
                        'line': line_num,
                        'reason': 'No matching classification found'
                    })
                    print(f"  ⚠ No classification found for: {case_id} / {case_no} (line {line_num})")

                # Write merged case
                outfile.write(json.dumps(case, ensure_ascii=False) + '\n')

                # Progress indicator
                if total_cases % 1000 == 0:
                    print(f"  Processed {total_cases} cases...")

            except json.JSONDecodeError as e:
                print(f"  Error: JSON decode error on line {line_num}: {e}")
            except Exception as e:
                print(f"  Error on line {line_num}: {e}")

    return merged_count, no_classification_found, total_cases


def check_classifications_not_in_train(train_file, classifications):
    print("\nChecking for classifications without matching train cases...")

    # Load all case_ids and case_nos from train.jsonl
    train_keys = set()

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                case = json.loads(line)
                case_id = case.get('case_id')
                case_no = case.get('case_no')
                if case_id and case_no:
                    train_keys.add((case_id, case_no))
            except:
                continue

    # Find classifications not in train
    classification_keys = set(classifications.keys())
    orphaned_classifications = classification_keys - train_keys

    return orphaned_classifications


def generate_validation_report(output_file, merged_count, total_cases,
                               no_classification_found, duplicates,
                               orphaned_classifications):
    report_file = output_file.replace('.jsonl', '_validation_report.txt')

    print(f"\nGenerating validation report: {report_file}")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MERGE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input file: train.jsonl\n")
        f.write(f"Output file: {os.path.basename(output_file)}\n")
        f.write(f"Date: 2025-11-03\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total cases in train.jsonl: {total_cases:,}\n")
        f.write(f"Successfully merged with classification: {merged_count:,}\n")
        f.write(f"Cases without classification: {len(no_classification_found):,}\n")
        f.write(f"Merge success rate: {(merged_count/total_cases*100) if total_cases > 0 else 0:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("ISSUES DETECTED\n")
        f.write("=" * 80 + "\n\n")

        # Duplicates in classifications
        f.write(f"1. Duplicate classifications: {len(duplicates):,}\n")
        if duplicates:
            f.write("   ⚠ WARNING: Duplicate case_id/case_no combinations found!\n\n")
            f.write("   First 10 duplicates:\n")
            for i, dup in enumerate(duplicates[:10], 1):
                f.write(f"   {i}. {dup['case_id']} / {dup['case_no']} in {dup['file']} line {dup['line']}\n")
            if len(duplicates) > 10:
                f.write(f"   ... and {len(duplicates) - 10} more\n")
        else:
            f.write("   ✓ No duplicates found\n")
        f.write("\n")

        # Cases without classification
        f.write(f"2. Cases in train.jsonl without classification: {len(no_classification_found):,}\n")
        if no_classification_found:
            f.write("   ⚠ WARNING: Some cases could not be matched!\n\n")
            f.write("   Details:\n")
            for i, missing in enumerate(no_classification_found[:20], 1):
                f.write(f"   {i}. case_id={missing['case_id']}, case_no={missing['case_no']}, ")
                f.write(f"line={missing['line']}, reason={missing['reason']}\n")
            if len(no_classification_found) > 20:
                f.write(f"   ... and {len(no_classification_found) - 20} more\n")
        else:
            f.write("   ✓ All cases successfully matched\n")
        f.write("\n")

        # Orphaned classifications
        f.write(f"3. Classifications without matching train case: {len(orphaned_classifications):,}\n")
        if orphaned_classifications:
            f.write("   ⚠ WARNING: Some classifications don't have matching train cases!\n\n")
            f.write("   First 20 orphaned classifications:\n")
            for i, (case_id, case_no) in enumerate(sorted(orphaned_classifications)[:20], 1):
                f.write(f"   {i}. case_id={case_id}, case_no={case_no}\n")
            if len(orphaned_classifications) > 20:
                f.write(f"   ... and {len(orphaned_classifications) - 20} more\n")
        else:
            f.write("   ✓ All classifications have matching train cases\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        if (len(no_classification_found) == 0 and
            len(duplicates) == 0 and
            len(orphaned_classifications) == 0):
            f.write("✓ PERFECT MERGE: All validations passed!\n")
            f.write(f"  - {merged_count:,} cases successfully merged\n")
            f.write("  - No mismatches, duplicates, or orphaned classifications\n")
        else:
            f.write("⚠ MERGE COMPLETED WITH ISSUES:\n")
            if len(no_classification_found) > 0:
                f.write(f"  - {len(no_classification_found):,} cases without classification\n")
            if len(duplicates) > 0:
                f.write(f"  - {len(duplicates):,} duplicate classifications\n")
            if len(orphaned_classifications) > 0:
                f.write(f"  - {len(orphaned_classifications):,} orphaned classifications\n")
            f.write("\n  Please review the issues above.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Validation report saved to: {report_file}")


def main():
    # File paths - relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, "Metadata Extraction Files", "gender_classification_JSNOL_results")
    train_file = os.path.join(project_root, "dataset", "train.jsonl")
    output_file = os.path.join(project_root, "dataset", "train_with_gender.jsonl")

    print("=" * 80)
    print("GENDER CLASSIFICATION MERGE")
    print("=" * 80)
    print()

    # Step 1: Load classifications
    classifications, duplicates = load_classifications(results_dir)

    # Step 2: Merge with train.jsonl
    merged_count, no_classification_found, total_cases = merge_with_train(
        train_file, classifications, output_file
    )

    # Step 3: Check for orphaned classifications
    orphaned_classifications = check_classifications_not_in_train(train_file, classifications)

    # Step 4: Generate validation report
    generate_validation_report(
        output_file, merged_count, total_cases, no_classification_found,
        duplicates, orphaned_classifications
    )

    # Final summary
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"\nTotal cases processed: {total_cases:,}")
    print(f"Successfully merged: {merged_count:,} ({merged_count/total_cases*100:.2f}%)")
    print(f"Missing classifications: {len(no_classification_found):,}")
    print(f"\nOutput file: {output_file}")

    if len(no_classification_found) == 0 and len(duplicates) == 0 and len(orphaned_classifications) == 0:
        print("\n✓ SUCCESS: All cases merged successfully with no issues!")
    else:
        print("\n⚠ WARNING: Some issues detected. See validation report for details.")
        if len(no_classification_found) > 0:
            print(f"  - {len(no_classification_found):,} cases without classification")
        if len(duplicates) > 0:
            print(f"  - {len(duplicates):,} duplicate classifications")
        if len(orphaned_classifications) > 0:
            print(f"  - {len(orphaned_classifications):,} orphaned classifications")

    print("=" * 80)


if __name__ == "__main__":
    main()
