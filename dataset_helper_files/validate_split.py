#!/usr/bin/env python3
"""
Validate that split files contain all cases from the original file
with no duplicates or missing case_ids.
"""

import json
import os
from collections import Counter

def validate_split(original_file, split_dir):
    """
    Validate that split files match the original file exactly.

    Args:
        original_file: Path to the original JSONL file
        split_dir: Directory containing the split files
    """
    print("Reading original file...")
    original_case_ids = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            original_case_ids.append(case['case_id'])

    print(f"Original file: {len(original_case_ids)} cases\n")

    # Read all split files
    print("Reading split files...")
    split_case_ids = []
    split_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.jsonl')])

    for split_file in split_files:
        file_path = os.path.join(split_dir, split_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_cases = []
            for line in f:
                case = json.loads(line)
                file_cases.append(case['case_id'])
                split_case_ids.append(case['case_id'])
            print(f"  {split_file}: {len(file_cases)} cases")

    print(f"\nTotal in split files: {len(split_case_ids)} cases\n")

    # Validation checks
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Check 1: Total count matches
    if len(original_case_ids) == len(split_case_ids):
        print("✓ Total count matches: PASS")
    else:
        print(f"✗ Total count mismatch: FAIL")
        print(f"  Original: {len(original_case_ids)}, Split: {len(split_case_ids)}")

    # Check 2: No duplicates in split files
    split_counter = Counter(split_case_ids)
    duplicates = {case_id: count for case_id, count in split_counter.items() if count > 1}

    if not duplicates:
        print("✓ No duplicates found: PASS")
    else:
        print(f"✗ Duplicates found: FAIL ({len(duplicates)} cases)")
        print("  Sample duplicates:")
        for case_id, count in list(duplicates.items())[:5]:
            print(f"    {case_id}: appears {count} times")

    # Check 3: All original cases present in splits
    original_set = set(original_case_ids)
    split_set = set(split_case_ids)

    missing_cases = original_set - split_set
    if not missing_cases:
        print("✓ All original cases present: PASS")
    else:
        print(f"✗ Missing cases: FAIL ({len(missing_cases)} cases)")
        print("  Sample missing cases:")
        for case_id in list(missing_cases)[:5]:
            print(f"    {case_id}")

    # Check 4: No extra cases in splits
    extra_cases = split_set - original_set
    if not extra_cases:
        print("✓ No extra cases: PASS")
    else:
        print(f"✗ Extra cases found: FAIL ({len(extra_cases)} cases)")
        print("  Sample extra cases:")
        for case_id in list(extra_cases)[:5]:
            print(f"    {case_id}")

    # Check 5: Order preservation (optional - check if first and last match)
    if original_case_ids[0] == split_case_ids[0]:
        print("✓ First case matches: PASS")
    else:
        print(f"✗ First case mismatch: WARNING")
        print(f"  Original: {original_case_ids[0]}, Split: {split_case_ids[0]}")

    if original_case_ids[-1] == split_case_ids[-1]:
        print("✓ Last case matches: PASS")
    else:
        print(f"✗ Last case mismatch: WARNING")
        print(f"  Original: {original_case_ids[-1]}, Split: {split_case_ids[-1]}")

    print("=" * 60)

    # Final verdict
    all_passed = (
        len(original_case_ids) == len(split_case_ids) and
        not duplicates and
        not missing_cases and
        not extra_cases
    )

    if all_passed:
        print("\n✓ ALL VALIDATION CHECKS PASSED!")
        print("The split files are valid and complete.")
    else:
        print("\n✗ VALIDATION FAILED!")
        print("Please check the errors above.")

    return all_passed

if __name__ == "__main__":
    # File paths - relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    original_file = os.path.join(project_root, "dataset", "train.jsonl")
    split_dir = os.path.join(project_root, "dataset", "train_splits")

    validate_split(original_file, split_dir)
