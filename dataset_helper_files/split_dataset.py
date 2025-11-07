#!/usr/bin/env python3
"""
Split train.jsonl into smaller chunks for processing with Claude.
Each chunk will contain approximately equal numbers of cases.
"""

import json
import os
from pathlib import Path

def split_jsonl(input_file, output_dir, num_splits=10):
    """
    Split a JSONL file into multiple smaller files.

    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save the split files
        num_splits: Number of files to split into
    """
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    lines_per_file = total_lines // num_splits

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total cases: {total_lines}")
    print(f"Cases per file: {lines_per_file}")
    print(f"Splitting into {num_splits} files...\n")

    # Split and write files
    for i in range(num_splits):
        start_idx = i * lines_per_file
        # For the last file, include any remaining lines
        if i == num_splits - 1:
            end_idx = total_lines
        else:
            end_idx = (i + 1) * lines_per_file

        chunk_lines = lines[start_idx:end_idx]

        # Create output filename with zero-padded numbers
        output_file = os.path.join(output_dir, f"train_split_{i+1:02d}.jsonl")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(chunk_lines)

        # Verify the first case_id in each file
        first_case = json.loads(chunk_lines[0])
        print(f"Created {output_file}: {len(chunk_lines)} cases (first: {first_case['case_id']})")

    print(f"\nSuccessfully split {input_file} into {num_splits} files in {output_dir}/")

if __name__ == "__main__":
    # File paths - relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_file = os.path.join(project_root, "dataset", "train.jsonl")
    output_dir = os.path.join(project_root, "dataset", "train_splits")

    split_jsonl(input_file, output_dir, num_splits=10)
