import json

from src.chunking.legal_chunker import LegalCaseChunker

DATASET_PATH = "dataset/train.jsonl"

chunker = LegalCaseChunker()

cases_with_gaps = []
total_cases = 0
total_source_paragraphs = 0
total_covered_paragraphs = 0

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        case = json.loads(line)
        total_cases += 1
        facts = case.get("facts", [])
        num_paragraphs = len(facts)
        total_source_paragraphs += num_paragraphs

        chunks = chunker.create_paragraph_chunks(case)
        covered_indices = set()
        for chunk in chunks:
            covered_indices.update(chunk["paragraph_indices"])

        missing = set(range(num_paragraphs)) - covered_indices
        total_covered_paragraphs += num_paragraphs - len(missing)

        if missing:
            cases_with_gaps.append(
                {
                    "case_id": case["case_id"],
                    "num_paragraphs": num_paragraphs,
                    "missing_indices": sorted(missing),
                    "missing_words": sum(
                        len(facts[i].split()) for i in missing
                    ),
                }
            )

        if total_cases % 1000 == 0:
            print(f"Checked {total_cases} cases...")

print()
print("=" * 60)
print("CHUNK COVERAGE VERIFICATION")
print("=" * 60)
print(f"Total cases: {total_cases}")
print(f"Total source paragraphs: {total_source_paragraphs}")
print(f"Covered paragraphs: {total_covered_paragraphs}")
print(f"Missing paragraphs: {total_source_paragraphs - total_covered_paragraphs}")
print(f"Cases with any gap: {len(cases_with_gaps)}")

if cases_with_gaps:
    print("\nFirst 10 cases with gaps:")
    for c in cases_with_gaps[:10]:
        print(f"  {c['case_id']}: missing paragraph indices {c['missing_indices']} "
              f"({c['missing_words']} words) out of {c['num_paragraphs']} paragraphs")
else:
    print("\nNo gaps found — every source paragraph is covered by at least one chunk.")
