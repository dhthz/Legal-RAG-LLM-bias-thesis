
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class MetadataMerger:

    def __init__(self):
        self.gender_data = {}
        self.age_data = {}
        self.sentiment_data = {}

    def load_gender_data(self, gender_file: str):
        print(f"Loading gender data from: {gender_file}")

        with open(gender_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                case_id = entry['case_id']
                self.gender_data[case_id] = {
                    'gender': entry.get('gender', 'unknown'),
                    'gender_confidence': entry.get('confidence', 'unknown'),
                }

        print(f"Loaded gender data for {len(self.gender_data)} cases")

    def load_age_data(self, age_file: str):
        print(f"Loading age data from: {age_file}")

        with open(age_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                case_id = entry['case_id']
                self.age_data[case_id] = {
                    'age_at_judgment': entry.get('age_at_judgment'),
                    'birth_year': entry.get('birth_year'),
                    'has_age_info': entry.get('has_age_info', False),
                }

        print(f"Loaded age data for {len(self.age_data)} cases")

    def load_sentiment_data(self, sentiment_file: str):
        print(f"Loading sentiment data from: {sentiment_file}")

        # Check if CSV or JSONL
        if sentiment_file.endswith('.csv'):
            import csv
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    case_id = row['case_id']
                    self.sentiment_data[case_id] = {
                        'nrc_fear': float(row.get('nrc_fear', 0)),
                        'nrc_anger': float(row.get('nrc_anger', 0)),
                        'nrc_sadness': float(row.get('nrc_sadness', 0)),
                        'nrc_disgust': float(row.get('nrc_disgust', 0)),
                        'nrc_emotional_intensity': float(row.get('nrc_emotional_intensity', 0)),
                        'passive_voice_ratio': float(row.get('passive_voice_ratio', 0)),
                        'first_person_pronouns': int(row.get('first_person_pronouns', 0)),
                        'third_person_pronouns': int(row.get('third_person_pronouns', 0)),
                        'perpetrator_mentions': int(row.get('perpetrator_mentions', 0)),
                        'victim_language_count': int(row.get('victim_language_count', 0)),
                    }
        else:
            # JSONL format
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    case_id = entry['case_id']
                    sentiment_info = entry.get('sentiment_info', {})
                    self.sentiment_data[case_id] = sentiment_info

        print(f"Loaded sentiment data for {len(self.sentiment_data)} cases")

    def merge_into_chunks(
        self,
        chunk_metadata_file: str,
        output_file: str
    ) -> Dict:
        print(f"\nLoading chunk metadata from: {chunk_metadata_file}")

        with open(chunk_metadata_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks")

        # Statistics
        stats = {
            'total_chunks': len(chunks),
            'chunks_with_gender': 0,
            'chunks_with_age': 0,
            'chunks_with_sentiment': 0,
            'chunks_fully_enriched': 0,
            'missing_gender': [],
            'missing_age': [],
            'missing_sentiment': [],
        }

        # Merge metadata into each chunk
        print("\nMerging metadata...")
        for i, chunk in enumerate(chunks):
            case_id = chunk['case_id']

            # Add gender data
            if case_id in self.gender_data:
                chunk.update(self.gender_data[case_id])
                stats['chunks_with_gender'] += 1
            else:
                if case_id not in stats['missing_gender']:
                    stats['missing_gender'].append(case_id)

            # Add age data
            if case_id in self.age_data:
                chunk.update(self.age_data[case_id])
                stats['chunks_with_age'] += 1
            else:
                if case_id not in stats['missing_age']:
                    stats['missing_age'].append(case_id)

            # Add sentiment data
            if case_id in self.sentiment_data:
                chunk.update(self.sentiment_data[case_id])
                stats['chunks_with_sentiment'] += 1
            else:
                if case_id not in stats['missing_sentiment']:
                    stats['missing_sentiment'].append(case_id)

            # Check if fully enriched
            if (case_id in self.gender_data and
                case_id in self.age_data and
                case_id in self.sentiment_data):
                stats['chunks_fully_enriched'] += 1

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")

        # Save enriched metadata
        print(f"\nSaving enriched metadata to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # Calculate percentages
        stats['gender_coverage_%'] = (stats['chunks_with_gender'] / stats['total_chunks']) * 100
        stats['age_coverage_%'] = (stats['chunks_with_age'] / stats['total_chunks']) * 100
        stats['sentiment_coverage_%'] = (stats['chunks_with_sentiment'] / stats['total_chunks']) * 100
        stats['fully_enriched_%'] = (stats['chunks_fully_enriched'] / stats['total_chunks']) * 100

        # Print summary
        print("\n" + "=" * 60)
        print("METADATA MERGE SUMMARY")
        print("=" * 60)
        print(f"Total chunks: {stats['total_chunks']:,}")
        print(f"\nCoverage:")
        print(f"  Gender:    {stats['chunks_with_gender']:,} ({stats['gender_coverage_%']:.1f}%)")
        print(f"  Age:       {stats['chunks_with_age']:,} ({stats['age_coverage_%']:.1f}%)")
        print(f"  Sentiment: {stats['chunks_with_sentiment']:,} ({stats['sentiment_coverage_%']:.1f}%)")
        print(f"  Fully enriched: {stats['chunks_fully_enriched']:,} ({stats['fully_enriched_%']:.1f}%)")

        if stats['missing_gender']:
            print(f"\n⚠️  Missing gender for {len(stats['missing_gender'])} cases")
        if stats['missing_age']:
            print(f"⚠️  Missing age for {len(stats['missing_age'])} cases")
        if stats['missing_sentiment']:
            print(f"⚠️  Missing sentiment for {len(stats['missing_sentiment'])} cases")

        print("=" * 60)

        return stats

    def verify_enriched_chunks(self, enriched_file: str, sample_size: int = 5):
        print(f"\n" + "=" * 60)
        print(f"VERIFYING ENRICHED CHUNKS (showing {sample_size} samples)")
        print("=" * 60)

        with open(enriched_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Show samples from different cases
        shown_cases = set()
        samples_shown = 0

        for chunk in chunks:
            if chunk['case_id'] not in shown_cases and samples_shown < sample_size:
                print(f"\nChunk: {chunk['chunk_id']}")
                print(f"  Case: {chunk['case_id']}")
                print(f"  Gender: {chunk.get('gender', 'MISSING')}")
                print(f"  Age: {chunk.get('age_at_judgment', 'MISSING')}")
                print(f"  NRC Fear: {chunk.get('nrc_fear', 'MISSING')}")
                print(f"  NRC Emotional Intensity: {chunk.get('nrc_emotional_intensity', 'MISSING')}")
                print(f"  Passive Voice Ratio: {chunk.get('passive_voice_ratio', 'MISSING')}")
                print(f"  Perpetrator Mentions: {chunk.get('perpetrator_mentions', 'MISSING')}")

                shown_cases.add(chunk['case_id'])
                samples_shown += 1

        print("=" * 60)


def main():

    print("=" * 60)
    print("METADATA MERGER FOR CHUNK ENRICHMENT")
    print("=" * 60)

    # File paths - UPDATE THESE TO YOUR ACTUAL FILE PATHS
    GENDER_FILE = "Metadata Extraction Files/gender_classification_results.jsonl"
    AGE_FILE = "Metadata Extraction Files/age_extraction_summary.jsonl"  # or your actual age file
    SENTIMENT_FILE = "Metadata Extraction Files/sentiment_analysis_JSONL_results/sentiment_extractions.jsonl"

    CHUNK_METADATA = "faiss_indices/paragraph_chunks_metadata.json"
    OUTPUT_FILE = "faiss_indices/paragraph_chunks_metadata_enriched.json"

    # Check if files exist
    missing_files = []
    for file_path in [GENDER_FILE, AGE_FILE, SENTIMENT_FILE, CHUNK_METADATA]:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("\n⚠️  ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease update the file paths in this script and re-run.")
        return

    # Initialize merger
    merger = MetadataMerger()

    # Load all metadata sources
    try:
        merger.load_gender_data(GENDER_FILE)
    except Exception as e:
        print(f"⚠️  Warning: Could not load gender data: {e}")

    try:
        merger.load_age_data(AGE_FILE)
    except Exception as e:
        print(f"⚠️  Warning: Could not load age data: {e}")

    try:
        merger.load_sentiment_data(SENTIMENT_FILE)
    except Exception as e:
        print(f"⚠️  Warning: Could not load sentiment data: {e}")

    # Merge into chunks
    stats = merger.merge_into_chunks(CHUNK_METADATA, OUTPUT_FILE)

    # Verify
    merger.verify_enriched_chunks(OUTPUT_FILE, sample_size=5)

    # Save statistics
    stats_file = "metadata_merge_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    print("\n" + "=" * 60)
    print("✓ METADATA MERGE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
