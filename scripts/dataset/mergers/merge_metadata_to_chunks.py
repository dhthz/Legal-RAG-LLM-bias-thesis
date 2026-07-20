
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class MetadataMerger:

    def __init__(self):
        self.gender_data = {}
        self.age_data = {}
        self.sentiment_data = {}

    def load_metadata(self, metadata_file: str):
        print(f"Loading case metadata from: {metadata_file}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                case_id = entry['case_id']

                classification = entry.get('classification', {})
                self.gender_data[case_id] = {
                    'gender': classification.get('gender', 'unknown'),
                    'gender_confidence': classification.get('confidence', 'unknown'),
                }

                age_info = entry.get('age_info', {})
                self.age_data[case_id] = {
                    'age_at_judgment': age_info.get('age_at_judgment'),
                    'birth_year': age_info.get('birth_year'),
                    'has_age_info': bool(age_info),
                }

                self.sentiment_data[case_id] = entry.get('sentiment_info', {})

        print(f"Loaded gender data for {len(self.gender_data)} cases")
        print(f"Loaded age data for {len(self.age_data)} cases")
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

    METADATA_FILE = "dataset/train_with_metadata.jsonl"

    CHUNK_METADATA = "faiss_indices/paragraph_chunks_metadata.json"
    OUTPUT_FILE = "faiss_indices/paragraph_chunks_metadata_enriched.json"

    # Check if files exist
    missing_files = []
    for file_path in [METADATA_FILE, CHUNK_METADATA]:
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

    # Load case-level metadata (gender/age/sentiment already merged by case_id)
    merger.load_metadata(METADATA_FILE)

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
