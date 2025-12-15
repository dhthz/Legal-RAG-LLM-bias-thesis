import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def visualize_chunking_results(chunked_file: str, output_dir: str = "visualizations"):
    """Create visualizations of chunking results"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading chunks from: {chunked_file}")

    # Load data
    chunks = []
    with open(chunked_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks")

    # Extract statistics
    word_counts = [c['word_count'] for c in chunks]
    chunk_types = [c['chunk_type'] for c in chunks]
    case_ids = [c['case_id'] for c in chunks]

    chunks_per_case = Counter(case_ids)
    unique_cases = len(chunks_per_case)

    # Visualization 1: Word count distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(word_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(word_counts), color='red', linestyle='--', label=f'Mean: {np.mean(word_counts):.0f}')
    plt.axvline(np.median(word_counts), color='green', linestyle='--', label=f'Median: {np.median(word_counts):.0f}')
    plt.xlabel('Words per Chunk')
    plt.ylabel('Frequency')
    plt.title('Distribution of Chunk Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(word_counts, vert=True)
    plt.ylabel('Words per Chunk')
    plt.title('Chunk Size Box Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/chunk_word_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/chunk_word_distribution.png")
    plt.close()

    # Visualization 2: Chunk types
    plt.figure(figsize=(10, 6))
    type_counts = Counter(chunk_types)
    types = list(type_counts.keys())
    counts = list(type_counts.values())

    plt.bar(types, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Chunk Type')
    plt.ylabel('Count')
    plt.title('Distribution of Chunk Types')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for i, (t, c) in enumerate(zip(types, counts)):
        plt.text(i, c, f'{c}\n({c/len(chunks)*100:.1f}%)', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/chunk_types.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/chunk_types.png")
    plt.close()

    # Visualization 3: Chunks per case
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    chunks_counts = list(chunks_per_case.values())
    plt.hist(chunks_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(chunks_counts), color='red', linestyle='--',
                label=f'Mean: {np.mean(chunks_counts):.1f}')
    plt.xlabel('Chunks per Case')
    plt.ylabel('Frequency')
    plt.title('Distribution of Chunks per Case')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(chunks_counts)), sorted(chunks_counts), alpha=0.5)
    plt.xlabel('Case Index (sorted)')
    plt.ylabel('Number of Chunks')
    plt.title('Chunks per Case (Sorted)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/chunks_per_case.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/chunks_per_case.png")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("CHUNK STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Unique cases: {unique_cases}")
    print(f"Chunks per case: {len(chunks) / unique_cases:.2f}")
    print(f"\nWord counts:")
    print(f"  Mean: {np.mean(word_counts):.1f}")
    print(f"  Median: {np.median(word_counts):.1f}")
    print(f"  Std: {np.std(word_counts):.1f}")
    print(f"  Min: {min(word_counts)}")
    print(f"  Max: {max(word_counts)}")
    print(f"  25th percentile: {np.percentile(word_counts, 25):.1f}")
    print(f"  75th percentile: {np.percentile(word_counts, 75):.1f}")

    print(f"\nChunk types:")
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type}: {count} ({count/len(chunks)*100:.1f}%)")

    # Estimate Mistral 7B context usage
    print(f"\n" + "=" * 60)
    print("MISTRAL 7B CONTEXT ESTIMATION")
    print("=" * 60)

    # Simulate: retrieve 25 chunks, take top 3 per case for top 5 cases
    avg_words_per_chunk = np.mean(word_counts)
    chunks_in_context = 15  # 5 cases × 3 chunks
    estimated_context_words = chunks_in_context * avg_words_per_chunk
    estimated_context_tokens = estimated_context_words * 1.3  # Word to token ratio

    print(f"Scenario: Retrieve 25 chunks, aggregate to 5 cases, use top 3 chunks per case")
    print(f"  Chunks in context: {chunks_in_context}")
    print(f"  Estimated words: {estimated_context_words:.0f}")
    print(f"  Estimated tokens: {estimated_context_tokens:.0f}")
    print(f"\nTotal prompt estimate:")
    print(f"  System + query + template: ~600 tokens")
    print(f"  Retrieved context: ~{estimated_context_tokens:.0f} tokens")
    print(f"  Generation buffer: ~1,500 tokens")
    print(f"  TOTAL: ~{600 + estimated_context_tokens + 1500:.0f} tokens")

    if (600 + estimated_context_tokens + 1500) < 8000:
        print(f"\n✓ Fits in optimal window (< 8,000 tokens)")
    else:
        print(f"\n⚠️  May exceed optimal window!")


def compare_case_before_after(case_id: str, original_file: str, chunked_file: str):
    """Compare a specific case before and after chunking"""

    print(f"\n" + "=" * 60)
    print(f"CASE COMPARISON: {case_id}")
    print("=" * 60)

    # Load original case
    with open(original_file, 'r') as f:
        for line in f:
            case = json.loads(line)
            if case['case_id'] == case_id:
                original_case = case
                break

    # Load chunks
    case_chunks = []
    with open(chunked_file, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['case_id'] == case_id:
                case_chunks.append(chunk)

    # Display comparison
    original_facts = original_case['facts']
    total_original_words = sum(len(p.split()) for p in original_facts)

    print(f"\nORIGINAL CASE:")
    print(f"  Paragraphs: {len(original_facts)}")
    print(f"  Total words: {total_original_words}")
    print(f"  Estimated tokens: {total_original_words * 1.3:.0f}")

    print(f"\nAFTER CHUNKING:")
    print(f"  Total chunks: {len(case_chunks)}")
    print(f"  Avg words per chunk: {np.mean([c['word_count'] for c in case_chunks]):.1f}")

    print(f"\nCHUNK DETAILS:")
    for i, chunk in enumerate(case_chunks[:5], 1):  # Show first 5
        print(f"\n  Chunk {i}:")
        print(f"    ID: {chunk['chunk_id']}")
        print(f"    Type: {chunk['chunk_type']}")
        print(f"    Words: {chunk['word_count']}")
        print(f"    Paragraphs: {chunk['paragraph_indices']}")
        print(f"    Preview: {chunk['chunk_text'][:150]}...")

    if len(case_chunks) > 5:
        print(f"\n  ... and {len(case_chunks) - 5} more chunks")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        chunked_file = sys.argv[1]
    else:
        chunked_file = "dataset/train_chunked_paragraphs.jsonl"

    visualize_chunking_results(chunked_file)

    # Optional: compare specific case
    # compare_case_before_after(
    #     case_id="001-59587",
    #     original_file="dataset/train.jsonl",
    #     chunked_file=chunked_file
    # )
