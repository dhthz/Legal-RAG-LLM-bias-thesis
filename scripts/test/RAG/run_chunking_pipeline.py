"""
Complete Chunking Pipeline Runner
=================================
This script runs the complete pipeline:
1. Chunk the dataset
2. Create embeddings
3. Build FAISS index
4. Test retrieval
5. Generate analysis report
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.chunking.legal_chunker import LegalCaseChunker, ChunkConfig
from src.embeddings.chunk_embedder import ChunkEmbedder, ChunkRetriever
import json
from pathlib import Path


def analyze_original_dataset(dataset_path: str):
    """Analyze the original dataset to inform chunking decisions"""
    print("\n" + "=" * 60)
    print("ANALYZING ORIGINAL DATASET")
    print("=" * 60)

    case_count = 0
    total_paragraphs = 0
    total_words = 0
    paragraph_counts = []
    word_counts_per_case = []
    word_counts_per_paragraph = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            case_count += 1

            facts = case.get('facts', [])
            paragraph_counts.append(len(facts))
            total_paragraphs += len(facts)

            case_words = 0
            for para in facts:
                words = len(para.split())
                case_words += words
                total_words += words
                word_counts_per_paragraph.append(words)

            word_counts_per_case.append(case_words)

    import numpy as np

    print(f"\nTotal cases: {case_count}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Avg paragraphs per case: {np.mean(paragraph_counts):.1f}")
    print(f"Median paragraphs per case: {np.median(paragraph_counts):.0f}")
    print(f"Min/Max paragraphs: {min(paragraph_counts)} / {max(paragraph_counts)}")

    print(f"\nTotal words: {total_words:,}")
    print(f"Avg words per case: {np.mean(word_counts_per_case):.0f}")
    print(f"Median words per case: {np.median(word_counts_per_case):.0f}")
    print(f"Min/Max words per case: {min(word_counts_per_case)} / {max(word_counts_per_case)}")

    print(f"\nAvg words per paragraph: {np.mean(word_counts_per_paragraph):.0f}")
    print(f"Median words per paragraph: {np.median(word_counts_per_paragraph):.0f}")
    print(f"Min/Max words per paragraph: {min(word_counts_per_paragraph)} / {max(word_counts_per_paragraph)}")

    # Estimate tokens (rough estimate: 1 word = 1.3 tokens)
    avg_tokens_per_case = np.mean(word_counts_per_case) * 1.3
    print(f"\nEstimated avg tokens per case: {avg_tokens_per_case:.0f}")

    if avg_tokens_per_case > 6000:
        print("⚠️  WARNING: Average case size exceeds Mistral 7B optimal context!")
        print("   Chunking is ESSENTIAL for this dataset.")
    else:
        print("✓ Average case size is manageable for Mistral 7B")

    return {
        'case_count': case_count,
        'avg_paragraphs_per_case': np.mean(paragraph_counts),
        'avg_words_per_case': np.mean(word_counts_per_case),
        'avg_words_per_paragraph': np.mean(word_counts_per_paragraph),
    }


def run_complete_pipeline():
    """Run the complete chunking and embedding pipeline"""

    print("\n" + "=" * 80)
    print(" " * 20 + "LEGAL CASE CHUNKING & EMBEDDING PIPELINE")
    print("=" * 80)

    # Configuration
    DATASET_PATH = "dataset/train.jsonl"
    CHUNKED_OUTPUT = "dataset/train_chunked_paragraphs.jsonl"
    INDEX_DIR = "faiss_indices"
    INDEX_NAME = "paragraph_chunks"

    # Step 0: Analyze original dataset
    original_stats = analyze_original_dataset(DATASET_PATH)

    # Step 1: Chunk the dataset
    print("\n" + "=" * 60)
    print("STEP 1: CHUNKING DATASET")
    print("=" * 60)

    config = ChunkConfig(
        target_words_per_chunk=150,     # ~200 tokens
        min_words_per_chunk=100,         # Don't go too small
        max_words_per_chunk=250,         # Don't go too large
        overlap_sentences=1,             # Overlap for continuity
        keep_case_intro=True,            # Include case intro
    )

    chunker = LegalCaseChunker(config)

    chunking_stats = chunker.chunk_dataset(
        dataset_path=DATASET_PATH,
        output_path=CHUNKED_OUTPUT,
        strategy='paragraph'  # or 'multi_paragraph'
    )

    # Step 2: Create embeddings and index
    print("\n" + "=" * 60)
    print("STEP 2: CREATING EMBEDDINGS & INDEX")
    print("=" * 60)

    embedder = ChunkEmbedder(
        model_name="nomic-ai/nomic-embed-text-v1",
        device="cuda",
        batch_size=32
    )

    embedding_stats = embedder.process_chunked_dataset(
        chunk_file=CHUNKED_OUTPUT,
        output_dir=INDEX_DIR,
        index_name=INDEX_NAME
    )

    # Step 3: Test retrieval
    print("\n" + "=" * 60)
    print("STEP 3: TESTING RETRIEVAL")
    print("=" * 60)

    retriever = ChunkRetriever(
        index_path=f"{INDEX_DIR}/{INDEX_NAME}_l2.index",
        metadata_path=f"{INDEX_DIR}/{INDEX_NAME}_metadata.json"
    )

    # Test queries
    test_queries = [
        "The applicant was detained without trial for several years",
        "Article 3 violation involving torture and inhuman treatment",
        "Family reunification and right to respect for family life",
        "Freedom of expression and defamation case involving journalist",
    ]

    print("\nTest Queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")

        chunks = retriever.retrieve_chunks(query, top_k=25)
        cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=5)

        print(f"   Retrieved {len(chunks)} chunks from {len(cases)} cases")
        print(f"   Top case: {cases[0]['case_id']} ({cases[0]['avg_similarity']:.3f})")
        print(f"   Articles: {cases[0]['violated_articles']}")

    # Step 4: Calculate context window usage
    print("\n" + "=" * 60)
    print("STEP 4: MISTRAL 7B CONTEXT WINDOW ANALYSIS")
    print("=" * 60)

    # Simulate retrieval for context estimation
    query = "The applicant was detained without trial"
    chunks = retriever.retrieve_chunks(query, top_k=25)
    cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=5)

    # Calculate total words in retrieved context
    total_context_words = 0
    total_chunks_used = 0

    for case in cases:
        # Take top 3 most relevant chunks per case
        for chunk in case['chunks'][:3]:
            total_context_words += chunk['word_count']
            total_chunks_used += 1

    # Estimate tokens (word * 1.3)
    total_context_tokens = total_context_words * 1.3

    print(f"\nContext Window Usage Estimate:")
    print(f"  Chunks retrieved: 25")
    print(f"  Cases aggregated: {len(cases)}")
    print(f"  Chunks used in context: {total_chunks_used}")
    print(f"  Total words: {total_context_words}")
    print(f"  Estimated tokens: {total_context_tokens:.0f}")

    print(f"\nMistral 7B Prompt Breakdown:")
    print(f"  System prompt:      ~200 tokens")
    print(f"  User query:         ~100 tokens")
    print(f"  Retrieved context:  ~{total_context_tokens:.0f} tokens")
    print(f"  Prompt template:    ~300 tokens")
    print(f"  Generation buffer:  ~1,500 tokens")
    print(f"  " + "-" * 40)
    print(f"  TOTAL:             ~{200 + 100 + total_context_tokens + 300 + 1500:.0f} tokens")

    if (200 + 100 + total_context_tokens + 300 + 1500) < 8000:
        print("\n✓ Context fits comfortably in Mistral 7B optimal window (8k tokens)")
    else:
        print("\n⚠️  WARNING: Context may exceed optimal window!")
        print("   Consider reducing chunks per case or number of cases")

    # Step 5: Generate summary report
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY REPORT")
    print("=" * 60)

    summary = {
        'original_dataset': original_stats,
        'chunking': chunking_stats,
        'embeddings': embedding_stats,
        'context_window_estimate_tokens': int(total_context_tokens),
        'total_pipeline_tokens_estimate': int(200 + 100 + total_context_tokens + 300 + 1500),
    }

    # Save report
    report_path = "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nPipeline report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 60)

    print(f"""
Next Steps:

1. Check the chunked dataset:
   $ head -n 5 {CHUNKED_OUTPUT}

2. Test retrieval interactively:
   $ python -i scripts/test_retrieval.py

3. Integrate with Mistral 7B:
   - Use ChunkRetriever to get relevant chunks
   - Aggregate to top 5 cases
   - Take top 3 chunks per case
   - Format into Mistral prompt template

4. For bias analysis:
   - Create semantic chunks using multi_paragraph strategy
   - Build separate index for bias testing
   - Compare retrieval patterns across both indices
    """)


if __name__ == "__main__":
    run_complete_pipeline()
