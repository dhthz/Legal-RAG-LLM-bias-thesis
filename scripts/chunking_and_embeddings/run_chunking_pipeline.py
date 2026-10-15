
import hashlib
import json
import os

import numpy as np

from src.chunking.legal_chunker import LegalCaseChunker, ChunkConfig
from src.embeddings.chunk_embedder import ChunkEmbedder, ChunkRetriever
from src.llm.config import PipelineConfig

# Rebuilds the chunked dataset and the FAISS index from dataset/train.jsonl, then checks the index against the
# SHA256 in the manifest. Next: merge_metadata_to_chunks, then tests.test_frozen_retrieval.
#   python -m scripts.chunking_and_embeddings.run_chunking_pipeline   (GPU, ~30 min)

MANIFEST_PATH = "FROZEN_BASELINE_MANIFEST.json"
REPORT_PATH = "logs/chunking/pipeline_report.json"


def analyze_original_dataset(dataset_path: str):
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

    return {
        'case_count': case_count,
        'avg_paragraphs_per_case': np.mean(paragraph_counts),
        'avg_words_per_case': np.mean(word_counts_per_case),
        'avg_words_per_paragraph': np.mean(word_counts_per_paragraph),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_complete_pipeline():

    print("\n" + "=" * 80)
    print(" " * 20 + "LEGAL CASE CHUNKING & EMBEDDING PIPELINE")
    print("=" * 80)

    DATASET_PATH = "dataset/train.jsonl"
    CHUNKED_OUTPUT = "dataset/train_chunked_paragraphs.jsonl"
    INDEX_DIR = "faiss_indices"
    INDEX_NAME = "paragraph_chunks"
    INDEX_PATH = f"{INDEX_DIR}/{INDEX_NAME}_l2.index"
    pipeline_config = PipelineConfig.load_from_manifest()

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
        strategy='paragraph'
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

    # Step 3: Check the index against the frozen one
    print("\n" + "=" * 60)
    print("STEP 3: CHECKING THE INDEX")
    print("=" * 60)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        frozen_sha256 = json.load(f)["faiss_index"]["sha256_checksum"]
    index_sha256 = file_sha256(INDEX_PATH)
    matches = index_sha256 == frozen_sha256
    print(f"Index SHA256: {index_sha256}")
    print(f"Manifest:     {frozen_sha256}")
    print("Index matches the frozen baseline" if matches else
          "Index differs from the frozen baseline: results will not reproduce the thesis")

    # Step 4: Test retrieval with the frozen settings
    print("\n" + "=" * 60)
    print("STEP 4: TESTING RETRIEVAL")
    print("=" * 60)

    retriever = ChunkRetriever(
        index_path=INDEX_PATH,
        metadata_path=f"{INDEX_DIR}/{INDEX_NAME}_metadata.json"
    )

    test_queries = [
        "The applicant was detained without trial for several years",
        "Article 3 violation involving torture and inhuman treatment",
        "Family reunification and right to respect for family life",
        "Freedom of expression and defamation case involving journalist",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")

        chunks = retriever.retrieve_chunks(query, top_k=pipeline_config.top_k_chunks)
        cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=pipeline_config.top_k_cases)

        print(f"   Retrieved {len(chunks)} chunks from {len(cases)} cases")
        print(f"   Top case: {cases[0]['case_id']} ({cases[0]['max_similarity']:.3f})")
        print(f"   Articles: {cases[0]['violated_articles']}")

    summary = {
        'original_dataset': original_stats,
        'chunking': chunking_stats,
        'embeddings': embedding_stats,
        'index_sha256': index_sha256,
        'index_matches_manifest': matches,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nPipeline report saved to: {REPORT_PATH}")

    print("""
Next steps:
1. Add the case metadata to the chunks:
   $ python -m scripts.dataset.mergers.merge_metadata_to_chunks
2. Check retrieval against the frozen audit run (335/335 identical):
   $ python -m tests.test_frozen_retrieval
3. Explore retrieval interactively:
   $ python -i tests/test_retrieval.py
""")


if __name__ == "__main__":
    run_complete_pipeline()
