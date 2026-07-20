from src.chunking.legal_chunker import ChunkConfig, LegalCaseChunker

config = ChunkConfig(
    target_words_per_chunk=150,  # ~200 tokens
    min_words_per_chunk=100,  # Don't go too small
    max_words_per_chunk=250,  # Don't go too large
    overlap_sentences=1,  # Overlap for continuity
    keep_case_intro=True,  # Include case intro
)

chunker = LegalCaseChunker(config)

chunking_stats = chunker.chunk_dataset(
    dataset_path="dataset/train.jsonl",
    output_path="dataset/testing_pipeline.jsonl",
    strategy="paragraph",
)

print(chunking_stats)
