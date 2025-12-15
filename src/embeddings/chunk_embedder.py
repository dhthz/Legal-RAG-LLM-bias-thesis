"""
Chunk Embedding Module
======================

Creates embeddings for chunked legal cases optimized for:
- Paragraph-level retrieval
- Mistral 7B context window
- Bias analysis at different granularities
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import torch
from typing import List, Dict, Optional
from pathlib import Path


class ChunkEmbedder:
    """
    Creates and manages embeddings for chunked legal cases
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize embedder

        Args:
            model_name: Sentence transformer model to use
            device: 'cuda' or 'cpu'
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Fix GPU memory fragmentation
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

        # Get embedding dimension
        sample_embedding = self.model.encode(["test"], show_progress_bar=False)
        self.embedding_dim = sample_embedding.shape[1]
        print(f"Embedding dimension: {self.embedding_dim}")

    def load_chunks(self, chunk_file: str) -> tuple[List[str], List[Dict]]:
        """
        Load chunks from JSONL file

        Args:
            chunk_file: Path to chunked dataset JSONL

        Returns:
            Tuple of (chunk_texts, chunk_metadata)
        """
        print(f"Loading chunks from: {chunk_file}")

        chunk_texts = []
        chunk_metadata = []

        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading chunks"):
                chunk = json.loads(line)
                chunk_texts.append(chunk['chunk_text'])

                # Store metadata (everything except the text itself)
                metadata = {k: v for k, v in chunk.items() if k != 'chunk_text'}
                chunk_metadata.append(metadata)

        print(f"Loaded {len(chunk_texts)} chunks")
        return chunk_texts, chunk_metadata

    def create_embeddings(
        self,
        chunk_texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for all chunks

        Args:
            chunk_texts: List of chunk text strings
            normalize: Whether to normalize embeddings (recommended for L2/cosine)

        Returns:
            Numpy array of embeddings
        """
        print(f"\nCreating embeddings for {len(chunk_texts)} chunks...")

        embeddings = self.model.encode(
            chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "L2"
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Numpy array of embeddings
            index_type: 'L2' or 'IP' (inner product)

        Returns:
            FAISS index
        """
        print(f"\nBuilding FAISS {index_type} index...")

        if index_type == "L2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "IP":
            index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings.astype('float32'))
        print(f"Index created with {index.ntotal} vectors")

        return index

    def save_index_and_metadata(
        self,
        index: faiss.Index,
        metadata: List[Dict],
        index_path: str,
        metadata_path: str
    ):
        """
        Save FAISS index and metadata to disk

        Args:
            index: FAISS index
            metadata: List of metadata dictionaries
            index_path: Where to save index
            metadata_path: Where to save metadata JSON
        """
        # Create directories if needed
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(index, index_path)
        print(f"Index saved to: {index_path}")

        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to: {metadata_path}")

        # Print file sizes
        index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
        metadata_size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
        print(f"\nFile sizes:")
        print(f"  Index: {index_size_mb:.2f} MB")
        print(f"  Metadata: {metadata_size_mb:.2f} MB")
        print(f"  Total: {index_size_mb + metadata_size_mb:.2f} MB")

    def process_chunked_dataset(
        self,
        chunk_file: str,
        output_dir: str,
        index_name: str = "paragraph_chunks"
    ) -> Dict:
        """
        Complete pipeline: load chunks -> embed -> build index -> save

        Args:
            chunk_file: Path to chunked JSONL file
            output_dir: Directory to save index and metadata
            index_name: Base name for output files

        Returns:
            Statistics dictionary
        """
        # Load chunks
        chunk_texts, chunk_metadata = self.load_chunks(chunk_file)

        # Create embeddings
        embeddings = self.create_embeddings(chunk_texts)

        # Build index
        index = self.build_faiss_index(embeddings)

        # Define output paths
        index_path = os.path.join(output_dir, f"{index_name}_l2.index")
        metadata_path = os.path.join(output_dir, f"{index_name}_metadata.json")

        # Save
        self.save_index_and_metadata(index, chunk_metadata, index_path, metadata_path)

        # Calculate statistics
        word_counts = [meta['word_count'] for meta in chunk_metadata]
        unique_cases = set(meta['case_id'] for meta in chunk_metadata)

        stats = {
            'total_chunks': len(chunk_texts),
            'total_cases': len(unique_cases),
            'chunks_per_case': len(chunk_texts) / len(unique_cases),
            'avg_chunk_words': np.mean(word_counts),
            'median_chunk_words': np.median(word_counts),
            'min_chunk_words': min(word_counts),
            'max_chunk_words': max(word_counts),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'index_path': index_path,
            'metadata_path': metadata_path,
        }

        print("\n" + "=" * 60)
        print("EMBEDDING STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if key not in ['index_path', 'metadata_path']:
                print(f"{key}: {value}")
        print("=" * 60)

        return stats


class ChunkRetriever:
    """
    Retrieve relevant chunks and aggregate to case level
    """

    def __init__(self, index_path: str, metadata_path: str, model_name: str = "nomic-ai/nomic-embed-text-v1"):
        """
        Initialize retriever

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            model_name: Model used for embeddings
        """
        self.index = faiss.read_index(index_path)
        print(f"Loaded index with {self.index.ntotal} chunks")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata)} chunks")

        # Load model for query encoding
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda')

    def retrieve_chunks(self, query: str, top_k: int = 25) -> List[Dict]:
        """
        Retrieve most similar chunks

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_meta = self.metadata[idx].copy()
            similarity_score = 1 - (distance ** 2 / 2)  # Convert L2 to similarity
            chunk_meta['similarity_score'] = float(similarity_score)
            chunk_meta['l2_distance'] = float(distance)
            chunk_meta['index'] = int(idx)
            results.append(chunk_meta)

        return results

    def aggregate_chunks_to_cases(self, chunks: List[Dict], top_k_cases: int = 5) -> List[Dict]:
        """
        Aggregate retrieved chunks to case level
        Keep the most relevant chunks per case

        Args:
            chunks: List of retrieved chunks
            top_k_cases: Number of unique cases to return

        Returns:
            List of cases with their most relevant chunks
        """
        from collections import defaultdict

        # Group chunks by case_id
        case_chunks = defaultdict(list)
        for chunk in chunks:
            case_chunks[chunk['case_id']].append(chunk)

        # Calculate case-level scores (average of chunk scores)
        case_scores = []
        for case_id, chunks_list in case_chunks.items():
            avg_score = np.mean([c['similarity_score'] for c in chunks_list])
            max_score = max([c['similarity_score'] for c in chunks_list])

            case_scores.append({
                'case_id': case_id,
                'avg_similarity': avg_score,
                'max_similarity': max_score,
                'num_chunks': len(chunks_list),
                'chunks': sorted(chunks_list, key=lambda x: x['similarity_score'], reverse=True),
                # Case-level metadata from first chunk
                'case_no': chunks_list[0]['case_no'],
                'title': chunks_list[0]['title'],
                'judgment_date': chunks_list[0]['judgment_date'],
                'violated_articles': chunks_list[0]['violated_articles'],
            })

        # Sort by average similarity
        case_scores.sort(key=lambda x: x['avg_similarity'], reverse=True)

        return case_scores[:top_k_cases]


# ==========================
# Example Usage
# ==========================

if __name__ == "__main__":
    print("=" * 60)
    print("CHUNK EMBEDDING PIPELINE")
    print("=" * 60)

    # Step 1: Create embeddings for chunked dataset
    embedder = ChunkEmbedder(
        model_name="nomic-ai/nomic-embed-text-v1",
        device="cuda",
        batch_size=32
    )

    stats = embedder.process_chunked_dataset(
        chunk_file="dataset/train_chunked.jsonl",
        output_dir="faiss_indices",
        index_name="paragraph_chunks"
    )

    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL")
    print("=" * 60)

    # Step 2: Test retrieval
    retriever = ChunkRetriever(
        index_path="faiss_indices/paragraph_chunks_l2.index",
        metadata_path="faiss_indices/paragraph_chunks_metadata.json"
    )

    query = "The applicant was detained without trial for 3 years"
    print(f"\nQuery: {query}")

    # Retrieve chunks
    chunks = retriever.retrieve_chunks(query, top_k=25)
    print(f"\nRetrieved {len(chunks)} chunks")

    # Aggregate to cases
    cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=5)

    print(f"\nTop {len(cases)} cases:")
    for i, case in enumerate(cases, 1):
        print(f"\n{i}. {case['case_id']} - {case['title']}")
        print(f"   Avg similarity: {case['avg_similarity']:.4f}")
        print(f"   Relevant chunks: {case['num_chunks']}")
        print(f"   Violated articles: {case['violated_articles']}")
        print(f"   Top chunk: {case['chunks'][0]['chunk_id']}")

    print("\n" + "=" * 60)
    print("Ready for RAG pipeline integration!")
    print("=" * 60)
