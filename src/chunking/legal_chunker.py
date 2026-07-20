import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class ChunkConfig:
    # Target chunk sizes (in words)
    target_words_per_chunk: int = 150  # ~200 tokens
    min_words_per_chunk: int = 100
    max_words_per_chunk: int = 250

    # Overlap for context continuity
    overlap_sentences: int = 1  # Number of sentences to overlap

    # Legal-specific settings
    preserve_paragraph_boundaries: bool = True
    keep_case_intro: bool = True  # Always include case intro with each chunk

    # Mistral 7B optimization
    max_chunks_per_retrieval: int = 25  # Total chunks to retrieve
    final_chunks_after_rrf: int = 15  # After Reciprocal Rank Fusion

    def to_dict(self) -> Dict:
        return asdict(self)


class LegalCaseChunker:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.reset_loss_stats()

    def reset_loss_stats(self):
        self.loss_stats = {
            "short_paragraphs_merged": 0,
            "short_paragraph_words_merged": 0,
            "tails_dropped": 0,
            "tails_words_dropped": 0,
        }

    def count_words(self, text: str) -> int:
        return len(text.split())

    def split_into_sentences(self, text: str) -> List[str]:
        # Protect legal patterns from sentence splitting
        text = re.sub(r"(Art\.|Article|Section|Paragraph)\s+(\d+)", r"\1_\2", text)
        text = re.sub(
            r"(\d+)\.\s+", r"\1_PERIOD_ ", text
        )  # Protect numbered paragraphs

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Restore protected patterns
        sentences = [s.replace("_PERIOD_", ".").replace("_", " ") for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def extract_case_intro(self, facts: List[str]) -> str:
        if not facts:
            return ""

        # Usually the first 1-2 paragraphs contain:
        # - Applicant info (name, birth year, location)
        # - Basic case circumstances
        intro_paragraphs = facts[:2] if len(facts) > 1 else facts[:1]
        intro_text = " ".join(intro_paragraphs)

        # Limit intro to ~50 words to avoid bloat
        words = intro_text.split()[:50]
        return " ".join(words) + "..." if len(words) == 50 else " ".join(words)

    def _build_chunk(self, case, chunk_id, chunk_text, paragraph_indices, chunk_type):
        return {
            "chunk_id": chunk_id,
            "case_id": case["case_id"],
            "chunk_text": chunk_text,
            "paragraph_indices": paragraph_indices,
            "chunk_type": chunk_type,
            "word_count": self.count_words(chunk_text),
            "case_no": case.get("case_no", ""),
            "title": case.get("title", ""),
            "judgment_date": case.get("judgment_date", ""),
            "violated_articles": case.get("violated_articles", []),
            "allegedly_violated_articles": case.get("allegedly_violated_articles", []),
        }

    def create_paragraph_chunks(
        self, case: Dict, include_intro: bool = True
    ) -> List[Dict]:
        facts = case.get("facts", [])
        if not facts:
            return []

        case_intro = self.extract_case_intro(facts) if include_intro else ""
        chunks = []
        pending_text = ""
        pending_indices = []

        # Process each paragraph
        for para_idx, paragraph in enumerate(facts):
            # Check if paragraph is small enough to be a single chunk

            # Skip very short paragraphs with statistics to see if changes to embedding rules need to be done (likely formatting artifacts)
            para_word_count = self.count_words(paragraph)
            if para_word_count < 10:
                pending_text = (pending_text + " " + paragraph).strip()
                pending_indices.append(para_idx)
                self.loss_stats["short_paragraphs_merged"] += 1
                self.loss_stats["short_paragraph_words_merged"] += para_word_count
                continue

            para_indices = pending_indices + [para_idx]
            if pending_text:
                paragraph = pending_text + " " + paragraph
                para_word_count = self.count_words(paragraph)
            pending_text, pending_indices = "", []

            if para_word_count <= self.config.max_words_per_chunk:
                # Single paragraph = single chunk
                chunk_text = paragraph
                if case_intro and include_intro:
                    chunk_text = f"{case_intro} [...] {paragraph}"

                chunks.append(
                    self._build_chunk(
                        case,
                        f"{case['case_id']}_p{para_idx}",
                        chunk_text,
                        para_indices,
                        "single_paragraph",
                    )
                )
            else:
                # Large paragraph: split into sentences
                sentences = self.split_into_sentences(paragraph)

                # Group sentences into chunks
                current_chunk_sentences = []
                current_word_count = 0
                chunk_emitted_for_paragraph = False

                for sent_idx, sentence in enumerate(sentences):
                    sent_words = self.count_words(sentence)

                    # Check if adding this sentence exceeds max
                    if (
                        current_word_count + sent_words
                        > self.config.max_words_per_chunk
                        and current_chunk_sentences
                    ):
                        # Save current chunk
                        chunk_text = " ".join(current_chunk_sentences)
                        if case_intro and include_intro:
                            chunk_text = f"{case_intro} [...] {chunk_text}"

                        chunks.append(
                            self._build_chunk(
                                case,
                                f"{case['case_id']}_p{para_idx}_s{sent_idx}",
                                chunk_text,
                                para_indices,
                                "sentence_group",
                            )
                        )

                        # Start new chunk with overlap
                        if self.config.overlap_sentences > 0:
                            overlap_start = max(
                                0,
                                len(current_chunk_sentences)
                                - self.config.overlap_sentences,
                            )
                            current_chunk_sentences = current_chunk_sentences[
                                overlap_start:
                            ]
                            current_word_count = sum(
                                self.count_words(s) for s in current_chunk_sentences
                            )
                        else:
                            current_chunk_sentences = []
                            current_word_count = 0

                    current_chunk_sentences.append(sentence)
                    current_word_count += sent_words

                # Don't forget the last chunk
                if (
                    current_chunk_sentences
                    and current_word_count >= self.config.min_words_per_chunk
                ):
                    chunk_text = " ".join(current_chunk_sentences)
                    if case_intro and include_intro:
                        chunk_text = f"{case_intro} [...] {chunk_text}"

                    chunks.append(
                        self._build_chunk(
                            case,
                            f"{case['case_id']}_p{para_idx}_final",
                            chunk_text,
                            para_indices,
                            "sentence_group",
                        )
                    )
                elif current_chunk_sentences:
                    self.loss_stats["tails_dropped"] += 1
                    self.loss_stats["tails_words_dropped"] += current_word_count

        # Attaches short trailing short paragraphs into the following paragraph to keep as many sentences whilst preserving context
        if pending_text:
            if chunks:
                chunks[-1]["chunk_text"] += " " + pending_text
                chunks[-1]["word_count"] = self.count_words(chunks[-1]["chunk_text"])
                chunks[-1]["paragraph_indices"] += pending_indices
            else:
                chunk_text = pending_text
                if case_intro and include_intro:
                    chunk_text = f"{case_intro} [...] {pending_text}"
                chunks.append(
                    self._build_chunk(
                        case,
                        f"{case['case_id']}_trailing",
                        chunk_text,
                        pending_indices,
                        "single_paragraph",
                    )
                )

        return chunks

    # Dead code kept for experimentation on later on stages
    # def create_multi_paragraph_chunks(self, case: Dict) -> List[Dict]:
    #     facts = case.get("facts", [])
    #     if not facts:
    #         return []

    #     chunks = []
    #     current_paragraphs = []
    #     current_para_indices = []
    #     current_word_count = 0

    #     for para_idx, paragraph in enumerate(facts):
    #         para_words = self.count_words(paragraph)

    #         # Check if we should start a new chunk
    #         if (
    #             current_word_count + para_words > self.config.max_words_per_chunk
    #             and current_paragraphs
    #         ):
    #             # Save current chunk
    #             chunk_text = " ".join(current_paragraphs)
    #             chunks.append(
    #                 {
    #                     "chunk_id": f"{case['case_id']}_mp{para_idx}",
    #                     "case_id": case["case_id"],
    #                     "chunk_text": chunk_text,
    #                     "paragraph_indices": current_para_indices.copy(),
    #                     "chunk_type": "multi_paragraph",
    #                     "word_count": current_word_count,
    #                     "case_no": case.get("case_no", ""),
    #                     "title": case.get("title", ""),
    #                     "judgment_date": case.get("judgment_date", ""),
    #                     "violated_articles": case.get("violated_articles", []),
    #                     "allegedly_violated_articles": case.get(
    #                         "allegedly_violated_articles", []
    #                     ),
    #                 }
    #             )

    #             # Start new chunk with overlap
    #             if self.config.overlap_sentences > 0 and current_paragraphs:
    #                 current_paragraphs = [current_paragraphs[-1]]  # Keep last paragraph
    #                 current_para_indices = [current_para_indices[-1]]
    #                 current_word_count = self.count_words(current_paragraphs[0])
    #             else:
    #                 current_paragraphs = []
    #                 current_para_indices = []
    #                 current_word_count = 0

    #         current_paragraphs.append(paragraph)
    #         current_para_indices.append(para_idx)
    #         current_word_count += para_words

    #     # Add final chunk
    #     if current_paragraphs and current_word_count >= self.config.min_words_per_chunk:
    #         chunk_text = " ".join(current_paragraphs)
    #         chunks.append(
    #             {
    #                 "chunk_id": f"{case['case_id']}_mp_final",
    #                 "case_id": case["case_id"],
    #                 "chunk_text": chunk_text,
    #                 "paragraph_indices": current_para_indices,
    #                 "chunk_type": "multi_paragraph",
    #                 "word_count": current_word_count,
    #                 "case_no": case.get("case_no", ""),
    #                 "title": case.get("title", ""),
    #                 "judgment_date": case.get("judgment_date", ""),
    #                 "violated_articles": case.get("violated_articles", []),
    #                 "allegedly_violated_articles": case.get(
    #                     "allegedly_violated_articles", []
    #                 ),
    #             }
    #         )

    #     return chunks

    def chunk_dataset(
        self,
        dataset_path: str,
        output_path: str,
        strategy: str = "paragraph",  # 'paragraph' or 'multi_paragraph'
    ) -> Dict:
        print(f"Processing dataset: {dataset_path}")
        print(f"Chunking strategy: {strategy}")
        print(f"Config: {self.config.to_dict()}")

        all_chunks = []
        case_count = 0
        total_paragraphs = 0
        self.reset_loss_stats()

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                case = json.loads(line)
                case_count += 1
                total_paragraphs += len(case.get("facts", []))

                # Choose chunking strategy
                if strategy == "paragraph":
                    chunks = self.create_paragraph_chunks(case)
                else:
                    chunks = self.create_multi_paragraph_chunks(case)

                all_chunks.extend(chunks)

                if case_count % 100 == 0:
                    print(
                        f"Processed {case_count} cases, {len(all_chunks)} chunks created..."
                    )

        # Save chunks to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # Calculate statistics
        word_counts = [c["word_count"] for c in all_chunks]
        stats = {
            "total_cases": case_count,
            "total_paragraphs": total_paragraphs,
            "total_chunks": len(all_chunks),
            "chunks_per_case_avg": len(all_chunks) / case_count,
            "chunks_per_paragraph_avg": len(all_chunks) / total_paragraphs
            if total_paragraphs > 0
            else 0,
            "avg_words_per_chunk": sum(word_counts) / len(word_counts)
            if word_counts
            else 0,
            "min_words_per_chunk": min(word_counts) if word_counts else 0,
            "max_words_per_chunk": max(word_counts) if word_counts else 0,
            "config": self.config.to_dict(),
            **self.loss_stats,
        }

        print("\n" + "=" * 60)
        print("CHUNKING STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if key != "config":
                print(f"{key}: {value}")
        print("=" * 60)

        return stats


# ==========================
# Example Usage
# ==========================

if __name__ == "__main__":
    # Create chunker with custom config
    config = ChunkConfig(
        target_words_per_chunk=150,
        min_words_per_chunk=100,
        max_words_per_chunk=250,
        overlap_sentences=1,
        keep_case_intro=True,
    )

    chunker = LegalCaseChunker(config)

    # Test with sample case
    sample_case = {
        "case_id": "001-59587",
        "case_no": "25702/94",
        "title": "CASE OF K. AND T. v. FINLAND",
        "judgment_date": "2001-07-12",
        "violated_articles": ["8"],
        "facts": [
            "The applicant was born in 1975 and lives in Helsinki.",
            "On 15 March 2001, the applicant was arrested by police without a warrant. "
            "The arrest took place at 3:00 AM at the applicant's residence. "
            "According to the police report, the applicant was suspected of theft.",
            "During the interrogation, which lasted for 6 hours, the applicant was not "
            "provided with legal counsel despite repeated requests. The interrogation "
            "was conducted in a small room without windows.",
        ],
    }

    print("\n" + "=" * 60)
    print("TESTING PARAGRAPH CHUNKING STRATEGY")
    print("=" * 60)

    chunks = chunker.create_paragraph_chunks(sample_case)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Type: {chunk['chunk_type']}")
        print(f"  Words: {chunk['word_count']}")
        print(f"  Paragraph indices: {chunk['paragraph_indices']}")
        print(f"  Text preview: {chunk['chunk_text'][:200]}...")

    print("\n" + "=" * 60)
    print("Ready to process full dataset!")
    print("=" * 60)
    print("""
To process your dataset:

    from src.chunking.legal_chunker import LegalCaseChunker, ChunkConfig

    chunker = LegalCaseChunker()
    stats = chunker.chunk_dataset(
        dataset_path='dataset/train.jsonl',
        output_path='dataset/train_chunked.jsonl',
        strategy='paragraph'
    )
    """)
