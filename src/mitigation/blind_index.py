from typing import Dict, List

import faiss

from src.mitigation.base import Corpus, MitigationArm
from src.mitigation.neutral_rewrite import neutralize_many

# Arm 2: fairness through unawareness on the document side (Dwork et al. 2012; Kusner et al. 2017).
# Every chunk (case intro and body) gets the gender-neutral rewrite of arm 1 and is re-embedded with the frozen model;
# ids and order are kept, so the frozen metadata applies. The LLM is given the same texts that were searched, as with
# the rewritten query of arm 1.
# Built by scripts/chunking_and_embeddings/build_blind_index.py.

BLIND_INDEX_PATH = "faiss_indices/blind_paragraph_chunks_l2.index"
BLIND_CHUNKS_PATH = "dataset/train_chunked_paragraphs_blind.jsonl"


def blind_chunks(chunks: List[Dict], batch_size: int = 64, n_process: int = 1) -> List[Dict]:
    texts = neutralize_many((c["chunk_text"] for c in chunks), batch_size=batch_size, n_process=n_process)
    return [{**c, "chunk_text": t} for c, t in zip(chunks, texts)]


class BlindIndex(MitigationArm):
    name = "blind_index"

    def __init__(self, index_path: str = BLIND_INDEX_PATH, chunks_path: str = BLIND_CHUNKS_PATH):
        self._corpus = Corpus(faiss.read_index(index_path), chunk_text_path=chunks_path)

    def corpus(self) -> Corpus:
        return self._corpus
