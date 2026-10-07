import json
from typing import Dict, Iterable, List

import faiss

from src.chunking.legal_chunker import LegalCaseChunker
from src.mitigation.base import Corpus, MitigationArm
from src.mitigation.neutral_rewrite import neutralize_many

# Arm 2: fairness through unawareness on the document side (Dwork et al. 2012; Kusner et al. 2017).
#   blind_index: every chunk (case intro and body) gets the gender-neutral rewrite of arm 1 and is re-embedded with the
#                frozen model; ids and order are kept, so the frozen metadata applies.
#   facts_only:  ablation, the corpus re-chunked without the case intro (name, birth year, city) that prefixes every
#                chunk; different chunks, so it carries its own metadata.
# In both, the LLM is given the same texts that were searched, as with the rewritten query of arm 1.
# Built by scripts/chunking_and_embeddings/build_blind_index.py.

BLIND_INDEX_PATH = "faiss_indices/blind_paragraph_chunks_l2.index"
BLIND_CHUNKS_PATH = "dataset/train_chunked_paragraphs_blind.jsonl"
FACTS_ONLY_INDEX_PATH = "faiss_indices/facts_only_paragraph_chunks_l2.index"
FACTS_ONLY_CHUNKS_PATH = "dataset/train_chunked_paragraphs_facts_only.jsonl"


def blind_chunks(chunks: List[Dict], batch_size: int = 64, n_process: int = 1) -> List[Dict]:
    texts = neutralize_many((c["chunk_text"] for c in chunks), batch_size=batch_size, n_process=n_process)
    return [{**c, "chunk_text": t} for c, t in zip(chunks, texts)]


def facts_only_chunks(cases: Iterable[Dict], chunker: LegalCaseChunker) -> List[Dict]:
    return [chunk for case in cases for chunk in chunker.create_paragraph_chunks(case, include_intro=False)]


def load_chunk_metadata(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [{k: v for k, v in json.loads(line).items() if k != "chunk_text"} for line in f]


class BlindIndex(MitigationArm):
    name = "blind_index"

    def __init__(self, index_path: str = BLIND_INDEX_PATH, chunks_path: str = BLIND_CHUNKS_PATH):
        self._corpus = Corpus(faiss.read_index(index_path), chunk_text_path=chunks_path)

    def corpus(self) -> Corpus:
        return self._corpus


class FactsOnlyIndex(MitigationArm):
    name = "facts_only"

    def __init__(self, index_path: str = FACTS_ONLY_INDEX_PATH, chunks_path: str = FACTS_ONLY_CHUNKS_PATH):
        self._corpus = Corpus(faiss.read_index(index_path), load_chunk_metadata(chunks_path), chunks_path)

    def corpus(self) -> Corpus:
        return self._corpus
