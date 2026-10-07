from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# answer_once(cases in prompt order) -> (Generation, parsed answer), supplied by the pipeline
AnswerOnce = Callable[[List[Dict]], Tuple[object, Dict]]


# A corpus an arm searches instead of the frozen one; metadata / chunk_text_path None = the frozen ones
@dataclass
class Corpus:
    index: object
    metadata: Optional[List[Dict]] = None
    chunk_text_path: Optional[str] = None


# A mitigation arm overrides only the hooks it needs; every hook defaults to a no-op,
# so a pipeline without arms (or with this base class) is exactly the frozen system.
class MitigationArm:
    name = "none"

    def rewrite_query(self, query: str) -> str:
        return query

    def transform_query_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector

    def corpus(self) -> Optional[Corpus]:
        return None

    def rescore_chunks(self, query: str, chunks: List[Dict], search: "Search", top_k: int) -> List[Dict]:
        return chunks

    # Generation hook: returns (Generation, parsed answer, extra log fields). Default: one answer, retrieved order
    def generate(self, cases: List[Dict], answer_once: AnswerOnce) -> Tuple[object, Dict, Dict]:
        generation, parsed = answer_once(cases)
        return generation, parsed, {}

    def overrides_generation(self) -> bool:
        return type(self).generate is not MitigationArm.generate


def arm_corpus(arms: Sequence[MitigationArm]) -> Optional[Corpus]:
    corpora = [c for arm in arms if (c := arm.corpus()) is not None]
    if len(corpora) > 1:
        raise ValueError("Only one arm can replace the corpus")
    return corpora[0] if corpora else None


# The search the active arms define (query-vector transform, corpus), so a rescoring arm searches and scores
# exactly what the first-stage retrieval did
@dataclass
class Search:
    retriever: object
    transform: Callable
    corpus: Optional[Corpus] = None

    @property
    def index(self):
        return self.corpus.index if self.corpus else self.retriever.index

    @property
    def metadata(self) -> List[Dict]:
        return self.corpus.metadata if self.corpus and self.corpus.metadata is not None else self.retriever.metadata

    def chunks(self, query: str, top_k: int) -> List[Dict]:
        return self.retriever.retrieve_chunks(query, top_k, transform=self.transform, index=self.index,
                                              metadata=self.metadata)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.transform(self.retriever.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))


# Retrieval with arms applied in order: rewrite the query, transform its vector, retrieve, rescore.
# Shared by the RAG pipeline and the retrieval harness so both apply an arm identically.
def retrieve_with_arms(arms: Sequence[MitigationArm], query: str, retriever, top_k: int) -> Tuple[str, List[Dict]]:
    for arm in arms:
        query = arm.rewrite_query(query)

    def transform(vector: np.ndarray) -> np.ndarray:
        for arm in arms:
            vector = arm.transform_query_vector(vector)
        return vector

    corpus = arm_corpus(arms)
    chunks = retriever.retrieve_chunks(query, top_k, transform=transform,
                                       index=corpus and corpus.index, metadata=corpus and corpus.metadata)
    search = Search(retriever, transform, corpus)
    for arm in arms:
        chunks = arm.rescore_chunks(query, chunks, search, top_k)
    return query, chunks
