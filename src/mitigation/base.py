from typing import Dict, List, Sequence, Tuple

import numpy as np


# A mitigation arm overrides only the hooks it needs; every hook defaults to a no-op,
# so a pipeline without arms (or with this base class) is exactly the frozen system.
class MitigationArm:
    name = "none"

    def rewrite_query(self, query: str) -> str:
        return query

    def transform_query_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector

    def rescore_chunks(self, query: str, chunks: List[Dict], retriever, top_k: int) -> List[Dict]:
        return chunks


# Retrieval with arms applied in order: rewrite the query, transform its vector, retrieve, rescore.
# Shared by the RAG pipeline and the retrieval harness so both apply an arm identically.
def retrieve_with_arms(arms: Sequence[MitigationArm], query: str, retriever, top_k: int) -> Tuple[str, List[Dict]]:
    for arm in arms:
        query = arm.rewrite_query(query)

    def transform(vector: np.ndarray) -> np.ndarray:
        for arm in arms:
            vector = arm.transform_query_vector(vector)
        return vector

    chunks = retriever.retrieve_chunks(query, top_k, transform=transform)
    for arm in arms:
        chunks = arm.rescore_chunks(query, chunks, retriever, top_k)
    return query, chunks
