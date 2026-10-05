from typing import Dict, List

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
