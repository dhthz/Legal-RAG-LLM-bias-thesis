from typing import Dict

import faiss
import numpy as np

from src.mitigation.base import Corpus, MitigationArm

# Arm 6: LEACE, least-squares linear concept erasure (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff &
# Biderman, NeurIPS 2023, "LEACE: Perfect linear concept erasure in closed form"), with the authors' library
# (concept-erasure). The eraser is fit on the frozen chunk vectors with each chunk's case gender as the concept;
# afterwards no linear classifier can predict gender from the vectors better than a constant
# (scripts/chunking_and_embeddings/build_leace_index.py builds the erased index and checks this with a probe).
# At query time the same eraser is applied to the query vector and the erased index is searched. As in the paper the
# erased vectors are used as they are: re-normalising them would leak gender back through their length (probe AUC
# 0.48 -> 0.56), and their length stays close to 1 (mean 0.991, sd 0.013), so L2 ranking stays close to cosine.

LEACE_INDEX_PATH = "faiss_indices/leace_paragraph_chunks_l2.index"
LEACE_ERASER_PATH = "faiss_indices/leace_eraser.npz"
LEACE_QUERY_ERASER_PATH = "faiss_indices/leace_query_eraser.npz"


def load_eraser(path: str = LEACE_ERASER_PATH) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in ("bias", "proj_left", "proj_right")}


# r(x) = x - ((x - bias) P_right^T) P_left^T, as LeaceEraser.__call__
def erase(vectors: np.ndarray, eraser: Dict[str, np.ndarray]) -> np.ndarray:
    x = vectors.astype(np.float64)
    return (x - ((x - eraser["bias"]) @ eraser["proj_right"].T) @ eraser["proj_left"].T).astype(np.float32)


class LEACE(MitigationArm):
    name = "leace"

    def __init__(self, index_path: str = LEACE_INDEX_PATH, eraser_path: str = LEACE_ERASER_PATH):
        self.eraser = load_eraser(eraser_path)
        self._corpus = Corpus(faiss.read_index(index_path))

    def transform_query_vector(self, vector: np.ndarray) -> np.ndarray:
        return erase(vector, self.eraser)

    def corpus(self) -> Corpus:
        return self._corpus


# Arm 6, revised: the concept is the pronoun gender of the query (he vs she), the direction a he/she swap moves the
# query along, labelled with he/she twins of dev-split queries (the concept defined by counterfactual pairs, as the
# definitional pairs of Bolukbasi et al. 2016; the erasure is LEACE's closed form). The case-gender concept above
# leaves it readable (probe AUC 0.90). Applied to the query vector only; the frozen index is searched unchanged.
class LEACEQuery(MitigationArm):
    name = "leace_query"

    def __init__(self, eraser_path: str = LEACE_QUERY_ERASER_PATH):
        self.eraser = load_eraser(eraser_path)

    def transform_query_vector(self, vector: np.ndarray) -> np.ndarray:
        return erase(vector, self.eraser)

