from typing import Dict

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# LEACE, least-squares linear concept erasure (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman,
# NeurIPS 2023), on the query: the concept is the pronoun gender of the query (he vs she), labelled with he/she twin
# questions embedded by the system's own embedder (the concept defined by counterfactual pairs, as the definitional
# pairs of Bolukbasi et al. 2016; the erasure is LEACE's closed form, concept-erasure library defaults). The eraser is
# applied to the query vector only; the document index is searched unchanged. The twins used to fit it must not be the
# audit's own questions.


# he_vectors[i] and she_vectors[i] are the embeddings of the two versions of the same question
def fit_query_eraser(he_vectors: np.ndarray, she_vectors: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.vstack([he_vectors, she_vectors]).astype(np.float64)
    z = np.r_[np.zeros(len(he_vectors)), np.ones(len(she_vectors))][:, None]
    eraser = LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(z))
    params = {"bias": eraser.bias.numpy(), "proj_left": eraser.proj_left.numpy(), "proj_right": eraser.proj_right.numpy()}
    # The numpy transform used at query time must match the library's eraser
    ref = eraser(torch.from_numpy(x[:1000])).numpy()
    assert np.allclose(erase(x[:1000], params), ref, atol=1e-5), "numpy erase() differs from LeaceEraser"
    return params


# r(x) = x - ((x - bias) P_right^T) P_left^T, as LeaceEraser.__call__. As in the paper the erased vectors are used as
# they are: re-normalising them can leak the concept back through their length.
def erase(vectors: np.ndarray, eraser: Dict[str, np.ndarray]) -> np.ndarray:
    x = vectors.astype(np.float64)
    return (x - ((x - eraser["bias"]) @ eraser["proj_right"].T) @ eraser["proj_left"].T).astype(np.float32)


def save_eraser(path: str, eraser: Dict[str, np.ndarray]) -> None:
    np.savez(path, **eraser)


def load_eraser(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in ("bias", "proj_left", "proj_right")}


# Held-out check: can a linear probe still tell he from she after erasure? AUC 0.5 = no. Fit the probe on the
# training twins, test on twins that were not used to fit the eraser.
def probe_auc(train_he, train_she, test_he, test_she, eraser) -> Dict[str, float]:
    x, y = np.vstack([train_he, train_she]), np.r_[np.zeros(len(train_he)), np.ones(len(train_she))]
    vx, vy = np.vstack([test_he, test_she]), np.r_[np.zeros(len(test_he)), np.ones(len(test_she))]
    out = {}
    for name, f in (("original", lambda v: v), ("erased", lambda v: erase(v, eraser))):
        clf = LogisticRegression(max_iter=5000).fit(f(x), y)
        out[name] = float(roc_auc_score(vy, clf.predict_proba(f(vx))[:, 1]))
    return out
