import json
import re
from typing import Dict, List, Optional

import numpy as np

from src.mitigation.base import MitigationArm
from src.mitigation.neutral_rewrite import swap_gender

# Arm 5: Plug-in Counterfactual Fairness (Zhou, Liu, Bai, Gao, Kocaoglu & Inouye, NeurIPS 2024,
# "Counterfactual Fairness by Combining Factual and Counterfactual Predictions"), Algorithm 1:
#   mu(x) = p(A=a) * phi(x, a) + p(A=1-a) * phi(x_cf, 1-a),   x_cf = G(x, a, 1-a)
# phi is the frozen retrieval score of a chunk, x the query, G the two-sided gender swap, p(A) the corpus prior
# over male vs female applicants. With the true G the score is identical for a query and its gender swap
# (their Proposition 3.5), so the ranking cannot depend on the pronoun.

MALE = re.compile(r"\b(he|him|his|himself|mr)\b", re.IGNORECASE)
FEMALE = re.compile(r"\b(she|her|hers|herself|mrs|ms)\b", re.IGNORECASE)
TRAIN_METADATA_PATH = "dataset/train_with_metadata.jsonl"


# The query's attribute a: the majority gender of its gendered words; None if it has none, "tie" if balanced
def query_attribute(text: str) -> Optional[str]:
    male, female = len(MALE.findall(text)), len(FEMALE.findall(text))
    if male == female:
        return None if male == 0 else "tie"
    return "male" if male > female else "female"


# Algorithm 1 for one candidate set: factual and counterfactual scores (same candidates, same order) -> fair scores
def pcf_scores(factual: np.ndarray, counterfactual: np.ndarray, p_factual: float) -> np.ndarray:
    return p_factual * factual + (1.0 - p_factual) * counterfactual


def corpus_male_prior(path: str = TRAIN_METADATA_PATH) -> float:
    male = female = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            gender = json.loads(line).get("classification", {}).get("gender")
            male += gender == "Male"
            female += gender == "Female"
    return male / (male + female)


class PCF(MitigationArm):
    name = "pcf"

    def __init__(self, p_male: Optional[float] = None):
        self.p_male = corpus_male_prior() if p_male is None else p_male

    def p_factual(self, attribute: str) -> float:
        return {"male": self.p_male, "female": 1.0 - self.p_male, "tie": 0.5}[attribute]

    # Candidates are the union of the factual and counterfactual top-k; every candidate is scored exactly against
    # both queries with its vector reconstructed from the flat index, then ranked by the PCF score
    def rescore_chunks(self, query: str, chunks: List[Dict], retriever, top_k: int) -> List[Dict]:
        attribute = query_attribute(query)
        counterfactual = swap_gender(query)
        if attribute is None or counterfactual == query:
            return chunks

        cf_chunks = retriever.retrieve_chunks(counterfactual, top_k)
        candidates = sorted({c["index"] for c in chunks} | {c["index"] for c in cf_chunks})
        vectors = np.vstack([retriever.index.reconstruct(i) for i in candidates])
        query_vectors = retriever.model.encode([query, counterfactual], convert_to_numpy=True, normalize_embeddings=True)
        scores = pcf_scores(vectors @ query_vectors[0], vectors @ query_vectors[1], self.p_factual(attribute))

        rescored = []
        for k in np.argsort(-scores, kind="stable")[:top_k]:
            chunk = retriever.metadata[candidates[k]].copy()
            chunk["similarity_score"] = float(scores[k])
            chunk["index"] = int(candidates[k])
            rescored.append(chunk)
        return rescored
