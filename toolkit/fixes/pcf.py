import json
import re
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .neutral_rewrite import swap_gender

# Plug-in Counterfactual Fairness (Zhou, Liu, Bai, Gao, Kocaoglu & Inouye, NeurIPS 2024,
# "Counterfactual Fairness by Combining Factual and Counterfactual Predictions"), Algorithm 1:
#   mu(x) = p(A=a) * phi(x, a) + p(A=1-a) * phi(x_cf, 1-a),   x_cf = G(x, a, 1-a)
# phi is the retriever's own score of a document for a query, x the query, G the two-sided gender swap, p(A) the
# collection prior over male vs female. With the true G the score is identical for a query and its gender swap
# (their Proposition 3.5), so the ranking cannot depend on the pronoun.

MALE = re.compile(r"\b(he|him|his|himself|mr)\b", re.IGNORECASE)
FEMALE = re.compile(r"\b(she|her|hers|herself|mrs|ms)\b", re.IGNORECASE)


# The query's attribute a: the majority gender of its gendered words; None if it has none, "tie" if balanced
def query_attribute(text: str) -> Optional[str]:
    male, female = len(MALE.findall(text)), len(FEMALE.findall(text))
    if male == female:
        return None if male == 0 else "tie"
    return "male" if male > female else "female"


# Algorithm 1 for one candidate set: factual and counterfactual scores (same candidates, same order) -> fair scores
def pcf_scores(factual: np.ndarray, counterfactual: np.ndarray, p_factual: float) -> np.ndarray:
    return p_factual * factual + (1.0 - p_factual) * counterfactual


# p(male) among the male and female documents of the audit's labels file (doc_labels.jsonl)
def prior_from_labels(path: str, male: str = "Male", female: str = "Female", attribute: str = "gender") -> float:
    counts = {male: 0, female: 0}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["attribute"] == attribute and row["value"] in counts:
                counts[row["value"]] += 1
    return counts[male] / (counts[male] + counts[female])


def p_factual(attribute: str, p_male: float) -> float:
    return {"male": p_male, "female": 1.0 - p_male, "tie": 0.5}[attribute]


# Algorithm 1 around any retriever. search(text, k) returns the top-k ids, score(text, ids) the retriever's score of
# each id for that text (higher = better, the same score the retriever ranks by). Candidates are the union of both
# queries' top-k, each scored against both queries, then ranked by the fair score. Queries without gendered words
# pass through unchanged.
def pcf_rank(query: str, search: Callable[[str, int], List], score: Callable[[str, Sequence], np.ndarray],
             p_male: float, top_k: int) -> Tuple[List, np.ndarray]:
    attribute = query_attribute(query)
    counterfactual = swap_gender(query)
    if attribute is None or counterfactual == query:
        ids = list(search(query, top_k))
        return ids, np.asarray(score(query, ids), dtype=float)
    candidates = list(dict.fromkeys(list(search(query, top_k)) + list(search(counterfactual, top_k))))
    fair = pcf_scores(np.asarray(score(query, candidates), dtype=float),
                      np.asarray(score(counterfactual, candidates), dtype=float), p_factual(attribute, p_male))
    order = np.argsort(-fair, kind="stable")[:top_k]
    return [candidates[i] for i in order], fair[order]
