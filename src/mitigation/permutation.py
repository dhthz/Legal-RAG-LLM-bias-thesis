from collections import Counter
from itertools import combinations, permutations
from statistics import median_low
from typing import Dict, List, Sequence, Tuple

from src.llm.client import Generation
from src.mitigation.base import AnswerOnce, MitigationArm

# Arm 3, permutation self-consistency (Tang et al., NAACL 2024, "Found in the Middle"): the LLM answers once for every
# order of the retrieved cases, and the answers are aggregated so that no case is favoured by its position.
# All n! orders are used (6 for top-3), so there is no sample size to choose. The identity order comes first and is
# the plain frozen prompt.
#   Cited cases, as in Tang et al.: each answer is a ranking (cited cases in the answer's order, then the uncited ones,
#   tied), aggregated by Kemeny-Young (the ranking with the fewest pairwise disagreements, ties to retrieval order).
#   The answer cites the top k of it, k = the median number of cases the answers cite.
#   Articles: majority vote (self-consistency, Wang et al., ICLR 2023), kept if in more than half of the answers.


def majority(answers: Sequence[Sequence[str]]) -> List[str]:
    votes = Counter(x for a in answers for x in set(a))
    return [x for x, n in votes.items() if n > len(answers) / 2]


# Pairwise preferences of one answer: a before b if a is cited earlier, or cited while b is not
def preferences(cited: Sequence[str], case_ids: Sequence[str]) -> set:
    position = {cid: i for i, cid in enumerate(dict.fromkeys(c for c in cited if c in case_ids))}
    return {(a, b) for a in position for b in case_ids if b != a and position[a] < position.get(b, len(case_ids))}


def kemeny(case_ids: List[str], answers: Sequence[Sequence[str]]) -> List[str]:
    prefs = [preferences(a, case_ids) for a in answers]

    def disagreements(ranking):
        return sum((b, a) in p for p in prefs for a, b in combinations(ranking, 2))
    # permutations() yields in retrieval order first, and min() keeps the first of equal scores
    return list(min(permutations(case_ids), key=disagreements))


def vote(cases: List[Dict], parsed: List[Dict]) -> Dict:
    answers = [p["cited_case_ids"] for p in parsed]
    k = median_low(len(set(a)) for a in answers)
    ranking = kemeny([c["case_id"] for c in cases], answers)
    cited = ranking[:k]

    case_articles = {}
    for cid in cited:
        lists = [p["case_articles"][cid] for p in parsed if cid in p["case_articles"]]
        if lists:
            case_articles[cid] = sorted(majority(lists), key=str)

    out = {"cited_case_ids": cited, "case_articles": case_articles, "kemeny_ranking": ranking,
           "json_parse_ok": sum(p["json_parse_ok"] for p in parsed) > len(parsed) / 2}
    # Logged only under the predict_articles prompt, as in the frozen pipeline
    predicted = [p["predicted_articles"] for p in parsed if isinstance(p.get("predicted_articles"), list)]
    out["predicted_articles"] = sorted(majority(predicted), key=str) if predicted else None
    return out


class PermutationSelfConsistency(MitigationArm):
    name = "permutation"

    def generate(self, cases: List[Dict], answer_once: AnswerOnce) -> Tuple[Generation, Dict, Dict]:
        runs = []
        for order in permutations(range(len(cases))):
            generation, parsed = answer_once([cases[i] for i in order])
            runs.append((order, generation, parsed))

        ok = [(g, p) for _, g, p in runs if not g.error]
        errors = [g.error for _, g, _ in runs if g.error]
        totals = {k: sum(getattr(g, k) for _, g, _ in runs) for k in ("total_tokens", "prompt_tokens", "completion_tokens")}
        # The voted answer has no single text; the per-order texts are in the log
        generation = Generation(text=None, **totals,
                                error=None if ok else f"all {len(runs)} orders failed: {errors[0]}")
        parsed = vote(cases, [p for _, p in ok]) if ok else {"cited_case_ids": [], "case_articles": {},
                                                             "json_parse_ok": False}
        extra = {"kemeny_ranking": parsed.pop("kemeny_ranking", None),
                 "permutations": [{"order": [cases[i]["case_id"] for i in order], "response": g.text,
                                   "generation_error": g.error, "cited_case_ids": p["cited_case_ids"],
                                   "case_articles": p["case_articles"], "json_parse_ok": p["json_parse_ok"],
                                   "predicted_articles": p.get("predicted_articles")}
                                  for order, g, p in runs],
                 "orders_failed": len(errors)}
        return generation, parsed, extra
