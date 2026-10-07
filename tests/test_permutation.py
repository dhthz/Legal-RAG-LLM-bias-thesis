import sys

from src.llm.client import Generation
from src.llm.prompts import build_legal_prompt
from src.mitigation import MitigationArm, build_arms
from src.mitigation.permutation import vote

# Arm 3 checks without the LLM: the first order is the frozen prompt byte for byte, every order of the cases is
# answered exactly once, the default hook answers once in retrieved order, and the vote follows its rule
QUERY = "The applicant complained that he had been detained without judicial review."
CASES = [{"case_id": f"001-{i}", "title": f"CASE OF {name} v. STATE", "judgment_date": "2010-01-01",
          "violated_articles": [str(a)], "avg_similarity": 0.5, "chunks": [{"chunk_text": f"Facts of case {i}."}]}
         for i, (name, a) in enumerate([("A", 5), ("B", 6), ("C", 8)], start=1)]


def recorder():
    prompts = []

    def answer_once(ordered):
        prompts.append(build_legal_prompt(QUERY, ordered, predict_articles=True))
        return Generation(text="x"), {"cited_case_ids": [ordered[0]["case_id"]], "case_articles": {},
                                      "json_parse_ok": True, "predicted_articles": ["5"]}
    return prompts, answer_once


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    return ok


def main():
    frozen = build_legal_prompt(QUERY, CASES, predict_articles=True)
    ok = True

    prompts, answer_once = recorder()
    _, parsed, extra = build_arms(["permutation"])[0].generate(CASES, answer_once)
    ok &= check("first order is the frozen prompt, byte for byte", prompts[0] == frozen)
    ok &= check("6 distinct orders answered", len(prompts) == 6 and len({p[1]["content"] for p in prompts}) == 6)
    ok &= check("each order is a permutation of the retrieved cases",
                all(sorted(o["order"]) == [c["case_id"] for c in CASES] for o in extra["permutations"]))
    # Each case is cited (first) in 2 of 6 orders: the Kemeny scores tie, so the consensus is the retrieval order
    ok &= check("pure position bias: consensus = retrieval order, cites 1", parsed["cited_case_ids"] == ["001-1"]
                and extra["kemeny_ranking"] == ["001-1", "001-2", "001-3"])
    ok &= check("unanimous article kept", parsed["predicted_articles"] == ["5"])

    prompts, answer_once = recorder()
    MitigationArm().generate(CASES, answer_once)
    ok &= check("default hook: one answer, retrieved order", prompts == [frozen])

    answers = [{"cited_case_ids": c, "case_articles": {"001-2": a}, "json_parse_ok": True, "predicted_articles": a}
               for c, a in [(["001-2", "001-3"], ["6"]), (["001-2"], ["6", "8"]), (["001-3", "001-2"], ["6"]),
                            (["001-3"], ["8"]), (["001-2"], ["6"]), (["001-1", "001-3"], ["6", "8"])]]
    v = vote(CASES, answers)
    # 2 vs 3 split 3:3, both beat 1 (4:1 and 3:1): (2,3,1) and (3,2,1) tie at 5 disagreements -> retrieval order
    ok &= check("Kemeny ranking, ties to retrieval order", v["kemeny_ranking"] == ["001-2", "001-3", "001-1"])
    ok &= check("cites the median number of cases (sizes 2,1,2,1,1,2 -> 1)", v["cited_case_ids"] == ["001-2"])
    ok &= check("articles kept if in > half", v["predicted_articles"] == ["6"])  # 6: 5/6, 8: 3/6

    unanimous = [{"cited_case_ids": ["001-3", "001-1"], "case_articles": {}, "json_parse_ok": True}] * 6
    ok &= check("unanimous answer kept in its own order", vote(CASES, unanimous)["cited_case_ids"] == ["001-3", "001-1"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
