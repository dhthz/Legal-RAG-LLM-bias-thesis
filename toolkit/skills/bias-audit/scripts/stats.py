import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import binomtest, chi2_contingency, chisquare, wilcoxon

from validate import BASELINE, arms_key, load_jsonl

SIMILARITY_GROUPS = ("different order", "same order, different passages shown", "identical input")


# Cramer's V for a goodness-of-fit test (1 x k table): V = sqrt(chi2 / (n*(k-1)))
def cramers_v_gof(chi2_stat, n, k):
    return (chi2_stat / (n * (k - 1))) ** 0.5


# Wilson score CI for a binary proportion
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


# Holm-Bonferroni across a named p-value family: sort ascending, multiply the
# i-th smallest by (m - i), enforce monotonicity, cap at 1.
def holm_bonferroni(named_pvals):
    m = len(named_pvals)
    ordered = sorted(named_pvals.items(), key=lambda kv: kv[1])
    corrected = {}
    running_max = 0.0
    for i, (name, p) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - i) * p))
        corrected[name] = running_max
    return corrected


def load(run_log_paths, labels_path, pairs_path, attribute):
    rows = [r for p in run_log_paths for r in load_jsonl(p)]
    labels = {r["doc_id"]: r["value"] for r in load_jsonl(labels_path) if r["attribute"] == attribute}
    queries = {r["query_id"]: r for r in load_jsonl(pairs_path)}
    twins = defaultdict(dict)
    for q in queries.values():
        if "pair_id" in q:
            twins[q["pair_id"]][q["variant"]] = q["query_id"]
    by_arms = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_arms[arms_key(r)][r["run_id"]][r["query_id"]] = r
    by_arms = {a: {run: runs[run] for run in sorted(runs)} for a, runs in by_arms.items()}
    return by_arms, labels, queries, dict(twins)


def cited(row):
    return set(row["cited_doc_ids"])


def retrieved_set(row):
    return {d["doc_id"] for d in row["retrieved"]}


# What the LLM actually read: document order plus the passages shown per document
def model_input(row, with_shown=True):
    docs = sorted(row["retrieved"], key=lambda d: d["rank"])
    return [(d["doc_id"],) + (tuple(d.get("shown_ids", ())) if with_shown else ()) for d in docs]


def similarity_group(a, b):
    if model_input(a) == model_input(b):
        return "identical input"
    if model_input(a, False) == model_input(b, False):
        return "same order, different passages shown"
    return "different order"


def chi2_two_rows(first, second):
    classes = sorted(v for v in set(first) | set(second) if first.get(v, 0) + second.get(v, 0) > 0)
    table = [[first.get(v, 0) for v in classes], [second.get(v, 0) for v in classes]]
    chi2, p, dof, expected = chi2_contingency(table)
    n = sum(sum(row) for row in table)
    return {"n": n, "dof": int(dof), "statistic": float(chi2), "p_raw": float(p), "effect_size_name": "Cramer's V",
            "effect_size": float((chi2 / n) ** 0.5), "min_expected": float(expected.min())}


# Retrieval side: does the attribute mix of everything retrieved match the whole collection?
def attribute_vs_collection(rows, labels):
    collection = Counter(labels.values())
    values = [v for v in sorted(collection) if collection[v] > 0]
    retrieved = Counter(labels[d["doc_id"]] for r in rows for d in r["retrieved"])
    n, n_collection = sum(retrieved.values()), sum(collection.values())
    observed = [retrieved.get(v, 0) for v in values]
    expected = [collection[v] / n_collection * n for v in values]
    chi2, p = chisquare(f_obs=observed, f_exp=expected)
    return {
        "test": "retrieved attribute vs collection",
        "test_name": "chi-square goodness-of-fit",
        "n": n,
        "observed": dict(zip(values, observed)),
        "expected": dict(zip(values, [round(x, 1) for x in expected])),
        "statistic": float(chi2),
        "p_raw": float(p),
        "effect_size_name": "Cramer's V",
        "effect_size": float(cramers_v_gof(chi2, n, len(values))),
        "min_expected": float(min(expected)),
    }


# Retrieval side: do queries about group A and group B retrieve a different attribute mix?
# Twin queries are left out: they are rewrites of the same base, not independent queries.
def retrieved_by_query_attribute(rows, labels, queries, groups):
    dist = defaultdict(Counter)
    for r in rows:
        q = queries.get(r["query_id"], {})
        if "pair_id" in q or q.get("attribute_value") not in groups:
            continue
        for d in r["retrieved"]:
            dist[q["attribute_value"]][labels[d["doc_id"]]] += 1
    if not all(dist[g] for g in groups):
        return None
    res = chi2_two_rows(dist[groups[0]], dist[groups[1]])
    return {"test": "retrieved attribute by query attribute", "test_name": "chi-square test of independence (2 x k)",
            "groups": list(groups), **res, "distributions": {g: dict(dist[g]) for g in groups}}


# Retrieval side: does the top document change between two twins? McNemar counts the pairs where the
# top document belongs to the watched group under one wording only.
def flip_test(run, labels, twins, va, vb, watched):
    flips, kept, a_only, b_only = [], [], 0, 0
    for pair, variants in sorted(twins.items()):
        if va not in variants or vb not in variants:
            continue
        top_a = min(run[variants[va]]["retrieved"], key=lambda d: d["rank"])["doc_id"]
        top_b = min(run[variants[vb]]["retrieved"], key=lambda d: d["rank"])["doc_id"]
        kept.append(pair)
        if top_a != top_b:
            flips.append(pair)
        a_w, b_w = labels[top_a] == watched, labels[top_b] == watched
        a_only += a_w and not b_w
        b_only += b_w and not a_w
    n, k = len(kept), len(flips)
    discordant = a_only + b_only
    return {
        "test": f"top document flip, {va} vs {vb}",
        "n_pairs": n,
        "top1_flips": k,
        "flip_rate": k / n if n else None,
        "flip_rate_wilson_95ci": list(wilson_ci(k, n)),
        "flipped_pairs": flips,
        "mcnemar_outcome": f"top document is {watched}",
        "mcnemar_discordant": {f"{va}_only": a_only, f"{vb}_only": b_only},
        "p_raw": float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else None,
    }


# Generation side: among queries with at least one citation, is the attribute mix of cited documents
# different from retrieved-but-not-cited ones? A difference is bias added by the LLM's choice.
def cited_vs_not_cited(rows, labels):
    yes, no, n_queries = Counter(), Counter(), 0
    for r in rows:
        if not r["cited_doc_ids"]:
            continue
        n_queries += 1
        for d in r["retrieved"]:
            (yes if d["doc_id"] in cited(r) else no)[labels[d["doc_id"]]] += 1
    if not n_queries:
        return None
    return {"test": "cited vs not cited attribute", "test_name": "chi-square test of independence (2 x k)",
            "n_queries_with_citations": n_queries, **chi2_two_rows(yes, no), "cited": dict(yes), "not_cited": dict(no)}


def retrieval_identical(runs):
    first = next(iter(runs.values()))
    same = sum(all(model_input(rows[q]) == model_input(first[q]) for rows in runs.values()) for q in first)
    return {"identical_queries": same, "of": len(first)}


# Same twin query, two runs: how often the cited set changes with nothing changed
def noise_floor(runs, twin_ids):
    if len(runs) < 2:
        return None
    pairs, rates = {}, []
    for r1, r2 in combinations(runs, 2):
        c = sum(cited(runs[r1][q]) != cited(runs[r2][q]) for q in twin_ids)
        pairs[f"{r1}-{r2}"] = {"cited_differs": c, "n": len(twin_ids)}
        rates.append(c / len(twin_ids))
    return {"run_pairs": pairs, "cited_rate_mean": float(np.mean(rates)), "cited_rate_range": [min(rates), max(rates)]}


# Twins with the same retrieved set, split by how similar the LLM's input really was. The input depends
# only on retrieval, so the group of a pair comes from the first run.
def input_similarity_split(runs, twins, va, vb):
    first = next(iter(runs.values()))
    groups = defaultdict(list)
    for pair, variants in sorted(twins.items()):
        if va not in variants or vb not in variants:
            continue
        a_row, b_row = first[variants[va]], first[variants[vb]]
        if retrieved_set(a_row) == retrieved_set(b_row):
            groups[similarity_group(a_row, b_row)].append(pair)

    out = {}
    for g in SIMILARITY_GROUPS:
        pairs = groups.get(g, [])
        if not pairs:
            out[g] = {"n_pairs": 0}
            continue
        wording, repeat = [], []
        for pair in pairs:
            qa, qb = twins[pair][va], twins[pair][vb]
            wording.append(np.mean([cited(rows[qa]) != cited(rows[qb]) for rows in runs.values()]))
            if len(runs) > 1:
                repeat.append(np.mean([cited(runs[r1][q]) != cited(runs[r2][q])
                                       for q in (qa, qb) for r1, r2 in combinations(runs, 2)]))
        rows_differ = int(round(sum(wording) * len(runs)))
        diffs = [w - r for w, r in zip(wording, repeat)]
        out[g] = {
            "n_pairs": len(pairs),
            "pairs": pairs,
            "rows_differ": rows_differ,
            "rows": len(pairs) * len(runs),
            "wording_change_rate": float(np.mean(wording)),
            "wording_change_wilson_95ci": list(wilson_ci(rows_differ, len(pairs) * len(runs))),
            "same_pairs_repeat_noise": float(np.mean(repeat)) if repeat else None,
            "p_wilcoxon_vs_repeat_noise": float(wilcoxon(diffs).pvalue) if len(pairs) > 1 and any(diffs) else None,
        }
    return out


def first_cited(row):
    return row["cited_doc_ids"][0] if row["cited_doc_ids"] else None


# Whole system, fix vs baseline: how often the twins' answers cite different documents (every run counts as a
# row). Each pair's change rate over the runs is compared with its baseline rate (Wilcoxon signed-rank).
def twin_answer_change(base_runs, arm_runs, twins, va, vb):
    every_run = list(base_runs.values()) + list(arm_runs.values())
    pairs = [v for _, v in sorted(twins.items()) if va in v and vb in v and all(v[va] in r and v[vb] in r for r in every_run)]
    out = {}
    for name, key in (("cited_set", cited), ("first_cited", first_cited)):
        base_rates, arm_rates = [], []
        for v in pairs:
            base_rates.append(np.mean([key(rows[v[va]]) != key(rows[v[vb]]) for rows in base_runs.values()]))
            arm_rates.append(np.mean([key(rows[v[va]]) != key(rows[v[vb]]) for rows in arm_runs.values()]))
        k_arm = int(round(sum(arm_rates) * len(arm_runs)))
        diffs = [a - b for a, b in zip(arm_rates, base_rates)]
        out[name] = {
            "rows_differ": k_arm,
            "rows": len(pairs) * len(arm_runs),
            "rate": k_arm / (len(pairs) * len(arm_runs)) if pairs else None,
            "wilson_95ci": list(wilson_ci(k_arm, len(pairs) * len(arm_runs))),
            "baseline_rows_differ": int(round(sum(base_rates) * len(base_runs))),
            "baseline_rows": len(pairs) * len(base_runs),
            "pairs_lower": int(sum(d < 0 for d in diffs)),
            "pairs_higher": int(sum(d > 0 for d in diffs)),
            "p_wilcoxon_vs_baseline": float(wilcoxon(diffs).pvalue) if any(diffs) else None,
        }
    return out


# Retrieval, fix vs baseline: the documents the fix dropped from each query's results (left) and the ones it brought
# in (entered). Left more often matching the query's own attribute than entered = the fix removed attribute-driven
# retrievals rather than just reshuffling. Twin queries are left out, as in the by-query-attribute test.
def changed_docs(base_rows, arm_rows, labels, queries, groups):
    left, entered, overlap = [], [], []
    for qid, arm in arm_rows.items():
        q = queries.get(qid, {})
        if qid not in base_rows or "pair_id" in q:
            continue
        ids_base, ids_arm = retrieved_set(base_rows[qid]), retrieved_set(arm)
        left += [(q, d) for d in ids_base - ids_arm]
        entered += [(q, d) for d in ids_arm - ids_base]
        overlap.append(len(ids_base & ids_arm) / len(ids_arm))

    def same_attribute(changed):
        hits = [labels[d] == q["attribute_value"] for q, d in changed if q.get("attribute_value") in groups]
        return float(np.mean(hits)) if hits else None

    return {"queries": len(overlap), "retrieved_overlap_with_baseline": float(np.mean(overlap)) if overlap else None,
            "docs_changed": len(left), "left_same_attribute_share": same_attribute(left),
            "entered_same_attribute_share": same_attribute(entered)}


def compare_to_baseline(base_runs, arm_runs, labels, queries, twins, groups, twin_pairs):
    return {
        "twin_answer_change": {f"{va}_vs_{vb}": twin_answer_change(base_runs, arm_runs, twins, va, vb) for va, vb in twin_pairs},
        "changed_docs": changed_docs(next(iter(base_runs.values())), next(iter(arm_runs.values())), labels, queries, groups),
    }


# All tests for the runs of one arms setting. The Holm family is this audit's own: the collection test,
# the query-attribute test, the first twin comparison's McNemar and the generation test of the first run.
def audit_arms(runs, labels, queries, twins, groups, twin_pairs):
    first = next(iter(runs.values()))
    first_rows = list(first.values())
    # Only the twins this setting actually ran (every run of a setting has the same queries)
    twins = {p: {v: q for v, q in vs.items() if q in first} for p, vs in twins.items()}
    twins = {p: vs for p, vs in twins.items() if len(vs) > 1}
    twin_ids = [q for variants in twins.values() for q in variants.values()]
    res = {
        "runs": list(runs),
        "queries_per_run": len(first),
        "retrieval_identical_across_runs": retrieval_identical(runs),
        "attribute_vs_collection": attribute_vs_collection(first_rows, labels),
        "retrieved_by_query_attribute": retrieved_by_query_attribute(first_rows, labels, queries, groups),
        "flip_tests": {f"{va}_vs_{vb}": flip_test(first, labels, twins, va, vb, groups[1]) for va, vb in twin_pairs},
        "cited_vs_not_cited_per_run": {run: cited_vs_not_cited(list(rows.values()), labels) for run, rows in runs.items()},
        "noise_floor_twin_rows": noise_floor(runs, twin_ids),
        "input_similarity_split": {f"{va}_vs_{vb}": input_similarity_split(runs, twins, va, vb) for va, vb in twin_pairs},
    }
    family = {"attribute_vs_collection": res["attribute_vs_collection"]["p_raw"]}
    if res["retrieved_by_query_attribute"]:
        family["retrieved_by_query_attribute"] = res["retrieved_by_query_attribute"]["p_raw"]
    if twin_pairs and res["flip_tests"][f"{twin_pairs[0][0]}_vs_{twin_pairs[0][1]}"]["p_raw"] is not None:
        family["flip_mcnemar"] = res["flip_tests"][f"{twin_pairs[0][0]}_vs_{twin_pairs[0][1]}"]["p_raw"]
    first_gen = res["cited_vs_not_cited_per_run"][res["runs"][0]]
    if first_gen:
        family["cited_vs_not_cited"] = first_gen["p_raw"]
    res["holm_bonferroni"] = {"family_size": len(family), "p_raw": family, "p_corrected": holm_bonferroni(family)}
    return res
