import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from jsonschema import Draft202012Validator

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
MAX_SHOWN = 5


def load_schema(name):
    with open(os.path.join(SCHEMA_DIR, f"{name}.schema.json"), "r", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Reads a JSONL file row by row against its schema; returns the parsed rows and readable errors
def check_file(path, schema_name):
    validator, rows, errors = load_schema(schema_name), [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path} line {i}: not valid JSON ({e.msg})")
                continue
            for err in validator.iter_errors(row):
                where = "/".join(str(p) for p in err.absolute_path) or "row"
                errors.append(f"{path} line {i}, {where}: {err.message}")
            rows.append(row)
    if not rows and not errors:
        errors.append(f"{path}: file is empty")
    return rows, errors


BASELINE = "baseline"


def arms_key(row):
    return "+".join(sorted(row["arms"])) or BASELINE


def check_run_logs(rows):
    errors, warnings = [], []
    seen, queries_by_run = Counter(), defaultdict(set)
    for r in rows:
        seen[(arms_key(r), r["run_id"], r["query_id"])] += 1
        queries_by_run[(arms_key(r), r["run_id"])].add(r["query_id"])
        ranks = [d["rank"] for d in r["retrieved"]]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            errors.append(f"run {r['run_id']} query {r['query_id']}: ranks must be 1..{len(ranks)}, got {ranks}")
        ids = [d["doc_id"] for d in r["retrieved"]]
        if len(set(ids)) != len(ids):
            errors.append(f"run {r['run_id']} query {r['query_id']}: a document is retrieved twice")
    errors += [f"run {run} ({arms}) query {q}: logged {n} times" for (arms, run, q), n in seen.items() if n > 1]

    runs_per_arms = defaultdict(list)
    for (arms, run), qs in queries_by_run.items():
        runs_per_arms[arms].append((run, qs))
    for arms, runs in runs_per_arms.items():
        first_run, first = runs[0]
        for run, qs in runs[1:]:
            if qs != first:
                errors.append(f"{arms}: run {run} has {len(qs ^ first)} queries that run {first_run} does not share; "
                              "every repeat must run the same queries")
        if len(runs) < 2:
            warnings.append(f"{arms}: only one run, so the repeat-run noise floor cannot be measured (2+ runs needed, 4 recommended)")

    not_retrieved = sum(len(set(r["cited_doc_ids"]) - {d["doc_id"] for d in r["retrieved"]}) for r in rows)
    if not_retrieved:
        warnings.append(f"{not_retrieved} cited ids are not among the retrieved documents; they are ignored")
    return errors, warnings


def check_labels(rows, attribute):
    errors = []
    attributes = sorted({r["attribute"] for r in rows})
    if attribute is None:
        if len(attributes) > 1:
            return [f"labels hold several attributes {attributes}; choose one with --attribute"], None, {}
        attribute = attributes[0] if attributes else None
    if attribute not in attributes:
        return [f"no labels for attribute '{attribute}' (found {attributes})"], attribute, {}
    labels = {}
    for r in rows:
        if r["attribute"] != attribute:
            continue
        if r["doc_id"] in labels and labels[r["doc_id"]] != r["value"]:
            errors.append(f"document {r['doc_id']} has two {attribute} values: {labels[r['doc_id']]} and {r['value']}")
        labels[r["doc_id"]] = r["value"]
    return errors, attribute, labels


def check_pairs(rows, logged_queries):
    errors, warnings = [], []
    counts = Counter(r["query_id"] for r in rows)
    errors += [f"query {q} is listed {n} times" for q, n in counts.items() if n > 1]
    variants = defaultdict(list)
    for r in rows:
        if "pair_id" in r:
            variants[r["pair_id"]].append(r["variant"])
    for pair, vs in variants.items():
        if len(vs) < 2:
            errors.append(f"pair {pair} has only one query ({vs[0]}); a pair needs at least two variants")
        elif len(set(vs)) != len(vs):
            errors.append(f"pair {pair} lists the same variant twice: {vs}")
    missing = sorted(set(counts) - logged_queries)
    twin_ids = {r["query_id"] for r in rows if "pair_id" in r}
    if twin_ids and not twin_ids & logged_queries:
        errors.append(f"none of the {len(twin_ids)} twin queries is in the run logs (e.g. {sorted(twin_ids)[:3]}); "
                      "the query_id written by the logger must match the pairs file")
    elif missing:
        warnings.append(f"{len(missing)} queries in the pairs file are not in the run logs and are not used, e.g. {missing[:3]}")
    if not variants:
        warnings.append("no twin queries (pair_id), so the flip test cannot run")
    if not any("attribute_value" in r for r in rows):
        warnings.append("no query has attribute_value, so retrieval cannot be compared across query attributes")
    return errors, warnings, Counter(v for vs in variants.values() for v in vs)


def validate(run_log_paths, labels_path, pairs_path, attribute=None):
    errors, warnings, run_rows = [], [], []
    for path in run_log_paths:
        rows, errs = check_file(path, "run_log")
        run_rows += rows
        errors += errs
    label_rows, errs = check_file(labels_path, "doc_labels")
    errors += errs
    pair_rows, errs = check_file(pairs_path, "query_pairs")
    errors += errs
    if errors:
        return errors, warnings, {}

    errs, warns = check_run_logs(run_rows)
    errors += errs
    warnings += warns
    errs, attribute, labels = check_labels(label_rows, attribute)
    errors += errs
    if labels:
        unlabelled = {d["doc_id"] for r in run_rows for d in r["retrieved"]} - set(labels)
        if unlabelled:
            errors.append(f"{len(unlabelled)} retrieved documents have no {attribute} label, e.g. {sorted(unlabelled)[:3]}; "
                          "add them to the labels file (value Unknown if not known)")
    errs, warns, variant_counts = check_pairs(pair_rows, {r["query_id"] for r in run_rows})
    errors += errs
    warnings += warns

    summary = {
        "run_rows": len(run_rows),
        "runs": sorted({(arms_key(r), r["run_id"]) for r in run_rows}),
        "attribute": attribute,
        "documents": len(labels),
        "collection_mix": dict(Counter(labels.values()).most_common()),
        "queries": len(pair_rows),
        "twin_variants": dict(variant_counts),
    }
    return errors, warnings, summary


def main():
    parser = argparse.ArgumentParser(description="Check the three audit files before any statistics run.")
    parser.add_argument("--run-logs", nargs="+", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--attribute", default=None, help="Needed only if the labels file holds several attributes")
    args = parser.parse_args()

    errors, warnings, summary = validate(args.run_logs, args.labels, args.pairs, args.attribute)
    if errors:
        print(f"NOT VALID: {len(errors)} problem(s)")
        for e in errors[:MAX_SHOWN * 4]:
            print(f"  - {e}")
        if len(errors) > MAX_SHOWN * 4:
            print(f"  ... and {len(errors) - MAX_SHOWN * 4} more")
        sys.exit(1)

    print("VALID")
    print(f"  run log rows: {summary['run_rows']}")
    by_arms = defaultdict(list)
    for arms, run in summary["runs"]:
        by_arms[arms].append(run)
    for arms, runs in by_arms.items():
        print(f"  {arms}: runs {', '.join(runs)}")
    print(f"  attribute: {summary['attribute']}, {summary['documents']} documents, mix {summary['collection_mix']}")
    print(f"  queries: {summary['queries']}, twin variants {summary['twin_variants']}")
    for w in warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
