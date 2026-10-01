import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

INPUT_FILE = "dataset/test.jsonl"
OUTPUT_FILE = "dataset/test_with_metadata.jsonl"
CHECKPOINT_FILE = "dataset/test_with_metadata.checkpoint.jsonl"

BASE_URL = "https://api.moonshot.ai/v1"
MODEL = "kimi-k3"
API_KEY_ENV = "MOONSHOT_API_KEY"

FACTS_INTRO_COUNT = 8
MAX_INTRO_CHARS = 4000
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 4

GENDER_VALUES = {"Male", "Female", "Multiple Applicants", "Unknown", "Needs Manual Classification"}
CONFIDENCE_VALUES = {"High", "Medium", "N/A"}

SYSTEM_PROMPT = """You classify the applicant(s) in a European Court of Human Rights case from the opening facts.
Return ONLY a JSON object, no prose, with exactly these keys:
- "gender": one of "Male", "Female", "Multiple Applicants", "Unknown", "Needs Manual Classification"
- "confidence": one of "High", "Medium", "N/A"

Rules:
- "Male"/"Female": a single natural-person applicant of that gender. Use "High" when the text is explicit (name, pronouns, "Mr"/"Ms"), "Medium" when inferred from weaker signals.
- "Multiple Applicants": more than one applicant, or an applicant group/organisation/company. confidence = "N/A".
- "Unknown": a single applicant whose gender cannot be determined. confidence = "N/A".
- "Needs Manual Classification": genuinely ambiguous cases needing a human. confidence = "N/A".
- Base everything only on the applicant, not third parties, victims, or officials mentioned."""


def build_user_prompt(case):
    facts = case.get("facts", [])
    intro = " ".join(facts[:FACTS_INTRO_COUNT])[:MAX_INTRO_CHARS]
    return f"Case facts (opening):\n{intro}"


def normalize_result(raw, case):
    gender = raw.get("gender")
    if gender not in GENDER_VALUES:
        gender = "Needs Manual Classification"

    confidence = raw.get("confidence")
    if gender in ("Male", "Female"):
        if confidence not in ("High", "Medium"):
            confidence = "Medium"
    else:
        confidence = "N/A"

    return {
        "classification": {"gender": gender, "confidence": confidence},
    }


def classify_case(client, case):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case)},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"},
            )
            raw = json.loads(resp.choices[0].message.content)
            return normalize_result(raw, case)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    ! giving up on {case.get('case_id')} after {MAX_RETRIES} tries: {e}")
                return normalize_result({}, case)
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"    retry {attempt}/{MAX_RETRIES} for {case.get('case_id')} in {wait}s ({e})")
            time.sleep(wait)


def load_checkpoint(path):
    done = {}
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["case_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N cases (smoke test)")
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"ERROR: set {API_KEY_ENV} in your environment before running.")
        print(f"  export {API_KEY_ENV}='your-moonshot-key'")
        return

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    cases = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    print(f"Loaded {len(cases)} cases from {INPUT_FILE}")

    if args.limit is not None:
        cases = cases[:args.limit]
        print(f"--limit set: processing only the first {len(cases)} cases")

    done = load_checkpoint(CHECKPOINT_FILE)
    if done:
        print(f"Resuming: {len(done)} cases already in checkpoint")

    start = time.time()
    checkpoint_out = open(CHECKPOINT_FILE, "a", encoding="utf-8")
    processed_this_run = 0

    for i, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        if case_id in done:
            continue

        meta = classify_case(client, case)
        record = {"case_id": case_id, **meta}
        done[case_id] = record
        checkpoint_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint_out.flush()
        processed_this_run += 1

        if processed_this_run % 25 == 0:
            elapsed = time.time() - start
            rate = processed_this_run / elapsed
            remaining = (len(cases) - len(done)) / rate if rate > 0 else 0
            print(f"  {len(done)}/{len(cases)} done "
                  f"({processed_this_run} this run, {rate:.2f}/s, ~{remaining/60:.1f} min left)")

    checkpoint_out.close()

    print(f"\nWriting merged output to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for case in cases:
            meta = done.get(case["case_id"], {})
            enriched = dict(case)
            enriched["classification"] = meta.get("classification", {"gender": "Needs Manual Classification", "confidence": "N/A"})
            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    print_summary(cases, done)


def print_summary(cases, done):
    genders = Counter()
    confidences = Counter()
    for case in cases:
        meta = done.get(case["case_id"], {})
        cls = meta.get("classification", {})
        genders[cls.get("gender", "MISSING")] += 1
        confidences[cls.get("confidence", "MISSING")] += 1

    n = len(cases)
    print("\n" + "=" * 60)
    print(f"METADATA EXTRACTION SUMMARY ({n} test cases, model={MODEL})")
    print("=" * 60)
    print("\nGender distribution:")
    for g, c in genders.most_common():
        print(f"  {g:<28} {c:>5} ({c/n*100:5.1f}%)")
    print("\nConfidence distribution:")
    for conf, c in confidences.most_common():
        print(f"  {conf:<28} {c:>5} ({c/n*100:5.1f}%)")
    print("=" * 60)
    print("\nNote: gender extracted via Kimi K2 API (train used Claude) — method differs from train; flag if comparing across splits.")


if __name__ == "__main__":
    main()
