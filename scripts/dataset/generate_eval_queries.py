import json
import random
import re
from pathlib import Path

INPUT_FILE = "dataset/train_splits/train_split_10.jsonl"
OUTPUT_FILE = "dataset/eval/retrieval_eval_queries.jsonl"
NUM_QUERIES = 100
SEED = 42
MIN_FACTS_WORDS = 50
MAX_QUERY_WORDS = 6000

TITLE_STOPWORDS = {"AND", "OTHERS", "THE", "OF", "V"}
GENERIC_TOKENS = TITLE_STOPWORDS | {
    "COMPANY", "COMPANIES", "SHIPPING", "LIMITED", "LTD", "GROUP",
    "HOLDING", "HOLDINGS", "TRADING", "ASSOCIATION", "FOUNDATION",
    "PARTY", "UNION", "NATIONAL", "INTERNATIONAL",
}


# Extract applicant names/parties from case title (e.g. "CASE OF X AND Y v. Z" → [X, Y])
def parse_title_phrases(title):
    m = re.match(r"CASE OF (.+?) v\.? .+", title, re.IGNORECASE)
    if not m:
        return []
    return [p.strip() for p in re.split(r"\s+AND\s+", m.group(1)) if p.strip()]


# Extract individual name tokens (surnames, first names) from multi-word party phrases, filter out generic corporate tokens
def name_tokens(phrases):
    tokens = set()
    for phrase in phrases:
        for word in re.split(r"[\s,]+", phrase):
            cleaned = word.strip(".()\"'").replace(".", "")
            if len(cleaned) >= 3 and cleaned.upper() not in GENERIC_TOKENS:
                tokens.add(cleaned)
    return tokens


# Remove all applicant/party names from query text to prevent leakage. Multi-pass: first full phrases, then individual tokens, then collapse duplicates and fix artifacts
def strip_names(text, phrases, case_no):
    multiword = [p for p in set(phrases) if len(p.split()) > 1]
    for phrase in sorted(multiword, key=len, reverse=True):
        pattern = re.compile(
            r"(?:\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+)?\b(?i:"
            + r"\s+".join(re.escape(w) for w in phrase.split()) + r")\b"
        )
        text = pattern.sub("the applicant", text)

    for token in sorted(name_tokens(phrases), key=len, reverse=True):
        pattern = re.compile(
            r"(?:\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+)?(?:\b[A-Z][\w’’-]*\s+){0,3}\b(?i:"
            + re.escape(token) + r")\b"
        )
        text = pattern.sub("the applicant", text)

    text = re.sub(r"(?i)\b(the applicant(?:[‘’]s)?)(?:[\s,]+the applicant(?:[‘’]s)?)+",
                  r"\1", text)
    text = re.sub(r"(?i)\bthe\s+(the applicant)\b", r"\1", text)
    if case_no:
        text = text.replace(case_no, "")
    return text


# Build a single query from a case: concatenate all facts, extract applicant names/phrases, strip them from text, normalize whitespace, cap length
def build_query(record):
    paragraphs = [re.sub(r"^\s*\d+\.\s*", "", p).strip() for p in record["facts"]]
    text = " ".join(p for p in paragraphs if p)

    phrases = [a.strip() for a in record.get("applicants", []) if a.strip()]
    phrases += parse_title_phrases(record.get("title", ""))

    text = strip_names(text, phrases, record.get("case_no", ""))
    text = re.sub(r"\s{2,}", " ", text).strip()

    words = text.split()
    if len(words) > MAX_QUERY_WORDS:
        text = " ".join(words[:MAX_QUERY_WORDS])
    return text, sorted(set(phrases))


# Generate 100 deterministic eval queries (seed=42) from train_split_10: clean data, strip names, verify no leakage, write frozen JSONL with traceability metadata
def main():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} cases from {INPUT_FILE}")

    eligible = []
    for r in records:
        total_words = sum(len(p.split()) for p in r.get("facts", []))
        if total_words >= MIN_FACTS_WORDS:
            eligible.append(r)
    print(f"Eligible cases (facts >= {MIN_FACTS_WORDS} words): {len(eligible)}")

    rng = random.Random(SEED)
    sampled = rng.sample(eligible, NUM_QUERIES)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_lengths = []
    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(sampled, start=1):
            query_text, stripped_names = build_query(record)
            query_lengths.append(len(query_text.split()))
            entry = {
                "query_id": f"q{i:03d}",
                "source_case_id": record["case_id"],
                "query_text": query_text,
                "title": record["title"],
                "judgment_date": record["judgment_date"],
                "violated_articles": record.get("violated_articles", []),
                "defendants": record.get("defendants", []),
                "stripped_names": stripped_names,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    leaks = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            for token in name_tokens(q["stripped_names"]):
                if re.search(r"\b" + re.escape(token) + r"\b", q["query_text"], re.IGNORECASE):
                    leaks.append((q["query_id"], token))
    if leaks:
        print(f"\nWARNING: {len(leaks)} name leaks detected:")
        for query_id, token in leaks:
            print(f"  {query_id}: '{token}'")
    else:
        print("\nLeakage check passed: no applicant name tokens remain in any query")

    print(f"\nWrote {NUM_QUERIES} queries to {OUTPUT_FILE} (seed={SEED})")
    print(f"Query length (words): min={min(query_lengths)}, "
          f"max={max(query_lengths)}, mean={sum(query_lengths)/len(query_lengths):.0f}")

    print("\nSample queries (first 60 words each):")
    with open(output_path, "r", encoding="utf-8") as f:
        for line in list(f)[:3]:
            q = json.loads(line)
            preview = " ".join(q["query_text"].split()[:60])
            print(f"\n[{q['query_id']}] gt={q['source_case_id']} ({q['title']})")
            print(f"  {preview}...")


if __name__ == "__main__":
    main()
