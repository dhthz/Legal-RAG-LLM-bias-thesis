import argparse
import csv
import json

# The per-case metadata this work added to the ECtHR cases dataset, as one shareable file:
#   gender, confidence  applicant gender (Claude Sonnet 4.5 labels; confidence "human" = labelled by hand)
#   NRC fields          from sentiment_extractor.py (NRC Emotion Lexicon, Mohammad & Turney 2013)
# export: dataset/train_with_metadata.jsonl -> docs/data/case_metadata.csv
# merge:  the downloaded train.jsonl + docs/data/case_metadata.csv -> dataset/train_with_metadata.jsonl

METADATA_CSV = "docs/data/case_metadata.csv"
NRC_FIELDS = ["word_count", "emotional_word_count", "nrc_fear", "nrc_anger", "nrc_sadness", "nrc_disgust",
              "nrc_emotional_intensity", "sentence_count"]
INT_FIELDS = {"word_count", "emotional_word_count", "sentence_count"}
COLUMNS = ["case_id", "gender", "confidence"] + NRC_FIELDS


def export(source, out):
    with open(source, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(COLUMNS)
        n = 0
        for line in f_in:
            case = json.loads(line)
            nrc = case["sentiment_info"]
            writer.writerow([case["case_id"], case["classification"]["gender"], case["classification"]["confidence"]]
                            + [repr(nrc[k]) for k in NRC_FIELDS])
            n += 1
    print(f"Saved: {out} ({n} cases)")


def load_metadata(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def merge(dataset, metadata, out):
    rows = load_metadata(metadata)
    n = 0
    with open(dataset, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            case = json.loads(line)
            row = rows.pop(case["case_id"])
            case["classification"] = {"gender": row["gender"], "confidence": row["confidence"]}
            case["sentiment_info"] = {k: int(row[k]) if k in INT_FIELDS else float(row[k]) for k in NRC_FIELDS}
            f_out.write(json.dumps(case, ensure_ascii=False) + "\n")
            n += 1
    if rows:
        raise ValueError(f"{len(rows)} metadata rows match no case in {dataset} (is it the training split?)")
    print(f"Saved: {out} ({n} cases)")


def main():
    parser = argparse.ArgumentParser(description="Export or merge the per-case gender and NRC metadata.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export")
    p_export.add_argument("--source", default="dataset/train_with_metadata.jsonl")
    p_export.add_argument("--out", default=METADATA_CSV)
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--dataset", default="dataset/train.jsonl")
    p_merge.add_argument("--metadata", default=METADATA_CSV)
    p_merge.add_argument("--out", default="dataset/train_with_metadata.jsonl")
    args = parser.parse_args()
    if args.command == "export":
        export(args.source, args.out)
    else:
        merge(args.dataset, args.metadata, args.out)


if __name__ == "__main__":
    main()
