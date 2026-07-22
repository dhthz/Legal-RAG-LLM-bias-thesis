import argparse
import json

CONFIGS = {
    "train": {
        "sentiment_file": "dataset/metadata_jsonl/sentiment_analysis_JSONL_results/sentiment_extractions.jsonl",
        "input_file": "dataset/train_with_gender_and_age.jsonl",
        "output_file": "dataset/train_with_metadata.jsonl",
        "key_fields": ("case_id", "case_no"),
    },
    "llm_queries": {
        "sentiment_file": "dataset/eval/llm_queries_metadata.jsonl",
        "input_file": "dataset/eval/llm_queries.jsonl",
        "output_file": "dataset/eval/llm_queries_enriched.jsonl",
        "key_fields": ("query_id",),
    },
}


def load_sentiment_data(sentiment_file, key_fields):
    sentiment_data = {}
    with open(sentiment_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            info = record.get("sentiment_info", {})
            for key_field in key_fields:
                key = record.get(key_field)
                if key:
                    sentiment_data[key] = info
    return sentiment_data


def merge(config):
    sentiment_data = load_sentiment_data(config["sentiment_file"], config["key_fields"])
    print(f"Loaded {len(sentiment_data)} sentiment records from {config['sentiment_file']}")

    merged_count = 0
    with open(config["input_file"], "r", encoding="utf-8") as f_in, \
         open(config["output_file"], "w", encoding="utf-8") as f_out:

        for line in f_in:
            case = json.loads(line)

            sentiment_info = None
            for key_field in config["key_fields"]:
                key = case.get(key_field)
                if key in sentiment_data:
                    sentiment_info = sentiment_data[key]
                    merged_count += 1
                    break

            case["sentiment_info"] = sentiment_info if sentiment_info else {}
            f_out.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\nMerged {merged_count} cases with sentiment data")
    print(f"Created: {config['output_file']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIGS.keys(), default="train", nargs="?",
                        help="Which dataset to merge sentiment into (default: train)")
    args = parser.parse_args()
    merge(CONFIGS[args.dataset])


if __name__ == "__main__":
    main()
