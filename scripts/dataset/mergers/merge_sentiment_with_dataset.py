import json

# Load sentiment extractions into a dictionary
sentiment_data = {}
with open('Metadata Extraction Files/sentiment_analysis_JSONL_results/sentiment_extractions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sentiment_case = json.loads(line)
        # Use both case_id and case_no as keys
        case_id = sentiment_case.get('case_id')
        case_no = sentiment_case.get('case_no')

        # Store sentiment_info
        if case_id:
            sentiment_data[case_id] = sentiment_case.get('sentiment_info', {})
        if case_no:
            sentiment_data[case_no] = sentiment_case.get('sentiment_info', {})

print(f"Loaded {len(sentiment_data)} sentiment records")

# Merge with train_with_gender_and_age.jsonl
merged_count = 0
with open('dataset/train_with_gender_and_age.jsonl', 'r', encoding='utf-8') as f_in, \
     open('dataset/train_with_metadata.jsonl', 'w', encoding='utf-8') as f_out:

    for line in f_in:
        case = json.loads(line)

        # Try to find sentiment_info by case_id or case_no
        case_id = case.get('case_id')
        case_no = case.get('case_no')

        sentiment_info = None
        if case_id in sentiment_data:
            sentiment_info = sentiment_data[case_id]
            merged_count += 1
        elif case_no in sentiment_data:
            sentiment_info = sentiment_data[case_no]
            merged_count += 1

        # Add sentiment_info to case
        if sentiment_info:
            case['sentiment_info'] = sentiment_info
        else:
            case['sentiment_info'] = {}  # Empty dict if no sentiment data

        # Write merged case
        f_out.write(json.dumps(case) + '\n')

print(f"\n Merged {merged_count} cases with sentiment data")
print(f" Created: dataset/train_with_metadata.jsonl")
print(f"\nThis file now contains:")
print(f"  - Original case data (facts, violated_articles, etc.)")
print(f"  - Gender classification")
print(f"  - Age information")
print(f"  - Sentiment analysis metadata")
