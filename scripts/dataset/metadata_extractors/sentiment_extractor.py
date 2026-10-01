import argparse
import json

import nltk
import spacy
from nrclex import NRCLex

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        nltk.download('punkt_tab', quiet=True)
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def get_text(record):
    if "facts" in record:
        return ' '.join(record['facts'])
    return record['query_text']


def extract_sentiment(record):
    text = get_text(record)

    # NRC fear/anger/sadness/disgust frequencies, summed and capped at 1.0 into a single emotional-intensity score
    text_object = NRCLex(text)
    emotions = text_object.affect_frequencies
    fear = emotions.get('fear', 0)
    anger = emotions.get('anger', 0)
    sadness = emotions.get('sadness', 0)
    disgust = emotions.get('disgust', 0)

    sentence_count = len(list(get_nlp()(text).sents))

    return {
        'word_count': len(text.split()),
        'emotional_word_count': len(text_object.affect_dict),
        'nrc_fear': fear,
        'nrc_anger': anger,
        'nrc_sadness': sadness,
        'nrc_disgust': disgust,
        'nrc_emotional_intensity': min(fear + anger + sadness + disgust, 1.0),
        'sentence_count': sentence_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/train.jsonl")
    parser.add_argument("--output", default="dataset/metadata_jsonl/sentiment_analysis_JSONL_results/sentiment_extractions.jsonl")
    parser.add_argument("--id-fields", nargs="+", default=["case_id", "case_no"],
                         help="Fields copied from each input record into the output record")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in, start=1):
            record = json.loads(line)
            out = {k: record.get(k, '') for k in args.id_fields}
            out['sentiment_info'] = extract_sentiment(record)
            f_out.write(json.dumps(out) + '\n')
            if i % 1000 == 0:
                print(f"  {i} records scored")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
