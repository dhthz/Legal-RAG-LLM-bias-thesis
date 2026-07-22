from nrclex import NRCLex
import spacy
from empath import Empath #probably wont be used
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #probably wont be used
import json
import pandas as pd
import nltk # For NRCLex
import spacy

FIRST_PERSON = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
THIRD_PERSON = ['he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their']
PERPETRATOR_WORDS = ['police', 'officer', 'officers', 'state', 'authorities',
                     'government', 'official', 'officials', 'detention', 'custody']
VICTIM_WORDS = ['applicant', 'victim', 'suffered', 'subjected', 'endured']

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        nltk.download('punkt_tab')
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_sentiment(case):
    nlp = get_nlp()
    facts = ' '.join(case['facts'])

    # NRCLex analysis for fear,anger,sadness & disgust, these results are aggregated to turn into a single normalised variable named emotional_intensity
    text_object = NRCLex(facts)
    nrc_emotions = text_object.affect_frequencies
    nrc_fear = nrc_emotions.get('fear', 0)
    nrc_anger = nrc_emotions.get('anger', 0)
    nrc_sadness = nrc_emotions.get('sadness', 0)
    nrc_disgust = nrc_emotions.get('disgust', 0)

    nrc_emotional_intensity = min(nrc_fear + nrc_anger + nrc_sadness + nrc_disgust, 1.0)

    doc = nlp(facts)
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    passive_count = 0

    for token in doc:
        if token.dep_ == 'nsubjpass':  # passive nominal subject
            passive_count += 1
    passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0

    first_person_count = 0
    third_person_count = 0

    for token in doc:
        if token.pos_ == 'PRON':
            text_lower = token.text.lower()
            if text_lower in FIRST_PERSON:
                first_person_count += 1
            elif text_lower in THIRD_PERSON:
                third_person_count += 1

    perpetrator_count = 0
    for token in doc:
        if token.text.lower() in PERPETRATOR_WORDS:
            perpetrator_count += 1

    victim_count = 0
    for token in doc:
        if token.lemma_.lower() in VICTIM_WORDS:
            victim_count += 1

    return {
        'case_id': case['case_id'],
        'case_no': case.get('case_no', ''),
        'sentiment_info': {
            'word_count': len(facts.split()),
            'emotional_word_count': len(text_object.affect_dict),
            'nrc_fear': nrc_fear,
            'nrc_anger': nrc_anger,
            'nrc_sadness': nrc_sadness,
            'nrc_disgust': nrc_disgust,
            'nrc_emotional_intensity': nrc_emotional_intensity,
            'sentence_count': sentence_count,
            'passive_voice_count': passive_count,
            'passive_voice_ratio': passive_ratio,
            'first_person_pronouns': first_person_count,
            'third_person_pronouns': third_person_count,
            'perpetrator_mentions': perpetrator_count,
            'victim_language_count': victim_count
        }
    }


def main():
    results = []
    with open('dataset/train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            results.append(extract_sentiment(case))

    # Write results to JSONL file (same format as age_extractions.jsonl)
    with open('Metadata Extraction Files/sentiment_analysis_JSONL_results/sentiment_extractions.jsonl', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Also create summary statistics CSV for analysis
    df = pd.DataFrame([{
        'case_id': r['case_id'],
        'case_no': r['case_no'],
        **r['sentiment_info']
    } for r in results])

    df.to_csv('sentiment_metadata_sample.csv', index=False)


if __name__ == "__main__":
    main()
