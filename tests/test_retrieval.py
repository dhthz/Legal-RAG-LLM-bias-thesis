
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.embeddings.chunk_embedder import ChunkRetriever
from pathlib import Path
import json
from typing import List, Optional


# ======================
# Configuration
# ======================

INDEX_DIR = "faiss_indices"
INDEX_NAME = "paragraph_chunks"
INDEX_PATH = f"{INDEX_DIR}/{INDEX_NAME}_l2.index"

# Try to use enriched metadata first, fallback to basic metadata
METADATA_PATH_ENRICHED = f"{INDEX_DIR}/{INDEX_NAME}_metadata_enriched.json"
METADATA_PATH_BASIC = f"{INDEX_DIR}/{INDEX_NAME}_metadata.json"

if Path(METADATA_PATH_ENRICHED).exists():
    METADATA_PATH = METADATA_PATH_ENRICHED
    print("✅ Using ENRICHED metadata (includes age, sentiment, gender)")
elif Path(METADATA_PATH_BASIC).exists():
    METADATA_PATH = METADATA_PATH_BASIC
    print("⚠️  Using BASIC metadata (age/sentiment/gender NOT available)")
    print("   Run: python scripts/dataset/mergers/merge_metadata_to_chunks.py to enrich")
else:
    print(f"❌ Error: No metadata found!")
    print(f"   Tried: {METADATA_PATH_ENRICHED}")
    print(f"   Tried: {METADATA_PATH_BASIC}")
    print("   Run the chunking pipeline first:")
    print("   $ python scripts/test/RAG/run_chunking_pipeline.py")
    sys.exit(1)

# Check if index exists
if not Path(INDEX_PATH).exists():
    print(f"❌ Error: Index not found at {INDEX_PATH}")
    print("   Run the chunking pipeline first:")
    print("   $ python scripts/test/RAG/run_chunking_pipeline.py")
    sys.exit(1)

# ======================
# Initialize Retriever
# ======================

print("=" * 70)
print(" " * 20 + "INTERACTIVE RETRIEVAL TESTER")
print("=" * 70)

print("\n🔄 Loading retrieval system...")
retriever = ChunkRetriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

# Global state for interactive use
_last_query = None
_last_chunks = None
_last_cases = None


# ======================
# Core Functions
# ======================

def query(
    q: str,
    top_k: int = 25,
    top_cases: int = 5,
    chunks_per_case: int = 3,
    show_preview: bool = True
) -> List[dict]:
    global _last_query, _last_chunks, _last_cases

    print(f"\n{'=' * 70}")
    print(f"🔍 QUERY: {q}")
    print(f"{'=' * 70}")

    # Retrieve chunks
    chunks = retriever.retrieve_chunks(q, top_k=top_k)

    # Aggregate to cases
    cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=top_cases)

    # Store for later use
    _last_query = q
    _last_chunks = chunks
    _last_cases = cases

    # Display results
    if show_preview:
        print(f"\n✅ Retrieved {len(chunks)} chunks from {len(cases)} cases\n")

        for i, case in enumerate(cases):
            print(f"\n{'─' * 70}")
            print(f"[{i}] {case['case_id']} | {case['title'][:50]}...")
            print(f"{'─' * 70}")
            print(f"  📅 Date: {case['judgment_date']}")
            print(f"  📜 Articles: {', '.join(case['violated_articles'])}")
            print(f"  📊 Similarity: {case['avg_similarity']:.4f} (avg) | {case['max_similarity']:.4f} (max)")
            print(f"  📦 Chunks: {case['num_chunks']} retrieved")

            # Show enriched metadata from first chunk (case-level data)
            first_chunk = case['chunks'][0] if case['chunks'] else {}

            # Gender classification
            if 'classification' in first_chunk:
                cls = first_chunk['classification']
                print(f"  👤 Gender: {cls.get('gender', 'N/A')} ({cls.get('confidence', 'N/A')})")

            # Age info
            if 'age_info' in first_chunk:
                age_info = first_chunk['age_info']
                if age_info.get('has_age_info') or age_info.get('age_at_judgment'):
                    age_str = str(age_info.get('age_at_judgment', 'N/A'))
                    print(f"  🎂 Age at judgment: {age_str}")

            # Sentiment info (case-level average)
            if 'sentiment_info' in first_chunk:
                sent = first_chunk['sentiment_info']
                print(f"  😰 Emotional intensity: {sent.get('nrc_emotional_intensity', 0):.2%} | "
                      f"Fear: {sent.get('nrc_fear', 0):.2%} | "
                      f"Anger: {sent.get('nrc_anger', 0):.2%}")

            # Show top N chunks
            print(f"\n  Top {chunks_per_case} chunks:")
            for j, chunk in enumerate(case['chunks'][:chunks_per_case]):
                print(f"\n    [{j}] {chunk['chunk_id']} (score: {chunk['similarity_score']:.4f})")
                print(f"        Words: {chunk['word_count']} | Para: {chunk.get('paragraph_number', 'N/A')}")
                # Show first 100 chars of text
                preview = chunk.get('chunk_text_preview', chunk.get('chunk_text', ''))[:100]
                print(f"        Preview: {preview}...")

    print(f"\n{'=' * 70}")
    print(f"💡 Use case_detail(i) to see full case info")
    print(f"💡 Use show_chunk(case_idx, chunk_idx) to see full chunk text")
    print(f"{'=' * 70}\n")

    return cases


def case_detail(case_idx: int):
    if _last_cases is None:
        print("❌ No query results. Run query() first.")
        return

    if case_idx >= len(_last_cases):
        print(f"❌ Case index {case_idx} out of range (0-{len(_last_cases)-1})")
        return

    case = _last_cases[case_idx]

    print(f"\n{'=' * 70}")
    print(f"CASE DETAILS: [{case_idx}]")
    print(f"{'=' * 70}")
    print(f"Case ID:          {case['case_id']}")
    print(f"Case Number:      {case['case_no']}")
    print(f"Title:            {case['title']}")
    print(f"Judgment Date:    {case['judgment_date']}")
    print(f"Violated Articles: {', '.join(case['violated_articles'])}")
    print(f"\nRelevance Scores:")
    print(f"  Average:        {case['avg_similarity']:.4f}")
    print(f"  Maximum:        {case['max_similarity']:.4f}")
    print(f"  Chunks found:   {case['num_chunks']}")

    print(f"\nAll Retrieved Chunks:")
    print(f"{'─' * 70}")
    for i, chunk in enumerate(case['chunks']):
        print(f"[{i}] {chunk['chunk_id']}")
        print(f"    Score: {chunk['similarity_score']:.4f} | Words: {chunk['word_count']}")
        preview = chunk.get('chunk_text_preview', chunk.get('chunk_text', ''))[:80]
        print(f"    {preview}...\n")

    print(f"{'=' * 70}\n")


def show_chunk(case_idx: int, chunk_idx: int):
    if _last_cases is None:
        print("❌ No query results. Run query() first.")
        return

    if case_idx >= len(_last_cases):
        print(f"❌ Case index {case_idx} out of range (0-{len(_last_cases)-1})")
        return

    case = _last_cases[case_idx]

    if chunk_idx >= len(case['chunks']):
        print(f"❌ Chunk index {chunk_idx} out of range (0-{len(case['chunks'])-1})")
        return

    chunk = case['chunks'][chunk_idx]

    print(f"\n{'=' * 70}")
    print(f"CHUNK: [{case_idx}][{chunk_idx}]")
    print(f"{'=' * 70}")
    print(f"Chunk ID:        {chunk['chunk_id']}")
    print(f"Case ID:         {chunk['case_id']}")
    print(f"Case Title:      {case['title']}")
    print(f"Similarity:      {chunk['similarity_score']:.4f}")
    print(f"L2 Distance:     {chunk.get('l2_distance', 'N/A')}")
    print(f"Word Count:      {chunk['word_count']}")
    print(f"Paragraph:       {chunk.get('paragraph_number', 'N/A')}")
    print(f"Chunk Type:      {chunk.get('chunk_type', 'N/A')}")

    # Show enriched metadata if available
    if 'sentiment_info' in chunk:
        sent = chunk['sentiment_info']
        print(f"\n{'─' * 70}")
        print("SENTIMENT METADATA:")
        print(f"{'─' * 70}")
        print(f"Emotional Intensity: {sent.get('nrc_emotional_intensity', 0):.2%}")
        print(f"Fear:                {sent.get('nrc_fear', 0):.2%}")
        print(f"Anger:               {sent.get('nrc_anger', 0):.2%}")
        print(f"Sadness:             {sent.get('nrc_sadness', 0):.2%}")
        print(f"Passive Voice:       {sent.get('passive_voice_ratio', 0):.2%}")
        print(f"Victim Language:     {sent.get('victim_language_count', 0)}")
        print(f"Perpetrator Refs:    {sent.get('perpetrator_mentions', 0)}")

    if 'age_info' in chunk:
        age = chunk['age_info']
        if age.get('has_age_info'):
            print(f"\n{'─' * 70}")
            print("AGE METADATA:")
            print(f"{'─' * 70}")
            print(f"Age at judgment:     {age.get('age_at_judgment', 'N/A')}")
            print(f"Birth year:          {age.get('birth_year', 'N/A')}")

    if 'classification' in chunk:
        cls = chunk['classification']
        print(f"\n{'─' * 70}")
        print("GENDER CLASSIFICATION:")
        print(f"{'─' * 70}")
        print(f"Gender:              {cls.get('gender', 'N/A')}")
        print(f"Confidence:          {cls.get('confidence', 'N/A')}")

    print(f"\n{'─' * 70}")
    print("FULL TEXT:")
    print(f"{'─' * 70}")

    # Try to get full text from metadata
    # Note: chunk_text might be in metadata or need to be loaded separately
    text = chunk.get('chunk_text', chunk.get('chunk_text_preview', '[Full text not available]'))
    print(text)

    print(f"{'=' * 70}\n")


def stats():
    print(f"\n{'=' * 70}")
    print("RETRIEVAL SYSTEM STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total chunks indexed:  {retriever.index.ntotal:,}")
    print(f"Metadata entries:      {len(retriever.metadata):,}")
    print(f"Embedding dimension:   {retriever.index.d}")

    # Calculate some stats from metadata
    unique_cases = set(m['case_id'] for m in retriever.metadata)
    word_counts = [m['word_count'] for m in retriever.metadata]

    import numpy as np
    print(f"\nCase Statistics:")
    print(f"  Unique cases:        {len(unique_cases)}")
    print(f"  Avg chunks per case: {len(retriever.metadata) / len(unique_cases):.1f}")

    print(f"\nChunk Statistics:")
    print(f"  Avg word count:      {np.mean(word_counts):.1f}")
    print(f"  Median word count:   {np.median(word_counts):.0f}")
    print(f"  Min word count:      {min(word_counts)}")
    print(f"  Max word count:      {max(word_counts)}")

    print(f"{'=' * 70}\n")


def compare_queries(queries: List[str], top_k: int = 25, top_cases: int = 3):
    print(f"\n{'=' * 70}")
    print(f"COMPARING {len(queries)} QUERIES")
    print(f"{'=' * 70}\n")

    all_results = []

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}] Query: \"{q}\"")
        print(f"{'─' * 70}")

        chunks = retriever.retrieve_chunks(q, top_k=top_k)
        cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=top_cases)

        all_results.append({
            'query': q,
            'chunks': chunks,
            'cases': cases
        })

        print(f"Top {len(cases)} cases:")
        for j, case in enumerate(cases, 1):
            print(f"  {j}. {case['case_id'][:12]}... | "
                  f"Sim: {case['avg_similarity']:.3f} | "
                  f"Arts: {', '.join(case['violated_articles'][:3])}")

    print(f"\n{'=' * 70}\n")
    return all_results


def test_context_window(q: str, max_cases: int = 5, chunks_per_case: int = 3):
    print(f"\n{'=' * 70}")
    print("MISTRAL 7B CONTEXT WINDOW TEST")
    print(f"{'=' * 70}")
    print(f"Query: {q}\n")

    # Retrieve
    chunks = retriever.retrieve_chunks(q, top_k=25)
    cases = retriever.aggregate_chunks_to_cases(chunks, top_k_cases=max_cases)

    # Calculate context
    total_words = 0
    total_chunks = 0

    print(f"Selected {len(cases)} cases:\n")

    for i, case in enumerate(cases, 1):
        case_words = sum(c['word_count'] for c in case['chunks'][:chunks_per_case])
        total_words += case_words
        total_chunks += min(chunks_per_case, len(case['chunks']))

        print(f"  [{i}] {case['case_id'][:15]}... | "
              f"{min(chunks_per_case, len(case['chunks']))} chunks | "
              f"{case_words} words")

    # Token estimate (1 word ≈ 1.3 tokens)
    total_tokens = int(total_words * 1.3)

    print(f"\n{'─' * 70}")
    print("PROMPT BREAKDOWN:")
    print(f"{'─' * 70}")
    print(f"  System prompt:       ~200 tokens")
    print(f"  User query:          ~100 tokens")
    print(f"  Retrieved context:   ~{total_tokens} tokens ({total_chunks} chunks, {total_words} words)")
    print(f"  Prompt template:     ~300 tokens")
    print(f"  Generation buffer:   ~1,500 tokens")
    print(f"  {'─' * 60}")
    print(f"  TOTAL:              ~{200 + 100 + total_tokens + 300 + 1500} tokens")

    if (200 + 100 + total_tokens + 300 + 1500) < 8000:
        print(f"\n  ✅ Fits in Mistral 7B optimal window (8k tokens)")
    else:
        print(f"\n  ⚠️  Exceeds optimal window! Consider reducing chunks.")

    print(f"{'=' * 70}\n")


def example_queries():
    examples = [
        # Detention/Article 5 cases
        "On 2 March 1994 Mr Dicle and Mr Doğan were taken into police custody on the orders of the public prosecutor at the Ankara National Security Court. On 4 March 1994 Mrs Zana suffered the same fate. A few days later the public prosecutor at the Ankara National Security Court ordered the detention of those three applicants in police custody to be extended until 16 March 1994. While in custody, the applicants made no statements to the police. On 16 March 1994 they were brought before a judge of the Ankara National Security Court and placed in detention pending trial.",

        # Custody/Family life Article 8 cases
        "At the beginning of the events relevant to the application, K. had a daughter, P., and a son, M., born in 1986 and 1988 respectively. From March to May 1989 K. was voluntarily hospitalised for about three months, having been diagnosed as suffering from schizophrenia. The applicants initially cohabited from the summer of 1991 to July 1993. In 1991 both P. and M. were living with them. From 1991 to 1993 K. and X were involved in a custody and access dispute concerning P. In May 1992 a residence order was made transferring custody of P. to X.",

        # Property rights/Article 1 Protocol 1 cases
        "In June 1949 plots of agricultural land owned by the applicant's father were expropriated by the former Doksy District National Council under the Czechoslovak New Land Reform Act No. 46/1948. The applicant's father had never obtained any compensation. In 1957 some of these plots were transferred to the ownership of natural persons in an assignment procedure under the 1948 Act. After the fall of the communist regime in Czechoslovakia, the Act No. 229/1991 on Adjustment of Ownership Rights in respect of Land and Other Agricultural Property entered into force on 24 June 1991.",

        # Freedom of expression/Article 10 cases
        "On 20 July 1992 the newspaper Telegraf published a poem by the applicant. It was dated 17 July 1992 and entitled 'Good night, my beloved'. One of its verses read as follows: 'In Prague prisoner Havel is giving up his presidential office. In Bratislava the prosecutor rules again. And rule by one party is above the law.' The poem was later published in another newspaper. On 30 July 1992 several newspapers published a statement which the applicant had distributed to the Public Information Service the day before.",

        # Torture/Article 3 cases
        "At 10 a.m. on 12 September 1994 the applicant's brother, Kenan Bilgin, was arrested at a taxi rank in Dikmen (Ankara) by plainclothes police officers. His family was not informed. The applicant received three anonymous telephone calls from someone who confirmed that his brother was being held at Gölbaşı (Ankara) with three other prisoners. He was told that his brother's condition was serious and that he was being administered serum. During the last conversation, which took place on 15 November 1994, the caller said that the applicant's brother had been moved elsewhere.",

        # Prison conditions/Article 3 cases
        "From 5 October 1993 the applicant served a sentence of nine years' imprisonment for the theft, possession and sale of firearms. On an unspecified date in early April 1998 he was transferred from Lukiškės Prison to Pravieniškės Prison. From the moment when the applicant arrived in Pravieniškės Prison he was placed in the separate segregation unit of the prison, located in Wing 5 of the prison. From 5 to 20 January 1999 the applicant was detained in solitary confinement. He was again placed in the segregation unit on 20 January 1999.",

        # Confiscation/Property rights cases
        "In 1946 the former Czechoslovakia confiscated the property of the applicant's father which was situated in its territory, including the painting in question, under Decree no. 12 on the 'confiscation and accelerated allocation of agricultural property of German and Hungarian persons and of those having committed treason and acted as enemies of the Czech and Slovak people', issued by the President of the former Czechoslovakia on 21 June 1945 (the Beneš Decrees).",

        # Judicial proceedings/Fair trial cases
        "On 5 August 1992 Mr Slobodník publicly declared that he would sue the applicant for the above statement. The applicant association published a book entitled Euskadi at war in 1987. On 29 April 1988 a ministerial order was issued by the French Ministry of the Interior banning the circulation, distribution and sale of the book in France in any of its four versions on the ground that 'the circulation in France of this book, which promotes separatism and vindicates recourse to violence, is likely to constitute a threat to public order'.",

        # Tax/Property cases
        "The applicant and another person transferred land, property and a sum of money to a limited liability company which the applicant had just formed and of which he owned almost the entire share capital. The company applied to the tax authorities for a reduction in the applicable rate of certain taxes payable on the transfer of property. In the first set of proceedings, the tax authorities served a supplementary tax assessment on the applicant on 31 August 1987 on the ground that the property transferred to the company had been incorrectly valued.",

        # Marriage/Family life cases
        "On 29 April 1962 the applicant married Mr A. Gigliozzi in a religious ceremony which was also valid in the eyes of the law. On 23 February 1987 the applicant petitioned the Rome District Court for judicial separation. In a judgment dated 2 October 1990 the District Court granted her petition and also ordered Mr Gigliozzi to pay the applicant maintenance of 300,000 Italian lira per month. On 20 November 1987, the applicant was summoned to appear before the Lazio Regional Ecclesiastical Court on 1 December 1987.",
    ]

    example_labels = [
        "Detention/Article 5 (police custody)",
        "Custody/Article 8 (family separation)",
        "Property/Protocol 1 (land expropriation)",
        "Expression/Article 10 (political speech)",
        "Torture/Article 3 (enforced disappearance)",
        "Prison/Article 3 (conditions of detention)",
        "Property/Protocol 1 (Beneš Decrees)",
        "Fair trial/Article 6 (publication ban)",
        "Tax/Property (tax assessment dispute)",
        "Marriage/Article 8 (ecclesiastical court)",
    ]

    print(f"\n{'=' * 70}")
    print("EXAMPLE QUERIES - Real Case Facts from Dataset")
    print(f"{'=' * 70}\n")
    print("These are actual case paragraphs. Use them to test retrieval of")
    print("similar cases based on factual patterns.\n")

    for i, (label, example) in enumerate(zip(example_labels, examples), 1):
        print(f"{i:2}. [{label}]")
        preview = example[:120] + "..." if len(example) > 120 else example
        print(f"   {preview}\n")

    print(f"{'─' * 70}")
    print("Try them with:")
    print("  >>> query(example_queries.examples[0])  # First example")
    print("  >>> query(example_queries.examples[4])  # Torture case")
    print("  >>> compare_queries([examples[0], examples[4]])")
    print(f"{'=' * 70}\n")

    # Store examples for easy access
    example_queries.examples = examples
    example_queries.labels = example_labels


def help():
    print(f"\n{'=' * 70}")
    print("AVAILABLE FUNCTIONS")
    print(f"{'=' * 70}\n")

    functions = [
        ("query(q)", "Search for relevant chunks and cases"),
        ("case_detail(i)", "Show detailed info about case i"),
        ("show_chunk(i, j)", "Show full text of chunk j in case i"),
        ("stats()", "Show system statistics"),
        ("compare_queries([...])", "Compare multiple queries"),
        ("test_context_window(q)", "Test Mistral 7B context usage"),
        ("example_queries()", "Show example queries to try"),
        ("help()", "Show this help message"),
    ]

    for func, desc in functions:
        print(f"  {func:30} - {desc}")

    print(f"\n{'─' * 70}")
    print("QUICK START:")
    print(f"{'─' * 70}")
    print('  >>> query("detention without trial")')
    print("  >>> case_detail(0)")
    print("  >>> show_chunk(0, 0)")
    print("  >>> example_queries()")
    print(f"{'=' * 70}\n")


# ======================
# Startup Message
# ======================

print("\n✅ Retrieval system loaded successfully!\n")
print(f"📊 Index: {retriever.index.ntotal:,} chunks")
print(f"📁 Metadata: {len(retriever.metadata):,} entries\n")

help()

print("💡 TIP: Start with example_queries() to see what you can search for!\n")
