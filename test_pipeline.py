"""
Test Script for RAGPipeline

This tests the complete pipeline: retrieval + LLM generation + logging
Make sure Ollama is running before executing this script.
"""

import json
from pathlib import Path
from typing import Optional
from src.rag.pipeline import RAGPipeline

def read_last_log_entry(log_path: str) -> Optional[dict]:
    """Read the last JSONL entry from the log file."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
    except FileNotFoundError:
        return None
    return None

def main():
    print("=" * 70)
    print(" " * 20 + "RAG PIPELINE TEST")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    print("📦 Initializing RAG Pipeline...")
    print("   - Loading FAISS index...")
    print("   - Loading metadata...")
    print("   - Connecting to LLM...")
    
    try:
        pipeline = RAGPipeline()
        print("✓ Pipeline initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    # Test query (same as your test.py)
    query = "The applicant's brother was arrested by plainclothes police and disappeared."
    
    print("=" * 70)
    print("📝 USER QUERY:")
    print("=" * 70)
    print(f"{query}\n")
    
    # Execute query
    print("🔍 Processing query...")
    print("   - Retrieving relevant chunks...")
    print("   - Aggregating to cases...")
    print("   - Building prompt...")
    print("   - Generating LLM response...")
    print("   - Logging interaction...\n")
    
    try:
        result = pipeline.query(query)
        print("✓ Query completed successfully!\n")
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display results
    print("=" * 70)
    print("📚 RETRIEVED CASES:")
    print("=" * 70)
    for i, case in enumerate(result['cases'], 1):
        # Extract metadata from first chunk
        first_chunk = case['chunks'][0]
        country = first_chunk.get('defendants', [None])[0]
        gender = first_chunk.get('classification', {}).get('gender', 'Unknown')
        age = first_chunk.get('age_info', {}).get('age_at_judgment')
        
        print(f"\n{i}. {case['title']}")
        print(f"   Case ID: {case['case_id']}")
        print(f"   Date: {case['judgment_date']}")
        print(f"   Country: {country}")
        print(f"   Gender: {gender}")
        print(f"   Age: {age if age else 'N/A'}")
        print(f"   Violated Articles: {', '.join(case['violated_articles'])}")
        print(f"   Similarity Score: {case['avg_similarity']:.4f}")
        print(f"   Chunks Retrieved: {case['num_chunks']}")
    
    print("\n" + "=" * 70)
    print("🤖 LLM RESPONSE:")
    print("=" * 70)
    print()
    print(result['response'])
    print()
    print("=" * 70)
    
    # Read token usage from log file
    log_entry = read_last_log_entry(pipeline.config.log_path)
    
    if log_entry and 'token_usage' in log_entry:
        token_usage = log_entry['token_usage']
        print("\n" + "=" * 70)
        print("📊 TOKEN USAGE:")
        print("=" * 70)
        print(f"   Prompt tokens:     {token_usage['prompt_tokens']:,}")
        print(f"   Completion tokens: {token_usage['completion_tokens']:,}")
        print(f"   Total tokens:      {token_usage['total_tokens']:,}")
        print("=" * 70)
    
    # Summary
    print("\n✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"   - Query: {len(result['query'])} characters")
    print(f"   - Cases retrieved: {len(result['cases'])}")
    print(f"   - Response length: {len(result['response'])} characters")
    if log_entry:
        print(f"   - Log file: {pipeline.config.log_path}")
        print(f"   - Interaction logged: ✓")
    print()
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
