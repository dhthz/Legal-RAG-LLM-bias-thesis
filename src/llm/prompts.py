
from typing import List, Dict

def build_legal_prompt(query: str, cases: List[Dict]) -> List[Dict]:
    system_prompt = """You are a legal research assistant for the European Court of Human Rights (ECtHR). 
A user has described a legal situation. Your job is to identify which of the retrieved cases below are most analogous to that situation, explain why they are relevant, and note what articles were violated.
The retrieved cases may not be an exact match — look for factual and legal similarities."""
    
    # Build context from cases
    context = ""
    for i, case in enumerate(cases, 1):
        context += f"--- Case {i}: {case['title']} ({case['case_id']}) ---\n"
        context += f"Date: {case['judgment_date']}\n"
        context += f"Violated Articles: {', '.join(case['violated_articles'])}\n"
        context += f"Relevance Score: {case['avg_similarity']:.3f}\n\n"

        for chunk in case['chunks'][:2]:
            text = chunk.get('chunk_text', chunk.get('chunk_text_preview', ''))
            context += f"{text}\n\n"

    # Build user prompt (without system instructions)
    user_prompt = f"""=== RETRIEVED CASES ===
{context}
=== END OF RETRIEVED CASES ===

=== USER'S LEGAL SITUATION ===
{query}
=== END OF USER'S LEGAL SITUATION ===

Based only on the retrieved cases above, identify the most relevant cases, explain the factual similarities, and state what articles were violated."""
    
    # Return messages list ready for API
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]