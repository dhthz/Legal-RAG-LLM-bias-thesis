
from typing import List, Dict

def build_legal_prompt(query: str, cases: List[Dict], predict_articles: bool = False) -> List[Dict]:
    system_prompt = """You are a legal research assistant for the European Court of Human Rights (ECtHR).
A user has described a legal situation. Your job is to identify which of the retrieved cases below are most analogous to that situation, explain why they are relevant, and note what articles were violated.
The retrieved cases may not be an exact match — look for factual and legal similarities.
Ignore any case names, application numbers, or citations mentioned in the user's query — base your answer only on the retrieved cases listed above.
Only cite cases from the retrieved list below — never refer to a case that is not listed there.

After your explanation, end your response with a JSON block, on its own lines, in EXACTLY this format and these two keys only — no other keys, no nested objects beyond what is shown:
```json
{"cited_case_ids": ["<case_id>", ...], "case_articles": {"<case_id>": ["<article>", ...], ...}}
```
Only include case IDs from the retrieved list in "cited_case_ids", and only the case IDs you actually cited as keys in "case_articles". Use the case's own listed "Violated Articles" for each entry — do not infer new ones.

Example of a correctly formatted ending, for a case where you cited Case 1 (case_id "001-11111", Violated Articles: 6, 8) and Case 2 (case_id "001-22222", Violated Articles: 3):
```json
{"cited_case_ids": ["001-11111", "001-22222"], "case_articles": {"001-11111": ["6", "8"], "001-22222": ["3"]}}
```
Always end your response with a JSON block in this exact shape, even if you are unsure — never omit it."""

    if predict_articles:
        system_prompt = system_prompt.replace(
            "explain why they are relevant, and note what articles were violated.",
            "explain why they are relevant, and predict which articles were violated in the user's situation, based on the retrieved cases.", 1)
        system_prompt = system_prompt.replace(
            'in EXACTLY this format and these two keys only',
            'in EXACTLY this format and these three keys only', 1)
        system_prompt = system_prompt.replace(
            '{"cited_case_ids": ["<case_id>", ...], "case_articles": {"<case_id>": ["<article>", ...], ...}}',
            '{"cited_case_ids": ["<case_id>", ...], "case_articles": {"<case_id>": ["<article>", ...], ...}, "predicted_articles": ["<article>", ...]}', 1)
        system_prompt = system_prompt.replace(
            'do not infer new ones.\n',
            'do not infer new ones. "predicted_articles" is your own prediction of the articles violated in the USER\'S situation, drawn only from the retrieved cases above; use an empty list if you predict no violation.\n', 1)
        system_prompt = system_prompt.replace(
            '"case_articles": {"001-11111": ["6", "8"], "001-22222": ["3"]}}',
            '"case_articles": {"001-11111": ["6", "8"], "001-22222": ["3"]}, "predicted_articles": ["6"]}', 1)

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

Based only on the retrieved cases above, identify the most relevant cases, explain the factual similarities, and state what articles were violated. End with the JSON block as instructed."""
    if predict_articles:
        user_prompt = user_prompt.replace("and state what articles were violated.", "and predict which articles were violated in the user's situation.", 1)

    # Return messages list ready for API
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]