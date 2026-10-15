# RAG Bias Audit

A Claude Code plugin that audits a retrieval-augmented generation (RAG) system for gender bias, in the user's own codebase, and applies and compares fixes.

Claude does the codebase work: it finds the retriever and the LLM call, adds logging, builds he/she twin questions with you, and applies the fixes. Fixed scripts do all the statistics, so every audit measures the same thing. The measurement reproduces the audit of the MSc thesis it comes from exactly (166/166 values), on a legal RAG system over European Court of Human Rights case law.

## Install

In Claude Code:

```
/plugin marketplace add dhthz/Legal-RAG-LLM-bias-thesis
/plugin install rag-bias-audit@rag-bias-audit
```

Then, in your RAG project, ask Claude to audit it for gender bias, or run `/rag-bias-audit:bias-audit`.

Python packages for the environment your system runs in: `requirements.txt` (minimum versions, so nothing you have is downgraded; the audit scripts and charts need only numpy, scipy, jsonschema and matplotlib, the twin questions and the fixes need the rest).

## What it does

1. Finds where your system retrieves documents and calls the LLM.
2. Labels the documents' gender (yours, or a labelling step you validate on a sample).
3. Builds he/she twins of your questions (only the pronouns and gendered words change) for you to review.
4. Logs 4 repeat runs of every question (the repeats measure chance variation), after telling you how long it will take and, for a paid API, what it will cost.
5. Reports where the bias enters: the search (does the top document change between twins?) or the LLM (do answers change when the LLM gets identical input?), with charts, and says how large an effect each test could have missed.
6. Applies fixes in your code: blind query (rewrite the question with singular they), PCF (Zhou et al., NeurIPS 2024) or LEACE on the query vector (Belrose et al., NeurIPS 2023), each checked by a tripwire that the baseline is unchanged.
7. Re-measures and compares each fix with the baseline; optionally checks that the search stays on topic (when your documents have a topic or category), and exports the full report with its charts as one PDF.

## The three audit files

Everything the statistics read: a run log, document labels and question pairs, each with a JSON Schema in `skills/bias-audit/scripts/schemas/`. `validate.py` checks them before any statistic runs.

## Scope

Gender (he/she) is the validated attribute: the twin rewrite matches hand-written twins word for word (581/581 words). Other attributes work with your own twin questions and labels but are not validated. Relevance is checked only when your documents have a topic or category, and a topic match is a rough stand-in: check your search with your own test (e.g. recall@k) before keeping a fix.

Only fixes that act on the question are applied. Fixes that rebuild the index (blind index) or retrain the embedder (CCD, Deng et al. 2024) worked in the thesis too, but change your index or model; Claude describes them when relevant and does not apply them.

## Credits

The fixes are implemented from their papers; the statistics are standard tests (exact McNemar, chi-square with Cramér's V, Wilcoxon signed-rank, Wilson intervals, Holm-Bonferroni).

- Blind query: the gender-neutral rewrite of Sun et al. (2021), [arXiv:2102.06788](https://arxiv.org/abs/2102.06788); query-side blindness as in Garg et al., AIES 2019.
- PCF: Zhou et al., *Counterfactual Fairness by Combining Factual and Counterfactual Predictions*, NeurIPS 2024, [arXiv:2409.01977](https://arxiv.org/abs/2409.01977).
- LEACE: Belrose et al., *LEACE: Perfect Linear Concept Erasure in Closed Form*, NeurIPS 2023, [arXiv:2306.03819](https://arxiv.org/abs/2306.03819), with the authors' [concept-erasure](https://github.com/EleutherAI/concept-erasure) library. Here it erases the he/she direction of the query, a concept defined by twin pairs as in Bolukbasi et al., NeurIPS 2016.
- Twin questions: counterfactual data augmentation (Zhao et al., NAACL 2018; Garg et al., AIES 2019).
