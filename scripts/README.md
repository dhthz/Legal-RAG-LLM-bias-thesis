# scripts/

Runnable steps of the thesis. Each runs from the repository root as a module, e.g. `python -m scripts.eval.run_bias_audit`, and reads and writes paths relative to the root (`dataset/`, `faiss_indices/`, `logs/`, `docs/`). The library code they use is in [`src/`](../src/).

## chunking_and_embeddings/: the index, and the fixes that need their own index or model

| File | What it does |
|---|---|
| `run_chunking_pipeline.py` | Chunks the 9,000 training judgments into paragraphs, embeds them with nomic-embed-text-v1 and builds the FAISS index; checks the index SHA256 against the manifest |
| `verify_chunk_coverage.py` | Counts how many source paragraphs ended up in some chunk (the chunker drops very short ones) |
| `build_blind_index.py` | Blind index fix: rewrites every chunk gender-neutrally and re-embeds the corpus |
| `build_leace_index.py` | LEACE fixes: fits the eraser on case gender (first build, new index) or on the query pronoun (`--concept query_pronoun`, `leace_query`) |
| `finetune_cda.py` | Embedder fine-tune with counterfactual twins (Kim et al. 2025 recipe): sweep, selection on dev, re-embedding |
| `finetune_ccd.py` | Embedder fine-tune with content conditional debiasing (Deng et al. 2024), then re-embedding |
| `index_variants.py` | Records every index or model built for a fix, with its SHA256, under `index_variants` in the manifest |

## dataset/: metadata, question sets and dataset charts

| File | What it does |
|---|---|
| `case_metadata.py` | `export` writes the per-case gender and NRC metadata to `docs/data/case_metadata.csv`; `merge` adds it back to a downloaded `train.jsonl` |
| `generate_eval_queries.py` | Builds the 100 retrieval-harness questions (case facts with names removed, leak-checked) |
| `metadata_extractors/sentiment_extractor.py` | NRC emotion scores (fear, anger, sadness, disgust, intensity) of a case or question |
| `metadata_extractors/kimi_metadata_extractor.py` | Applicant gender labels for the held-out test split through an LLM API (used to sample the audit questions and the 300 automatic he/she pairs) |
| `mergers/merge_metadata_to_chunks.py` | Copies each case's gender and NRC scores onto its chunks (`paragraph_chunks_metadata_enriched.json`) |
| `gender_label_validation/export_gender_validation_sheet.py` | Draws the 300 cases for the hand check of the gender labels and writes the blind sheet |
| `gender_label_validation/analyze_gender_validation.py` | Scores the labels against the hand check (accuracy per label, Wilson intervals, reweighted to the corpus) |
| `splitters/split_dataset.py` | Splits the training set into 10 parts (the harness questions are drawn from the last, `train_split_10`) |
| `variant_set_construction/select_variant_bases.py` | Draws the base cases for the he/she/neutral/emotional question variants, with quotas and exclusion gates |
| `variant_set_construction/make_variant_templates.py` | Writes the template rows that the variants were written into |
| `variant_set_construction/check_variant_gate.py` | Checks each variant set: he and she identical apart from gender words, emotional lift and length within limits |
| `metadata_visualizers/visualize_dataset_overview.py` | Dataset charts: articles, outcomes, countries, years |
| `metadata_visualizers/visualize_chunks.py` | Chunk charts: length, chunks per case, chunk types |
| `metadata_visualizers/visualize_gender_metadata.py` | Gender label counts and charts, including the hand-check accuracy |
| `metadata_visualizers/visualize_nrc_metadata.py` | NRC emotion summary and chart, for the corpus and by applicant gender |
| `metadata_visualizers/visualize_llm_queries.py` | Charts of the audit question set: gender by article, quotas, countries, length, variant check |
| `metadata_visualizers/visualize_bias_audit.py` | The audit charts 01–06 |
| `metadata_visualizers/visualize_mitigation.py` | The fix charts 07–08, from `mitigation_results.json` |

The variant scripts document how the frozen question files in `docs/llm_queries/` were built; their intermediate drafts are not kept.

## eval/: the audit, the fixes and the robustness checks

| File | What it does |
|---|---|
| `retrieval_eval.py` | Retrieval harness: recall@k and MRR on the 100 harness questions, with confidence intervals |
| `run_bias_audit.py` | Runs the audit questions through the pipeline and logs every interaction; `--arms` applies fixes, `--whole-system` makes the 4 predict runs |
| `bias_statistics.py` | Statistics of one audit run: he/she flip test, retrieved gender vs corpus and by question gender, cited vs not cited, with Holm correction |
| `multi_run_stability.py` | Across the 4 runs: repeat-run noise and the LLM test split by what the LLM was shown (identical input, different order) |
| `llm_output_bias.py` | Predicted violated articles: accuracy and change between wordings, against noise |
| `export_audit_results.py` | Phase E results into `docs/bias_audit/results/` (`final_results.json`, tables 01–10) |
| `mitigation_eval.py` | Search-side evaluation of a fix or combination with the audit's instruments and the harness |
| `rewrite_fidelity.py` | How faithful the automatic gender swap and neutral rewrite are, against the hand-written variants |
| `run_fixed_case_test.py` | Runs the he/she questions with identical cases, so only the pronoun differs (robustness check R3b) |
| `run_swap_set_test.py` | Builds and runs the 300 automatic he/she pairs (robustness check R4) |
| `robustness_checks.py` | Robustness checks R1–R5: topic control, gender-relevant precedents, detectable effects, LLM with identical cases, large pair set, retrieval cost |
| `export_mitigation_results.py` | Phase F results into `docs/bias_audit/results/` (`mitigation_results.json`, tables 11–15, robustness summaries) |
