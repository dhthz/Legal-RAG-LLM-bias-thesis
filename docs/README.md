# docs/

Everything a reader needs from the thesis: results, charts, the question sets and the released metadata. Every file is made by a script in [`scripts/`](../scripts/) from the logs (which are not published) or from the files here; results files list the SHA256 of each log they were computed from.

## bias_audit/: the audit (Phase E) and the fixes (Phase F)

| File | What it shows | Made by |
|---|---|---|
| `results/final_results.json` | Phase E headline results, with the index, log and question checksums | `eval/export_audit_results.py` |
| `results/tables/01_master_holm.csv` | All audit tests with Holm-corrected p-values | same |
| `results/tables/02_paired_variants_top1.csv` | He/she and neutral/emotional swaps: how often the top case changes | same |
| `results/tables/03_retrieved_gender_by_query_gender.csv` | Gender of the retrieved cases by the question's applicant gender | same |
| `results/tables/04_llm_citation_by_input_similarity.csv` | Does the cited case change with the wording, split by what the LLM was shown | same |
| `results/tables/05`–`07_predicted_articles_*.csv` | Predicted violated articles: accuracy, tests and shares by wording | same |
| `results/tables/08_noise_floors.csv` | Repeat-run noise of the answers | same |
| `results/tables/09_per_run_summary.csv` | Each of the 4 audit runs | same |
| `results/tables/10_gender_label_validation.csv` | Accuracy of the gender labels against the 300-case hand check | same |
| `results/mitigation_results.json` | Phase F results per fix, with the SHA256 of all 71 logs read | `eval/export_mitigation_results.py` |
| `results/tables/11_mitigation_tradeoff.csv` | The trade-off table: bias removed against retrieval cost, every fix | same |
| `results/tables/12_mitigation_retrieval.csv` | Each fix on the search: flips (40 and 300 pairs), recall, MRR, topical precision | same |
| `results/tables/13_mitigation_whole_system.csv` | Each fix on the answers: he/she and tone citation changes against repeat noise and the baseline | same |
| `results/tables/14_mitigation_changed_cases.csv` | The cases each fix dropped and added, and gender-relevant precedents | same |
| `results/tables/15_robustness_checks.csv` | Robustness checks R1–R5, one line each | same |
| `results/robustness/*.json` | The full R1–R5 summaries | `eval/robustness_checks.py` (copied by the export) |
| `visualizations/01`–`06_*.png` | Audit charts: retrieved gender, he/she grid, flips, LLM input split, findings summary, predicted articles | `dataset/metadata_visualizers/visualize_bias_audit.py` |
| `visualizations/07_fixes_tradeoff.png` | Bias removed against relevance kept, per fix | `dataset/metadata_visualizers/visualize_mitigation.py` |
| `visualizations/08_fixes_whole_system.png` | He/she answers citing different cases, per fix, against repeat noise | same |

## The toolkit overview

| File | What it shows | Made by |
|---|---|---|
| `bias_audit/toolkit plugin review.pdf` | The toolkit in six pages: how it works, what is in the plugin, one session stage by stage, what the scripts measure and how they were validated, why an agent-guided plugin, the test and the limitations | written by hand |

## llm_queries/: the question sets

| File | What it is | Made by |
|---|---|---|
| `audit_queries_main.jsonl` | The 175 audit questions: the facts of held-out test cases as they are (names and pronouns kept), sampled by applicant gender, article and country | drawn from `test.jsonl`, gender labels by `metadata_extractors/kimi_metadata_extractor.py` |
| `audit_queries_variants.jsonl` | 40 bases × neutral, he, she and emotional versions (160 questions) | neutral, he and she written by hand; emotional by swapping in NRC emotion words; all checked with `variant_set_construction/check_variant_gate.py` |
| `variant_batch1_analogs.csv`, `variant_batch2_selection.csv` | How the 40 bases were chosen | `variant_set_construction/select_variant_bases.py` |
| `retrieval_harness_queries.jsonl` | The 100 retrieval-harness questions (published without the removed names) | `dataset/generate_eval_queries.py` |
| `visualizations/*.png` | Charts of the audit question set | `dataset/metadata_visualizers/visualize_llm_queries.py` |

## data/: metadata added to the dataset

| File | What it is | Made by |
|---|---|---|
| `case_metadata.csv` | Applicant gender and NRC emotion scores of the 9,000 indexed cases (`confidence` = `human` for the 209 hand-labelled) | `dataset/case_metadata.py export` |
| `gender_validation_key.csv`, `gender_validation_tagged.csv` | The 300-case hand check of the gender labels: the AI's label and the hand label | `gender_label_validation/export_gender_validation_sheet.py`, labelled by hand |

## dataset/ and metadata_reports/: the dataset described

| File | What it shows | Made by |
|---|---|---|
| `dataset/visualizations/case_analysis/*.png` | Articles, outcomes, countries, years | `dataset/metadata_visualizers/visualize_dataset_overview.py` |
| `dataset/visualizations/chunk_analysis/*.png` | Chunk length, chunks per case, chunk types | `dataset/metadata_visualizers/visualize_chunks.py` |
| `dataset/visualizations/gender_analysis/*.png` | Gender labels: distribution, by article, over time, hand-check accuracy | `dataset/metadata_visualizers/visualize_gender_metadata.py` |
| `dataset/visualizations/sentiment_analysis/emotion_composition.png` | NRC emotion scores, corpus and by applicant gender | `dataset/metadata_visualizers/visualize_nrc_metadata.py` |
| `metadata_reports/gender_labels.csv` | Gender label counts, before and after the hand labelling | `dataset/metadata_visualizers/visualize_gender_metadata.py` |
| `metadata_reports/nrc_emotion.csv` | NRC emotion scores: mean, SD and median, corpus and by applicant gender | `dataset/metadata_visualizers/visualize_nrc_metadata.py` |

## retrieval_eval/: retrieval design

| File | What it is | Made by |
|---|---|---|
| `eval_20260715_142527.json`, `eval_20260720_192521.json` | Harness runs from the retrieval design phase, with mean-similarity aggregation (recall@1 0.90 and 0.89; the frozen system uses max similarity, 0.95) | `eval/retrieval_eval.py` |
