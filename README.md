# Gender Bias in a Legal RAG System

**An audit of gender bias in a retrieval-augmented legal assistant, six published fixes compared, and a Claude Code plugin that runs the same audit on any RAG system.**

MSc thesis. The system under audit answers questions on European Court of Human Rights case law: 9,000 judgments split into 217,903 paragraph chunks, retrieved with FAISS and `nomic-embed-text-v1`, answered by Mistral 7B through Ollama. Every interaction is logged, and every claim is tested against repeat-run noise with exact tests, effect sizes and Holm-Bonferroni correction.

![Swapping "he" for "she" changes which case is found in 15 of 40 pairs](docs/bias_audit/visualizations/03_he_she_flip_summary.png)

## Findings

| Finding | Evidence |
|---|---|
| **Bias enters at retrieval** | Changing only "he" to "she" changes the top case in 15 of 40 hand-written pairs; all 11 changes of the case's gender go toward a female applicant (exact McNemar, Holm p = 0.0029). On 300 automatic pairs: 104 changes. |
| **It reaches the answer** | The he and she versions of a question cite different cases in 63.7% of 160 pairs, against 2.5% when the same question is asked twice. |
| **The LLM adds no bias of its own** | Given identical cases in the same order, its citations change no more than repeat noise. |
| **The LLM amplifies retrieval** | Given the same cases in a different order, it cites a different case 69% of the time (position bias). |
| **Emotional tone has no effect** | Neither on retrieval nor on the answer. |

Full results: [`docs/bias_audit/results/`](docs/bias_audit/results/) (tables and `final_results.json`), charts in [`docs/bias_audit/visualizations/`](docs/bias_audit/visualizations/).

## Fixes

Five mitigation methods from the literature and two classic baselines, covering every point of the pipeline. Each is implemented here from its paper, measured on the same tests, and checked for retrieval cost (recall@1, MRR and topical precision of the retrieved cases). Where this setting needed a change from the paper, the change is stated below the table and argued in the thesis.

| Fix | Where it acts | Top case changes (300 pairs) | Answers differ he/she | Retrieval cost |
|---|---|---:|---:|---|
| None (baseline) | | 104 | 63.7% | |
| Blind query, a gender-neutral rewrite [1, 2] | query text | 0 | 1.9% | none |
| PCF [3] | retrieval scores | 1 | 5.0% | none |
| Content conditional debiasing fine-tune [4] | embedder | 15 | 11.2% | none |
| LEACE on the query [5, 6] | query vector | 8 | 13.8% | none |
| Blind index, the same rewrite on every stored chunk | stored documents | 41 | 31.2% | none |
| Counterfactual fine-tune of the embedder [7, 8] | embedder | 27 | 41.2% | lower precision |
| Permutation self-consistency [9] | generation | 104 | 53.1% | none |

Changes from the papers: LEACE erases the he/she direction of the query rather than each document's gender (the paper's setup failed here, 114 changes); the content conditional fine-tune uses a stronger anchor to the original model (the paper's setting found a shortcut) [10]; the counterfactual fine-tune uses label-preserving twins rather than the paper's group-level objective.

The fixes that act on the search reduce the effect, the best ones to near zero, without losing relevance. A fix inside the LLM cannot: the bias is already in the cases it is given. Trade-off table: [`11_mitigation_tradeoff.csv`](docs/bias_audit/results/tables/11_mitigation_tradeoff.csv).

## The audit toolkit

The audit is packaged as a Claude Code plugin. Claude adapts it to your codebase (finds the retriever and the LLM call, adds logging, builds he/she twin questions with you, applies the fixes); fixed scripts do all the statistics, so every audit measures the same thing. On this thesis's data the scripts reproduce the audit exactly (166 of 166 values).

```
/plugin marketplace add dhthz/Legal-RAG-LLM-bias-thesis
/plugin install rag-bias-audit@rag-bias-audit
```

Then, in your RAG project, ask Claude to audit it for gender bias, or run `/rag-bias-audit:bias-audit`. Details: [`toolkit/README.md`](toolkit/README.md).

## Repository

| Path | Contents |
|---|---|
| [`src/`](src/) | The RAG pipeline (chunking, embedding, retrieval, prompting, logging) and the fixes as plain functions (`src/mitigation/`) |
| [`scripts/`](scripts/) | Runnable steps: index building, query construction, the audit statistics and the robustness checks (`scripts/eval/`) |
| [`toolkit/`](toolkit/) | The Claude Code plugin |
| [`docs/bias_audit/`](docs/bias_audit/) | Results, charts and the toolkit overview |
| [`tests/`](tests/) | End-to-end test and the frozen-retrieval tripwire |
| [`FROZEN_BASELINE_MANIFEST.json`](FROZEN_BASELINE_MANIFEST.json) | Every pipeline setting, the index checksum and the prompt variants |

The dataset, the FAISS indices and the run logs are not in the repository (size).

## Running it

Python 3.12, CUDA, and Ollama with `mistral` pulled.

```bash
python -m venv thesis_venv && source thesis_venv/bin/activate
pip install -r requirements.txt
python -m scripts.chunking_and_embeddings.run_chunking_pipeline   # build the chunked dataset and the FAISS index (GPU, ~30 min)
python -m tests.test_frozen_retrieval        # retrieval must match the frozen baseline exactly
python -m tests.test_pipeline                # one question end to end
```

Audit statistics, after the runs in `logs/bias_audit/`:

```bash
python -m scripts.eval.bias_statistics --log <run log> --out-dir <stats dir>
python -m scripts.eval.multi_run_stability && python -m scripts.eval.llm_output_bias
python -m scripts.eval.export_audit_results
```

## References

1. Sun, T., Webster, K., Shah, A., Wang, W. Y., & Johnson, M. (2021). They, Them, Theirs: Rewriting with Gender-Neutral English. [arXiv:2102.06788](https://arxiv.org/abs/2102.06788)
2. Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2019). Counterfactual Fairness in Text Classification through Robustness. *AIES 2019*. [doi:10.1145/3306618.3317950](https://dl.acm.org/doi/10.1145/3306618.3317950)
3. Zhou, Z., Liu, T., Bai, R., Gao, J., Kocaoglu, M., & Inouye, D. I. (2024). Counterfactual Fairness by Combining Factual and Counterfactual Predictions. *NeurIPS 2024*. [arXiv:2409.01977](https://arxiv.org/abs/2409.01977)
4. Deng, W., Chen, B., Zhao, B., Zhang, C., Li, X., & Thrampoulidis, C. (2024). LLM-Assisted Content Conditional Debiasing for Fair Text Embedding. [arXiv:2402.14208](https://arxiv.org/abs/2402.14208)
5. Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. *NeurIPS 2023*. [arXiv:2306.03819](https://arxiv.org/abs/2306.03819)
6. Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS 2016*. [arXiv:1607.06520](https://arxiv.org/abs/1607.06520)
7. Kim, T., Springer, J., Raghunathan, A., & Sap, M. (2025). Mitigating Bias in RAG: Controlling the Embedder. *Findings of ACL 2025*. [arXiv:2502.17390](https://arxiv.org/abs/2502.17390)
8. Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018). Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. *NAACL 2018*. [arXiv:1804.06876](https://arxiv.org/abs/1804.06876)
9. Tang, R., Zhang, X., Ma, X., Lin, J., & Ture, F. (2024). Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. *NAACL 2024*. [arXiv:2310.07712](https://arxiv.org/abs/2310.07712)
10. Kaneko, M., & Bollegala, D. (2021). Debiasing Pre-trained Contextualised Embeddings. *EACL 2021*. [arXiv:2101.09523](https://arxiv.org/abs/2101.09523)

Built with: [FAISS](https://github.com/facebookresearch/faiss), [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1), [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b) through [Ollama](https://ollama.com), [spaCy](https://spacy.io), GPT-2 (for the rewrite), and the LEACE authors' [concept-erasure](https://github.com/EleutherAI/concept-erasure) library.

## Status

- Audit and fixes: complete.
- Toolkit: v0.1, reproduces the thesis audit exactly; testing on a second, unseen RAG system in progress.
- Gender is the validated attribute. The system is English-only and ECtHR-only, with one embedder and one LLM.
