# src/

The RAG system under audit and the fixes, as importable library code (the runnable steps are in [`scripts/`](../scripts/)). Every setting comes from [`FROZEN_BASELINE_MANIFEST.json`](../FROZEN_BASELINE_MANIFEST.json).

```
question → retrieve chunks (FAISS) → group into cases → prompt with the top 3 cases → Mistral 7B → parse cited cases → log
```

| File | What it does |
|---|---|
| `chunking/legal_chunker.py` | Splits a judgment's facts into paragraph chunks of about 150 words, each prefixed with the case intro |
| `embeddings/chunk_embedder.py` | Embeds chunks with nomic-embed-text-v1, builds the FAISS index, retrieves chunks and groups them into cases (max similarity) |
| `llm/config.py` | `PipelineConfig`, loaded from the manifest |
| `llm/client.py` | Mistral 7B through Ollama's native chat API (honours the context length) |
| `llm/prompts.py` | The legal prompt: the question, the top cases with two chunks each, and the JSON answer format |
| `rag/pipeline.py` | `RAGPipeline`: retrieve, group, prompt, generate, parse and log one interaction per question |
| `rag/response_parsing.py` | Reads the cited cases and articles from the answer, checked against the retrieved cases |

## mitigation/: the fixes

Each fix is a plain function plus a thin `MitigationArm` that hooks it into the pipeline. With no fix the pipeline is exactly the frozen system (checked by [`tests/test_frozen_retrieval.py`](../tests/test_frozen_retrieval.py)). The rewrite, PCF and LEACE cores are shared with the plugin in [`toolkit/fixes/`](../toolkit/fixes/).

| File | Fix | Acts on |
|---|---|---|
| `__init__.py` | Registry of the fixes by name (`--arms` in the scripts) | |
| `base.py` | `MitigationArm` hooks and `retrieve_with_arms` | |
| `neutral_rewrite.py` | Blind query: gender-neutral rewrite of the question (Sun et al. 2021; Garg et al. 2019) | question text |
| `pcf.py` | PCF: averages the retrieval scores of the he and she versions (Zhou et al. 2024) | retrieval scores |
| `leace.py` | LEACE: erases gender from the vectors; on case gender (first build) or on the question's pronoun (`leace_query`) (Belrose et al. 2023) | vectors |
| `blind_index.py` | Blind index: searches the corpus rewritten gender-neutrally | stored chunks |
| `finetuned_embedder.py` | Fine-tuned embedders: counterfactual twins (Kim et al. 2025), its control, content conditional debiasing (Deng et al. 2024) | embedder |
| `permutation.py` | Permutation self-consistency: answers for every order of the cases and aggregates (Tang et al. 2024) | generation |
