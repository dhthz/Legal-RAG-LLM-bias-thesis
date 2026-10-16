# tests/

Run from the repository root with `python -m tests.<name>`.

| File | What it checks | Needs |
|---|---|---|
| `test_frozen_retrieval.py` | Tripwire: retrieval of the 335 audit questions is identical to the frozen audit run, with no fix and with a no-op fix that exercises every hook | index and data, GPU |
| `test_permutation.py` | The permutation fix without the LLM: the first order is the frozen prompt, every order answered once, the vote follows its rule | nothing |
| `test_pipeline.py` | One question end to end, after 8 pilot questions | index and data, GPU, Ollama |
| `test_retrieval.py` | Not a test: an interactive retrieval explorer (`python -i tests/test_retrieval.py`) | index and data |

| Data file | What it is |
|---|---|
| `frozen_retrieval_reference.jsonl` | The reference for the tripwire: one SHA256 per audit question of its retrieved cases, chunks and scores (`--write-reference` rebuilds it from the audit log) |
| `smoke_queries_pilot.jsonl`, `smoke_results_pilot.jsonl` | The 8 pilot questions and their last results |
