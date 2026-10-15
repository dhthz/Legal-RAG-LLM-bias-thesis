---
name: bias-audit
description: Audit a retrieval-augmented generation (RAG) system for gender bias and apply fixes. Use when the user asks to check their RAG, search or question-answering system for gender bias, to test whether he/she wording changes the retrieved documents or the answers, or to apply and compare bias mitigations (blind query, PCF, LEACE). Works in the user's own codebase; the statistics are fixed, validated scripts.
---

# Gender bias audit for a RAG system

You carry a fixed audit protocol into the user's codebase. **You do the codebase work** (find the retriever and the LLM call, add logging, apply fixes). **The bundled scripts do all the statistics**: run them, never rewrite or re-implement them, and never compute bias numbers yourself. The measurement is the one validated in an MSc thesis on a legal RAG system (it reproduces that audit exactly).

Work in 7 stages. Never change the user's retrieval, prompts or model settings without saying so first.

Scripts live in `${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts/`, fixes in `${CLAUDE_PLUGIN_ROOT}/fixes/`. Run them with the Python of the user's system (their venv).

## How you work with the user

- **Ask with options, not open questions.** Every decision and every end-of-stage confirmation goes through the multiple-choice question tool: your recommended option first, marked "(Recommended)", 2–4 options in total. The user should rarely need to type.
- **Keep messages short:** what you found or changed (file and line), what comes next, the question.
- **Before anything that runs a model, give the time, and the cost if it is a paid API** (see "Time and cost"). Never start a run the user has not approved with these numbers in front of them.
- **When you offer a fix or a test, use its explanation from "Fix explanations" word for word**, then add one line on what it means for their code.

## Files you create

Keep the user's project tidy. Everything the audit makes goes in one folder, and the only other changes are the fixes package and the switches in their code:

```
bias_audit/
  README.md            what is here, the model and settings audited, how to rerun (update after each stage)
  run_audit.py         the logging runner (stage 4)
  questions.jsonl      the user's questions; from make_pairs.py: audit_queries.jsonl, query_pairs.jsonl,
                       neutral_versions.jsonl (for the fixed-documents test), twins_left_out.jsonl
  doc_labels.jsonl     the labels (stage 2); doc_topics.jsonl, the optional topics (stage 2)
  runs/                run_log.jsonl, baseline_reference.jsonl, tripwire checks, progress output (*.out)
  results/             results.json, report.md, figures/, report.pdf (optional, stage 7)
bias_fixes/            the copied fixes package, only if fixes are applied (stage 6)
```

- No scratch files in the project root or elsewhere in their repo; temporary checks go in `bias_audit/runs/`.
- Their app must never read files from `bias_audit/` at run time: a value it needs (such as the PCF prior) goes into their code as a constant, with a comment saying where it came from.
- Match their code style; add no docstrings or comments beyond what their code has.
- If the project is a git repository, offer to add `bias_audit/runs/` to `.gitignore` (run logs are large).

## Time and cost

Before every batch of model calls (stage 4 baseline, stage 7 fixes):

1. **Count the calls:** questions × runs × settings (e.g. 196 questions × 4 runs × 2 fixes = 1,568).
2. **Time them the way the runner will run:** in one process, discard the first 2 calls (loading), then time 5. Give the total and the clock time it should finish, and say what the estimate is based on. After the first progress check, correct the estimate if it is off by more than a quarter.
3. **If the LLM or the embedder is a paid API:** take the tokens per call from the API's usage fields on those 5 calls, multiply by the number of calls and the price per token, and give the cost. Ask the user for the price if you do not know it; never guess silently. Ask with options such as: run everything / run fewer questions or runs / stop.
4. **For API models:** retry on rate limits with backoff, log failures as `generation_error` (never drop the row), keep the runner resumable, and use a pinned model version (a dated snapshot, not an alias such as "latest"). Record the model and its settings in `bias_audit/README.md`; the baseline and the fixes must use the same one.

## Before you start

Check the Python packages in the environment the user's system runs in:

```bash
python -c "import numpy, scipy, jsonschema, matplotlib"   # audit scripts and charts
python -c "import spacy, torch, transformers; spacy.load('en_core_web_sm')"   # question twins and the fixes
```

If something is missing, install only the missing packages, with the version lines from `${CLAUDE_PLUGIN_ROOT}/requirements.txt` (they are minimum versions), and never upgrade or downgrade what the user already has. Ask first.

## The three audit files

Every script reads only these files (JSON Lines, one object per line). Schemas: `scripts/schemas/`.

1. **Run log** (`bias_audit/runs/run_log.jsonl`), one row per question per run:
   `run_id` (repeat name, e.g. "A"), `query_id`, `query` (the exact text sent to retrieval, after any rewrite), `arms` (list of fixes switched on, `[]` for the baseline), `retrieved` (the documents handed to the LLM: `doc_id`, `rank` from 1, optional `score`, optional `shown_ids` = ids of the passages of that document the LLM actually saw, in prompt order), `cited_doc_ids` (documents the answer cites; `[]` if none), `response`, optional `generation_error`.
2. **Document labels** (`bias_audit/doc_labels.jsonl`), one row per document in the **whole collection**: `doc_id`, `attribute` ("gender"), `value` (e.g. "Male", "Female", "Unknown"). Unlabelled documents are listed with "Unknown": the rows also give the collection's mix.
3. **Question pairs** (`bias_audit/query_pairs.jsonl`), one row per question: `query_id`; twins add `pair_id` and `variant` ("male", "female", "neutral"); a question about a person can carry `attribute_value`. Built by `make_pairs.py`, not by hand.

## Stage 1. Understand the system

Read the repo. Show the user, with file and line: where documents are retrieved (embedder, index, top-k), where the prompt is built and the LLM is called, how the answer names the documents it uses (citations), where the document collection lives, and whether the LLM and the embedder are local or a paid API. If answers do not cite documents, agree with the user how to extract `cited_doc_ids`. If the search is a hosted service, check whether it returns scores for given document ids (PCF needs this) and lets you change the query vector before searching (LEACE needs this), and say which fixes are possible. Confirm.

## Stage 2. Labels

The audit needs the gender of the person each document is about. Ask the user for existing labels (metadata, a CSV). If there are none, propose a labelling method and **validate it**: label the collection, have the user check a random sample of 50–100 documents by hand, and report the agreement before using the labels. Write `bias_audit/doc_labels.jsonl` covering every document.

**Optional relevance check.** Ask, with options: "Do your documents have a topic or category (a field such as profession, product type or legal article)? If yes, the audit can also check that the fixes keep the search on topic." If yes, write `bias_audit/doc_topics.jsonl` in the labels format (`doc_id`, `attribute` such as "topic", `value`; a document with several topics gets several rows), and in stage 3 give each question the topics a relevant document should have (`topics`, a list). Show the user the question-to-topic mapping and flag your judgement calls before using it. If no, the report keeps its note that relevance is not measured.

Confirm.

## Stage 3. Questions and twins

Get 100–400 realistic questions from the user (logs, a test set, or written with them), in `bias_audit/questions.jsonl` (`query_id`, `query`; optional `attribute_value`, `topics`). Questions that describe a person with he or she can be twinned. Then:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts/make_pairs.py" --questions bias_audit/questions.jsonl \
  --groups Male Female --out-queries bias_audit/audit_queries.jsonl --out-pairs bias_audit/query_pairs.jsonl
```

`--groups` are the he and she values exactly as in the labels file. Aim for at least 40 twinned questions (fewer cannot show an effect); `--max-twins` limits them. The swap changes pronouns and gendered person nouns together ("my nephew… he" becomes "my niece… she", "his wife" becomes "her husband"), so each twin carries one gender cue. A question with a gendered word the swap cannot change (e.g. "actress") is not twinned and is listed in `twins_left_out.jsonl`, with the word: tell the user how many and which, and offer to reword them with their approval. Show the user 10 twins side by side as a table and ask them to approve.

## Stage 4. Log and run the baseline

Write `bias_audit/run_audit.py`, a thin runner that imports the user's own retrieval, prompt and LLM functions and writes run-log rows in the format above. Do not change retrieval, the prompt or the model settings. It skips rows already logged (so it can resume), prints progress with an estimated time left, and has a retrieval-only mode for the tripwire. Then:

1. Run every question in `bias_audit/audit_queries.jsonl` **4 times** (`run_id` A, B, C, D) with no fixes. Repeats measure how much answers change by chance (the noise floor). Give the time (and cost) first, as in "Time and cost". Run it in the background and check on progress.
2. Validate:
   ```bash
   python "${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts/validate.py" --run-logs bias_audit/runs/run_log.jsonl \
     --labels bias_audit/doc_labels.jsonl --pairs bias_audit/query_pairs.jsonl
   ```
   Fix what it reports in the logging code, not by editing the files.
3. Keep a copy: `cp bias_audit/runs/run_log.jsonl bias_audit/runs/baseline_reference.jsonl`. It is the reference for the tripwire in stage 6.

Confirm the logging change and the run.

## Stage 5. Measure

```bash
S="${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts"
python "$S/audit.py" --run-logs bias_audit/runs/run_log.jsonl --labels bias_audit/doc_labels.jsonl \
  --pairs bias_audit/query_pairs.jsonl --groups Male Female --out bias_audit/results/results.json \
  [--topics bias_audit/doc_topics.jsonl]   # only with the optional relevance check
python "$S/figures.py" --results bias_audit/results/results.json --out-dir bias_audit/results/figures
python "$S/report.py" --results bias_audit/results/results.json --out bias_audit/results/report.md \
  --figures bias_audit/results/figures
```

Explain `report.md` in plain language, in this order: (1) does the he/she wording change what the **search** returns (the top-document flip test), (2) does the **LLM** add a bias of its own (identical input vs repeat noise) or amplify the search through the **order** of documents, (3) the background mixes. Base "where the bias enters" on the report's twin tests only. Say clearly when an effect is not detected, give the noise floor, and say how large an effect the test could have missed (the report gives it): "not detected" with few twins is weak evidence, not proof of no bias. Point the user to the charts in `bias_audit/results/figures/`.

**If the report says the LLM's own effect could not be measured** (no twin pair got identical input, which happens when the search bias is strong), or fewer than 20 pairs got identical input, offer the LLM test with fixed documents (explanation below). Add a mode to `run_audit.py`: for each twin pair, search once with its neutral version (`bias_audit/neutral_versions.jsonl`), then answer the he and the she question with those same documents in the same order, using the user's own prompt and LLM functions. Log the twin questions only, as `query` the twin question, with `arms: ["fixed_documents"]` and `retrieved` = the neutral version's documents, 4 runs, into the same run log. Give the time (and cost) first. Then rerun `audit.py`, `figures.py` and `report.py`: the report gets a section "The LLM with fixed documents", and its LLM verdict replaces "not measurable". It does not change the user's app, so no tripwire is needed.

Then ask whether to continue to fixes, offering the fixes that match where the bias enters.

## Stage 6. Apply fixes

All three fixes act on the search and need no retraining. Offer them with their explanations below. Copy `${CLAUDE_PLUGIN_ROOT}/fixes/` into the user's repo as `bias_fixes/` (`__init__.py` and the modules of the chosen fixes; PCF also needs `neutral_rewrite.py`) and call it from their code. Each fix is behind a switch, off by default, that writes its name into `arms`.

- **Blind query** (`neutral_rewrite.py`, `neutralize`): rewrite the question before retrieval and send the rewritten question to the LLM too.
- **PCF** (`pcf.py`, `pcf_rank`): needs two functions from the user's retriever: `search(text, k)` returning ids and `score(text, ids)` returning the retriever's own scores. The prior: compute `prior_from_labels("bias_audit/doc_labels.jsonl")` once and write the value into their code as a constant.
- **LEACE on the query** (`leace.py`): fit it with `fit_query_eraser` on the user's embedder's vectors of at least 200 he/she twin questions that are **not** audit questions (build them with `make_pairs.py` from other questions), check it with `probe_auc` on held-out twins (erased AUC near 0.5), save it with `save_eraser` next to their code (not in `bias_audit/`), and apply `erase` to each query vector before the search.

After adding a fix with it switched **off**, run the baseline questions once, retrieval only, into `bias_audit/runs/`, and check that nothing changed:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts/tripwire.py" --reference bias_audit/runs/baseline_reference.jsonl \
  --check bias_audit/runs/tripwire_check.jsonl
```

It must say PASSED. If it fails, the fix leaks into the baseline: find and fix that before measuring. Show the user the code change and the tripwire output, and confirm.

## Stage 7. Re-measure and compare

Give the time (and cost) for the fix runs, then run every question 4 times with each fix on (rows with `arms: ["pcf"]` etc.) into the same run log, and run `validate.py`, `audit.py`, `figures.py` and `report.py` again. The report now ends with a table and a chart comparing each fix with the baseline: top-document changes, how often the he and she answers cite different documents, how much of the search stayed the same, and whether the documents a fix dropped matched the question's gender more than those it added. Explain it. With the relevance check, explain its section too (a fix that costs relevance lowers the share on topic, with more questions lower than higher; a topic match is only a rough stand-in for relevance). Without it, tell the user that the report does not measure relevance: before keeping a fix they should check the search still finds the right documents (e.g. recall@k on questions with a known answer). The user decides what to keep. Finish `bias_audit/README.md`: the files, the model and settings, the fixes and their switches, and the commands to rerun.

Then offer, as an option, the full report as one PDF with the charts (for sharing, a thesis or a review):

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/bias-audit/scripts/export_report.py" --report bias_audit/results/report.md \
  --out bias_audit/results/report.pdf
```

It needs Chrome, Chromium or Edge on the machine (found automatically, also the Windows browser from WSL); without one it keeps `report.html` for the user to print from any browser. Do not write the PDF or the HTML yourself: the script builds both from the report. If your environment can publish pages, you may offer that too, but only on request, since it sends the audit's data off the machine.

## Fix explanations

Use these word for word when you offer a fix or the fixed-documents test.

**LLM test with fixed documents (a test, not a fix).** The audit could not tell whether the LLM itself treats he and she differently, because the search gave the twins different documents. This test gives both versions of each question exactly the same documents (found with a they version of the question), so the pronoun is the only difference the LLM sees, and compares their answers with the noise of asking twice. What changes in your code: nothing in your app, only an extra mode in the audit runner. Cost: 4 runs of the twin questions.

**Blind query.** Before searching, the question is rewritten so it no longer says he or she ("she wants to start her firm" becomes "they want to start their firm"). The search and the LLM both get the rewritten question, so the pronoun cannot steer either. What changes in your code: one switch before the search, and a small local model (GPT-2, on CPU) for the rewrite. Trade-off: the user's question is reworded before the LLM sees it.

**PCF (counterfactual ranking).** Each search runs twice, once with the question and once with he and she swapped, and documents are ranked by a weighted average of the two scores (the weights come from your collection's gender mix). Both versions of a question then get the same ranking. What changes in your code: two small helpers next to your search (find ids, score ids), and each search costs twice as much. The question text is not changed. Trade-off: it only acts on questions that use he or she.

**LEACE on the query.** The he/she direction is removed from the question's vector before searching, with a linear projection learned from 200+ he/she question pairs that are not audit questions. What changes in your code: one line before your vector search, and a small saved file with the projection. Trade-off: it needs that extra question set, a check that the direction is really gone, and access to the query vector (some hosted search services do not allow it).

## Rules

- Never edit the scripts, schemas or the numbers they produce; if a script refuses the files, fix the logging.
- Never claim an effect the report does not show, and never hide one it does.
- Gender (he/she) is the validated attribute. Other attributes only with the user's own twin questions and labels, and say they are not validated.
