# Mode 1: Programmatic validation
---

## Objective
Apply cheap, deterministic checks **and** lexical/semantic scores to seed instruction JSONL before LLM/human review. Hard-fail on schema/IPC errors and very low score floors; otherwise keep examples with scores attached for later stages.

## Scope
All three tasks under `data/derived/instruction_generation/{model_slug}/<task>/`.

## Inputs
Alpaca-style JSONL (`part-*.jsonl` / `.jsonl.gz`).

## Structural checks

### Shared
* Required keys: `task`, `application_number`, `instruction`, `input`, `output`, `meta`
* Non-empty `instruction`, `input`, `output`

### IPC reasoning — IPC regex
```text
IPC_RE = ^[A-H][0-9]{2}[A-Z](?:[0-9]{1,4}/[0-9]{2,6})?$
```
Normalize: uppercase, strip spaces.

Per record:
1. `meta.primary_ipc` matches `IPC_RE`
2. `output` parses as `Classification: <code>\nJustification: <body>`
3. Classification code == `meta.primary_ipc` (normalized)
4. Any other `IPC_RE` matches inside the justification body must equal the primary code (strict)

### Light task checks
* **abstract_drafting:** length sanity vs input
* **mrc:** `input` is `Question: …\n\nClaims: …`; question contains `?`; answer shorter than claims

## Lexical scores

| Task | Metric | Pair |
|------|--------|------|
| `ipc_reasoning` | ROUGE-L F1 | Justification body vs `input` |
| `abstract_drafting` | ROUGE-L F1 | abstract (`output`) vs claims (`input`) |
| `mrc` | Best-span token-F1 | answer vs sliding windows of claims; verbatim containment short-circuits to 1.0 |

Also record: `len_input_tokens`, `len_output_tokens`, `compression_ratio = len_out / len_in`.

Library: `rouge-score` (ROUGE-L F1).

## Semantic similarity

| Task | Pair |
|------|------|
| `ipc_reasoning` | Justification vs `input` |
| `abstract_drafting` | abstract vs claims |

Not computed for `mrc` (short extractive answers vs the full claim set are a weak quality signal).

* **Model:** `nomic-ai/nomic-embed-text-v1.5` (8,192-token context; document prefix `search_document:` on both sides)
* **Score:** cosine similarity of mean-pooled embeddings
* **Long text:** truncate each side to 8,192 tokens. That covers typical claim bundles in this corpus (median ~560–750 MiniLM-equivalent tokens; p95 ~2k). MiniLM’s 256/512 window truncated most IPC/abstract claim sides.

## Soft floors (quarantine)
Fail only if:
* semantic cosine `< 0.15` (IPC reasoning / abstract drafting only), or
* ROUGE-L F1 `< 0.02` (IPC reasoning / abstract drafting), or
* MRC: best-span token-F1 `< 0.5`

Borderline scores remain in the pass set with `meta.validation` filled in.

## Outputs
Runnable stage: `dataset-validation/` → `scripts/validate_instruction_data.py`

* `passed/part-*.jsonl.gz` — record + `meta.validation` scores
* `rejected/part-*.jsonl.gz` — failures + `meta.validation.failed_rules`
* `report.json` — means, fail counts, simple histograms
