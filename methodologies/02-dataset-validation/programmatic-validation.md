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
4. IPC-shaped tokens in the justification must be the primary code **or a coarser ancestor** of it (subclass `G05D` vs `G05D1/00`, parent subgroup `C12Q1/68` vs `C12Q1/6876`). Unrelated or sibling symbols fail `conflicting_ipc_in_justification`.
5. `primary_ipc` has a WIPO `definition_statement` in the catalog (`data/ipc-codes/ipc_codes_20260101.jsonl`); otherwise `wipo_definition_missing`
6. `input` parses as `Abstract: …\n\nClaims: …`

### Light task checks
* **abstract_drafting:** `input` starts like claim 1 (`1.` / `1)`); `output` is not a numbered claim list; abstract token count `<` claims token count (compression inversion); abstract not longer than `max(2 × claims, 50_000)` chars
* **mrc:** `input` is `Question: …\n\nClaims: …`; question contains `?`; answer shorter than claims

## Lexical scores

| Task | Metric | Pair |
|------|--------|------|
| `ipc_reasoning` | ROUGE-L F1 | Justification vs WIPO title + definition (floor + copy ceiling). Claims-side ROUGE is recorded, not a fail rule. |
| `abstract_drafting` | ROUGE-L F1 | abstract (`output`) vs claims (`input`) — recorded, not a fail rule (formula-dense claims break LCS) |
| `mrc` | Best-span token-F1 | answer vs sliding windows of claims; verbatim containment short-circuits to 1.0 |

Also record: `len_input_tokens`, `len_output_tokens`, `compression_ratio = len_out / len_in`.

Library: `rouge-score` (ROUGE-L F1).

## Semantic similarity

| Task | Pair |
|------|------|
| `ipc_reasoning` | Justification vs WIPO title + definition **and** vs claims |
| `abstract_drafting` | abstract vs claims |

Not computed for `mrc` (short extractive answers vs the full claim set are a weak quality signal). IPC uses two cosine pairs: WIPO grounding (is the rationale about the assigned place) and claims (is it about this invention). A justification that is mostly the WIPO text (ROUGE-L vs title+definition `> 0.60`) fails as a near-copy. Abstract drafting cosine is pairing only (gold abstract vs claims); official abstracts are not judged for writing quality.

* **Model:** `nomic-ai/nomic-embed-text-v1.5` (8,192-token context; document prefix `search_document:` on both sides)
* **Score:** cosine similarity of mean-pooled embeddings
* **Long text:** truncate each side to 8,192 tokens. That covers typical claim bundles in this corpus (median ~560–750 MiniLM-equivalent tokens; p95 ~2k). MiniLM’s 256/512 window truncated most IPC/abstract claim sides.

## Soft floors (quarantine)
Fail only if:
* IPC reasoning: WIPO cosine `< 0.55` or WIPO ROUGE-L F1 `< 0.08` or `> 0.60` (near-copy), or claims cosine `< 0.50`, or
* abstract drafting: Nomic cosine `< 0.40` (topic mismatch). ROUGE is not a fail rule. Compression inversion (`abstract` tokens `>=` claims tokens) is a structural fail (`abstract_not_shorter_than_claims`), or
* MRC: best-span token-F1 `< 0.5`

Borderline scores remain in the pass set with `meta.validation` filled in.

## Outputs
Runnable stage: `dataset-validation/` → `scripts/validate_instruction_data.py`

* `passed/part-*.jsonl.gz` — record + `meta.validation` scores
* `rejected/part-*.jsonl.gz` — failures + `meta.validation.failed_rules`
* `report.json` — means, fail counts, simple histograms
