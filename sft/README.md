# sft

Prepare flat SFT datasets from **frozen** evaluation splits, then run per-dataset QLoRA (TRL / PEFT) on CUDA.

This stage does **not** invent train/val/test membership. It only transforms rows already assigned by `scripts/split_eval_data.py` / `evaluation/src/split.py` (global `application_number` holdout, temporal test pool, seed 42). Use the same `generator_slug` end-to-end as baselines so SFT and baseline tables share test apps.

**Default generator:** `qwen/qwen3-235b-a22b-2507` (`sft/config/sft.yaml`). Override with `--generator` when needed.

## Datasets (flat)

| SFT dataset id | Reads eval split task | Transform |
|----------------|----------------------|-----------|
| `ipc_reasoning_full` | `ipc_reasoning` | `output` unchanged (Classification + Justification) |
| `ipc_reasoning_classification_only` | `ipc_reasoning` | keep `Classification:` only; skip + count unparseable |
| `abstract_drafting` | `abstract_drafting` | pass-through |
| `mrc` | `mrc` | pass-through |

Full claims stay in `input` as stored (no claim-1 truncation). One QLoRA run per `--dataset` (no pooled multi-task adapter in this stage).

## Layout

```text
sft/
  config/sft.yaml
  src/prepare.py
  src/run_train.py
  README.md

scripts/prepare_sft_data.py
scripts/run_sft.py
requirements-sft.txt
```

Outputs:

```text
data/derived/sft/{generator_slug}/
  ipc_reasoning_full/{train,val,test}.jsonl.gz
  ipc_reasoning_full/manifest.json
  …
data/derived/sft/runs/{run_name}/
```

## Prepare

```bash
.venv/bin/python scripts/split_eval_data.py --generator qwen/qwen3-235b-a22b-2507
.venv/bin/python scripts/prepare_sft_data.py --dataset all
# defaults to qwen/qwen3-235b-a22b-2507; override with --generator …
# --dataset ipc_reasoning_full|ipc_reasoning_classification_only|abstract_drafting|mrc
# --limit N   # caps train only; val/test unchanged
```

If a needed eval split dir is missing, the script exits and tells you to run `split_eval_data.py`.

## Train (CUDA / DigitalOcean)

Install GPU stack on the droplet (keep root `requirements.txt` free of these pins):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-sft.txt
```

```bash
.venv/bin/python scripts/run_sft.py --dataset ipc_reasoning_full
.venv/bin/python scripts/run_sft.py --dataset ipc_reasoning_classification_only
.venv/bin/python scripts/run_sft.py --dataset abstract_drafting
.venv/bin/python scripts/run_sft.py --dataset mrc
```

Trains on `train` + validates on `val`; never fits on `test`. Default student: `meta-llama/Llama-3.1-8B-Instruct` (`sft/config/sft.yaml`). Fails clearly without CUDA / bitsandbytes.

## Split alignment

Reusing eval splits satisfies temporal holdout + no train/test app overlap (see `evaluation/README.md`, `methodologies/03-fine-tuning-and-evaluation/`, `holes-gemini.md` Hole 4 structure). Prepare does **not** re-sample, change the YAML cutoff (`2024-01-01`), or add IPC-section stratification.
