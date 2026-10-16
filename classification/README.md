# classification

**Stage:** labeling and enrichment of already-fetched records.

Reads base + API-enriched tables from `data/`, applies taxonomies / rules / models, and writes labeled outputs to `data/derived/` and `data/processed/`.

## Rules

- No IP Australia (or other patent-source) API fetching here — that belongs in `scrape/`.
- Keep label definitions in `schemas/` separate from application code in `src/`.
- Large model weights belong under `models/` (gitignored) or external storage; document how to obtain them.

## Layout

- `src/` — labeling pipelines, heuristics, model inference
- `config/` — classifier paths and inference knobs
- `schemas/` — taxonomy and label definitions (cite-friendly)
- `models/` — trained artifacts or pointers (bulk files ignored)
- `requirements-patentbert.txt` — **separate** TF1 env deps (do not install into root `.venv`)

## PatentBERT (CPC subclass, claim-level)

Multi-label CPC-subclass inference from the released PatentBERT checkpoint (paper: https://doi.org/10.1016/j.wpi.2020.101965). One prediction per row of `data/derived/patent_search_text/claims.csv` (no claim concatenation).

| | |
|---|---|
| Config | `config/patentbert.yaml` |
| Modules | `src/download_patentbert.py`, `src/run_patentbert.py`, `src/patentbert/` (TF1) |
| Checkpoint | `models/patentbert/` (`model.ckpt-181172*`, `labels_group_id.tsv`, `vocab.txt`, `bert_config.json`) |
| Writes | `data/derived/patentbert/` — `row_map.csv`, `predict_result.txt`, `predictions.csv` (add `--gzip` for `*.gz`) |
| Streaming | Claims CSV is read row-by-row; TF1 sees at most `infer.chunk_size` rows per invoke (default `1000`) |

**`multi_hot_threshold`** (default `0.3`): sigmoid probability cutoff. CPC subclasses with score `>` threshold are kept. Lower → more labels per claim; higher → fewer.

### Environment

PatentBERT is **TensorFlow 1.15** (not TF2). It does **not** install into the repo root `.venv` (Python 3.13 / Apple Silicon only sees TF 2.x on PyPI).

Use a **dedicated** env:

| Platform | How |
|----------|-----|
| Linux x86_64 | Python 3.7–3.8 venv + `pip install tensorflow==1.15.5` then `pip install -r classification/requirements-patentbert.txt` |
| macOS Apple Silicon | Conda **osx-64** (Rosetta) Python 3.7 — see below |
| Colab | Runtime with TF1 / legacy notebook env |

#### Apple Silicon (this machine)

```bash
# once: Rosetta + Miniconda (if needed)
/usr/sbin/softwareupdate --install-rosetta --agree-to-license
brew install --cask miniconda
# restart shell, then: conda init zsh && exec zsh

conda create -y -n patentbert-tf1
conda activate patentbert-tf1
conda config --env --set subdir osx-64
conda install -y python=3.7
conda install -y -c apple tensorflow=1.15
# if import errors:
#   conda install -y tensorflow-estimator=1.15.1
# then pip (pins numpy 1.17–1.18 for matplotlib + TF1):
pip install -r classification/requirements-patentbert.txt
# if estimator still wrong after pip:
#   conda install -y tensorflow-estimator=1.15.1

python scripts/download_patentbert.py
python scripts/run_patentbert.py --max-predictions 5
python scripts/run_patentbert.py --gzip --chunk-size 1000
python scripts/run_patentbert.py --gzip
```

Prepare TSV only (works from root `.venv`; no TF1 needed):

```bash
python scripts/run_patentbert.py --prepare-only
python scripts/run_patentbert.py --prepare-only --gzip
```

## IPC LCPN (step-by-step IPC classification)

This is the **eventual final model** path: the **fine-tuned student** classifying a patent by walking the IPC tree.

How it works:

1. Start at the top of the tree.
2. Show the model the patent text and the list of child IPC codes at the current place.
3. The model picks one child, or says `STOP` if none fit.
4. Move to that child and repeat until `STOP` or there are no children left.
5. Print the path of codes it chose.

Code lives in `src/ipc_lcpn/`. Settings live in `config/ipc_lcpn.yaml`. The IPC tree comes from the top-level `ipc/` package. The prompt format matches the `ipc_lcpn` training task in `instruction-generation/`.

Which model to call is set in the YAML under `llm.provider`:

- `local` — your machine (e.g. Ollama). Default. Good for a free quick test.
- `openrouter` — cloud API (needs an API key).
- `hf` — load the Hugging Face student model (same stack as `sft/`), optionally with a trained adapter path.

Run one patent from the command line:

```bash
.venv/bin/python scripts/run_ipc_lcpn.py \
  --abstract "A hand tool for digging soil." \
  --claim-1 "A spade comprising a blade and a handle."
```

Optional: add `--json-out some_file.json` to save the full step-by-step trace.

Tests:

```bash
.venv/bin/python -m unittest classification.tests.test_ipc_lcpn -v
```

## Inputs / outputs

- **Reads:** enriched tables from `data/raw/` and/or `data/derived/` (after scrape), plus the base application dump as needed
- **Writes:** `data/derived/`, `data/processed/`
- **IPC LCPN runner:** takes abstract / claim text on the CLI; can write a JSON trace with `--json-out`

## How to run

See the PatentBERT and IPC LCPN sections above.
