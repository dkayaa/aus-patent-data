# AGENTS

Guidance for Cursor agents working in this repository.

## Python environment

- **`.venv/` at the repo root is the authoritative Python environment.**
- Prefer `.venv/bin/python` and `.venv/bin/pip` for installs and script runs.
- Do not invent a second venv, rely on a global interpreter, or use pyenv/system Python when `.venv` exists.
- If `.venv` is missing, create it with `python3 -m venv .venv` and install from `requirements.txt` before running pipeline code.

## Pipeline stages

See the root `README.md` “For Cursor agents” section for stage boundaries (`data/`, `scrape/`, `classification/`) and hard rules.
