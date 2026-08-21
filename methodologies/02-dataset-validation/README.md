# 02 — Dataset validation

Validate seed instruction-tuning JSONL from [`01-instruction-data-generation`](../01-instruction-data-generation/) before using it for SFT.

| Mode | Doc | Scope |
|------|-----|--------|
| Programmatic | [programmatic-validation.md](programmatic-validation.md) | Schema/IPC + lexical (ROUGE-L / token-F1) + semantic (Nomic Embed cosine); see `dataset-validation/` |
| LLM-as-a-judge | [llm-as-a-judge.md](llm-as-a-judge.md) | Frontier model grades a **sample** (cite Zheng et al., MT-Bench) |
| Human evaluation | [human-evaluation.md](human-evaluation.md) | Small expert-labeled slice; calibrate judge + report agreement |

Suggested order: programmatic filter/score → LLM judge on a sample of survivors → human audit on a stratified subsample.

## Run Mode 1

```bash
.venv/bin/python scripts/validate_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/validate_instruction_data.py --all
```

See [`dataset-validation/README.md`](../../dataset-validation/README.md).
