# 02 — Dataset validation

Validate seed instruction-tuning JSONL from [`01-instruction-data-generation`](../01-instruction-data-generation/) before using it for SFT.

| Mode | Doc | Scope |
|------|-----|--------|
| Programmatic | [programmatic-validation.md](programmatic-validation.md) | Schema/IPC (WIPO-grounded cosine/ROUGE) + lexical (ROUGE-L / MRC best-span F1) + semantic (Nomic Embed); abstract drafting is gold-to-gold pairing; see `dataset-validation/` |
| LLM-as-a-judge | [llm-as-a-judge.md](llm-as-a-judge.md) | Frontier model grades a **sample** (pointwise; same Accept definition as Mode 3) |
| Human evaluation | [human-evaluation.md](human-evaluation.md) | Small expert-labeled slice; calibrate `pass_score_min` + report agreement |

Suggested order: programmatic filter/score → LLM judge on a sample of survivors → human audit on a stratified subsample.

## Run Mode 1

```bash
.venv/bin/python scripts/validate_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/validate_instruction_data.py --all
```

See [`dataset-validation/README.md`](../../dataset-validation/README.md).
