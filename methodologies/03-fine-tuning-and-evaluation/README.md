# 03 — Fine-tuning and evaluation

Fine-tune one or more open-source models on validated instruction data from [`02-dataset-validation`](../02-dataset-validation/), then measure them against known baselines and each other.

| Doc | Scope |
|-----|--------|
| [fine-tuning.md](fine-tuning.md) | Supervised fine-tuning (SFT) setup per task and multi-task options |
| [baselining.md](baselining.md) | Zero-/few-shot and published patent/general LLM baselines |
| [comparison.md](comparison.md) | Held-out eval protocol and metrics used to compare models |

Suggested order: freeze train/val/test splits from Mode 1 (+ optional Mode 2/3) survivors → fine-tune open-source students → run the same eval suite on students and baselines → report side-by-side metrics.

Runnable **baseline eval** (no training) lives in [`evaluation/`](../../evaluation/):

```bash
.venv/bin/python scripts/split_eval_data.py --generator meta-llama/llama-3.3-70b-instruct
.venv/bin/python scripts/run_baselines.py --all --generator meta-llama/llama-3.3-70b-instruct
.venv/bin/python scripts/score_baselines.py --generator meta-llama/llama-3.3-70b-instruct
```

SFT / LoRA, PatentBERT, local HF generate, and LLM-as-a-judge on predictions are deferred; they should reuse the same frozen test IDs. See [`evaluation/README.md`](../../evaluation/README.md).
