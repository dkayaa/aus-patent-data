# Faithfulness audit — ipc_reasoning

**Current design** (decide-first META + three-way bands). Stats below are
**reclassified** from the last MiniCheck run (100 records / 306 atoms) without
re-invoking the model. Re-run for a fresh sidecar:

```bash
.venv/bin/python scripts/run_faithfulness_ipc.py --limit 100
```

## Design

1. Atomicise justification  
2. Drop **META** *before* MiniCheck (alignment markers or zero claim terms)  
3. Score remaining atoms on **combined** claims+definition  
4. Bands: **SUPPORTED** P≥0.7 · **UNDECIDED** [0.3, 0.7) · **UNSUPPORTED** P<0.3  

Non-gating. Wrong-bridge (true facts, wrong classifying feature) is out of
scope — expert audit. Undecided is the natural expert-review band.

## Atom-level (reclassified)

| state | n | of all atoms | of scored |
|-------|---|--------------|-----------|
| META (skipped) | 174 | 56.9% | — |
| SUPPORTED | 62 | 20.3% | 47.0% |
| UNDECIDED | 15 | 4.9% | 11.4% |
| UNSUPPORTED | 55 | 18.0% | 41.7% |

- META reasons: alignment 164, empty 10  
- Mean faithfulness among scored ≈ **0.47**  
- Trap check: **69/174** META atoms had P≥0.5 in the old run — decide-first
  avoids counting those as supported  

## Related artifacts

| Path | Role |
|------|------|
| `reports/faithfulness_calibration.md` | Human good/bad pool MiniCheck floor test |
| `dataset-validation/config/faithfulness_calibration.jsonl` | Calibration pool |
| `scripts/run_faithfulness_ipc.py` | Full audit runner |
| `scripts/calibrate_faithfulness_ipc.py` | Pool calibrator |

## Findings (kept short)

- Majority of generated justification atoms are **connective/META**, not
  checkable substance.  
- Among checkable content ≈ half supported / half unsupported (+ ~11%
  undecided).  
- Claims-text mangling correlates only mildly with unsupported rate
  (Spearman ~0.26); dirt helps at the extreme but does not explain the
  headline.  
