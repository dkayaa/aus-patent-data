# Note: Label-only vs label + rationale SFT

## What we ablate

On `ipc_reasoning`, compare student models fine-tuned to emit:

1. **Classification only** — office `primary_ipc` (gold).
2. **Classification + justification** — same gold code plus a synthetic WIPO-grounded paragraph.

**Primary test metric:** held-out exact (and hierarchical) IPC accuracy against the office label. Same temporal split, same `instruction` / `input`. Justification quality is secondary characterisation, not the discovery endpoint.

**Claim of the contrast:** whether extra supervised rationale tokens act as useful auxiliary supervision or as noise for predicting the office code — **not** whether the model has acquired examiner-grade reasoning.

With the current generation policy, justifications are written by a teacher that already knows `primary_ipc` (post-hoc rationalisation). The ablation is therefore **label-only SFT vs label + post-hoc rationale SFT**, not true pre-decision chain-of-thought.

## Prior art (the ablation template is not new)

### Annotator rationales (classical NLP)

- **Zaidan, Eisner & Piatko (NAACL 2007).** *Using “Annotator Rationales” to Improve Machine Learning for Text Categorization.* Human highlights / free-text “why” as richer supervision than labels alone for text categorization.
- Follow-ons ask **when** explanation data helps (e.g. Carton et al., Findings of ACL 2022, *What to Learn, and How: Toward Effective Learning from Rationales*; Hase & Bansal on learning from explanations). Gains are method- and dataset-dependent (masks, attention regularisers, multi-task losses), not automatic from concatenating rationale text into the target.

### LLM rationale-augmented fine-tuning (RAFT)

- **Findings of ACL 2025:** *Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability.* Directly compares finetuning on \(y\) alone vs on rationale-augmented targets \((r, y)\) / \(y; r\), evaluating **label accuracy** (and calibration). Rationales help on some tasks/models and **hurt** on others.
- Related LLM work distinguishes **Reason** (\(r\) before \(y\)) vs **Explain** (\(y\) before \(r\)). Some results suggest free-text rationales only help when **label learning is protected** (e.g. separate losses / multi-task mixing), rather than when a long narrative dominates the sequence loss.

### Takeaway for this project

| Settled in literature | Not settled / our instance |
|----------------------|----------------------------|
| Label vs label+rationale → score the label is a standard contrast | High-cardinality **IPC**, gold office codes, **synthetic** definition-grounded justifications, patent claims as input, temporal AU holdout |
| Outcomes are mixed (help / hurt / depend on format and loss) | Whether *this* noisy, post-hoc channel moves IPC accuracy |

**Paper positioning:** cite the annotator-rationale and RAFT lines; run the same contrast; claim the **domain and data-construction regime**, not inventing the ablation. Do not frame a win as “models reason like examiners,” or a loss as “rationales never help.”

## Also see (methodology in this repo)

- [`methodologies/03-fine-tuning-and-evaluation/comparison.md`](../methodologies/03-fine-tuning-and-evaluation/comparison.md) — IPC automatic metrics (code accuracy is headline).
- [`methodologies/01-instruction-data-generation/ipc-reasoning-and-classification.md`](../methodologies/01-instruction-data-generation/ipc-reasoning-and-classification.md) — student assign+justify task; teacher still treats office IPC as gold when synthesising prose.
