# Proposal: iterative LLM as a retrieved-menu LCPN for full-depth IPC

Single-label classification of office `primary_ipc` on Australian patents (IP Australia clean dump). Taxonomy is WIPO IPC 2026 (`data/ipc-codes/ipc_codes_20260101.jsonl`): section → class → subclass → main group → n-dot subgroup (`kind` `s|c|u|m|1–9`). Gold depth is **variable**. Gold is **often a non-leaf** (office stopped although children exist). Primaries in this corpus sit at main group or deeper, never at section/class/subclass alone.

**Not** Jiang et al. 2025 RHC (one generation of a fixed Section→Class→Subclass path, no child menu, halt always at subclass). **Not** a flat retrieve-then-rank of K subclasses. **Not** classical LCPN with one separate classifier per parent node.

## Decision problem (one hop)

A shared instruction-tuned LM implements **local classifier per parent node (LCPN)** (Silla & Freitas 2011) with **non-mandatory leaf-node prediction (NMLNP)**.

Each inference call is a **fresh context** (no trajectory in the prompt):

| Field | Content |
|---|---|
| Document | Claims (optional abstract). Same text every hop. |
| `now` | Current IPC place: code + official title. Root call: synthetic root whose children are sections A–H. |
| Menu | `children_of(now)` from the 2026 scheme tree (parent/child recovered from scheme order + `kind`, not from symbol spelling). Each child: code + title. |
| Action space | Exactly one child on the menu, **or** `STOP`. |

Output:

```text
<short claim-grounded justification>
Classification: <child_code | STOP>
```

If the action is a child, the next call sets `now` to that child and replaces the menu with its children. Halt on `STOP` or when `now` is a scheme leaf. Predicted label = last non-`STOP` place.

**Children are in the student input at train and at test.** Omitting them at test is a different method (open-ended generation / path memorisation).

### Menu construction (context is small; `num_ctx` is not increased)

Median parent has ~2 children; P99 ~15; max 144 (`B29K`). If `|children| ≤ 24`, list all in scheme order. If larger: **pin** the gold child when training; include nearest scheme-order neighbours of `now`/gold; optionally a few title-overlap negatives; never “first 24 only” and never a random 24 as the sole policy. Truncation is a prompt budget, not a taxonomy rule.

Decoding is **constrained** to the menu ∪ `{STOP}` (illegal edges = 0 by construction).

## Training

**Teacher (data construction only).** Sees claims + `now` + menu + **gold action for this hop**. Reconstructs the justification. Must not propose a different child. Gold is not copied into the student `input`.

**Student SFT.** Same `now` + menu + claims; target is justification + `Classification: <gold_action>`. One office path yields one row per hop, including a final `STOP` row when gold is a non-leaf (e.g. gold `A61P35/00` with children `A61P35/02`, `A61P35/04` → last action `STOP`).

Train on **gold prefixes** (`now` is the true ancestor). Test `now` is the model’s previous action (exposure bias).

Optional mix (~10–20%): catalog-only rows `children of X?` covering every parent in the JSONL, splitting huge nodes across rows, so rare places appear even if no patent gold visits them. These are auxiliary; the classification task remains hop LCPN.

Ablation required: hop label-only (`Classification:` only) vs hop label + rationale. Jiang-style evidence: CoT SFT without a label/RL objective can hurt accuracy; do not assume rationales help.

## Inference

```text
now ← ROOT
path ← []
loop:
  y ← LM(claims, now, children_of(now))   # new context
  if y = STOP or children_of(now) = ∅: return path
  now ← y; path.append(y)
```

4–8 calls per patent typical. Batch across patents, not along a path.

## Metrics (must report all)

- Exact office `primary_ipc` (canonicalise office `C07K16/30` ↔ catalog `C07K16/3`).
- Prefix accuracy at section, class, subclass, main group, and last symbol.
- **STOP accuracy** on golds that are non-leaves vs true leaves.
- Hop accuracy given gold `now` (teacher-forced) vs free-run path accuracy.
- Head vs tail symbols; illegal-edge rate (should be ~0 with constraints).
- Temporal holdout as in this repo (`filedDate` cutoff), application-level split.

## What to search for (has this been done?)

Match **all** of the following, not a subset:

1. **LCPN / “local classifier per parent node” / top-down child selection** on a tree.
2. **NMLNP / stop at internal node / variable-depth gold** (not mandatory leaf, not fixed L = subclass).
3. **One shared generative LM** (not N independent SVMs/BERT heads, one per parent).
4. **Child list retrieved into the prompt** (menu / candidates / constrained decode to children of `now`).
5. Applied to **IPC or CPC at group/subgroup depth**, not only 4-character subclass.
6. Optional: **natural-language justification per hop**.

Near-misses that are **not** this method:

- Jiang et al., arXiv:2510.07167 — one-shot Section→Class→Subclass CoT, no menu, no STOP, subclass only.
- Flat RAG of top-K subclass definitions, then constrain the LM to that K-set.
- Retrieve-then-rank over the whole label set (e.g. bi-encoder + cross-encoder).
- Classical LCPN (separate model per parent) on IPC.
- Hierarchical softmax / global HTC (Cai & Hofmann, HiAGM, etc.) with no per-hop menu in context.
- This repo’s current `ipc_reasoning`: one-shot full symbol; catalog in the **teacher** prompt only; student `input` is claims/abstract without a child menu.
