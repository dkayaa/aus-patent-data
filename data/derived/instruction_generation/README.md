# Instruction-generation datasets

IPC LCPN hop JSONL under this directory. Each row is one hop: claim 1, the current IPC place, its children, and the gold choice. `_pools/`, `_samples/`, and `_archive/` are not these runs.

None of these runs includes a `<ROOT>` hop. Hop 0 starts at the section (`A`–`H`) and the gold is the class. The first group (`…/00`) is the gold at hop 2, under the subclass.

## `qwen-qwen3-235b-a22b-2507_claim1_nojustify_pilot_2000_v1`

Symbol-only pilot, version 1. `justify` is false, so each output is the gold symbol and the teacher is not called. 2,000 filings, 10,431 hops, nothing skipped. When the office primary still has children, the last hop's gold is `<STOP>`. Started 2026-10-19.

## `qwen-qwen3-235b-a22b-2507_claim1_justify_pilot_2000_v1`

Same 2,000 filings as the symbol-only pilot, version 1, with teacher justifications. Provider OpenRouter, model `qwen/qwen3-235b-a22b-2507`, temperature 0. Each output is a short reason and then `Classification: <symbol or STOP>`. 99 filings were skipped because claim 1 did not fit the teacher context, leaving 1,901 filings and 9,900 hops. `<STOP>` hops are included on the same rule as the symbol-only pilot. Started 2026-10-20.
