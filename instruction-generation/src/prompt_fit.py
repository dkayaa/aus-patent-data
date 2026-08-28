"""Guarantee teacher prompts fit in num_ctx before they are sent to Ollama.

Ollama drops the BEGINNING of over-limit prompts (see
scripts/probe_ollama_truncation.py). Our prompts are instruction-first,
claims-second, so overflow silently destroyed the instruction. This module
makes that structurally impossible: we trim CLAIMS ONLY (by whole claim),
never the instruction, and refuse to send if claim 1 alone still overflows.

Tokenizer: HuggingFace ``NousResearch/Meta-Llama-3-8B`` — same Llama-3 BPE
vocabulary as ``llama3.1:8b``. Meta's gated ``meta-llama/Meta-Llama-3.1-8B``
tokenizer is equivalent but requires HF auth; NousResearch is the public twin.
Ollama does not expose a tokenize API for chat models, so HF is the documented
stand-in for exact counts.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERSIZED_PATH = REPO_ROOT / "reports" / "oversized_records.jsonl"

# Default HF id for llama3.1:8b (documented above).
DEFAULT_TOKENIZER_ID = "NousResearch/Meta-Llama-3-8B"

_CLAIM_SPLIT_RE = re.compile(r"(?=\n\s*\d+\.\s)")
_CLAIM_NUM_RE = re.compile(r"^\s*(\d+)\.\s")


@dataclass(frozen=True)
class PromptBudget:
    num_ctx: int
    max_output_tokens: int
    safety_margin: int = 64
    repeat_instruction: bool = True
    tokenizer_id: str = DEFAULT_TOKENIZER_ID

    @property
    def input_budget(self) -> int:
        return int(self.num_ctx) - int(self.max_output_tokens) - int(self.safety_margin)


@dataclass
class FittedPrompt:
    messages: list[dict[str, str]]
    claims_text_sent: str
    claims_trimmed: bool
    claims_dropped: list[int]
    n_claims_original: int
    n_claims_sent: int
    prompt_tokens: int
    input_budget: int
    skipped_oversized: bool = False


class OversizedPromptError(RuntimeError):
    """Instruction + claim 1 alone exceeds the input budget — do not send."""

    def __init__(
        self,
        *,
        application_number: str,
        task: str,
        prompt_tokens: int,
        input_budget: int,
        n_claims_original: int,
    ) -> None:
        self.application_number = application_number
        self.task = task
        self.prompt_tokens = prompt_tokens
        self.input_budget = input_budget
        self.n_claims_original = n_claims_original
        super().__init__(
            f"oversized: app={application_number} task={task} "
            f"tokens={prompt_tokens} budget={input_budget} "
            f"(instruction + claim 1 still overflows)"
        )


_tokenizer = None
_tokenizer_id_loaded: str | None = None


def get_tokenizer(tokenizer_id: str = DEFAULT_TOKENIZER_ID):
    """Lazy-load and cache the HuggingFace tokenizer for the target model."""
    global _tokenizer, _tokenizer_id_loaded
    if _tokenizer is not None and _tokenizer_id_loaded == tokenizer_id:
        return _tokenizer
    from transformers import AutoTokenizer

    logger.info("Loading prompt tokenizer: %s", tokenizer_id)
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    _tokenizer_id_loaded = tokenizer_id
    return _tokenizer


def count_tokens(text: str, *, tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> int:
    tok = get_tokenizer(tokenizer_id)
    return len(tok.encode(text, add_special_tokens=False))


def count_messages_tokens(
    messages: Sequence[dict[str, str]],
    *,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
) -> int:
    """Count tokens in a chat message list (contents + small per-message overhead)."""
    parts = []
    for msg in messages:
        parts.append(str(msg.get("role") or ""))
        parts.append(str(msg.get("content") or ""))
    # Llama-3 chat template overhead is a handful of special tokens per message.
    return count_tokens("\n".join(parts), tokenizer_id=tokenizer_id) + 4 * len(messages)


def split_claims(claims: Sequence[str] | str) -> list[tuple[int, str]]:
    """Return [(claim_number, claim_text), ...] preserving order.

    Prefer a list of claim strings (pipeline PatentText.claims). Fall back to
    splitting a joined claims blob on numbered boundaries.
    """
    if isinstance(claims, str):
        text = claims.strip()
        if not text:
            return []
        parts = [p.strip() for p in _CLAIM_SPLIT_RE.split("\n" + text) if p.strip()]
        if len(parts) <= 1 and not _CLAIM_NUM_RE.match(text):
            return [(1, text)]
        out: list[tuple[int, str]] = []
        for i, part in enumerate(parts, start=1):
            m = _CLAIM_NUM_RE.match(part)
            num = int(m.group(1)) if m else i
            out.append((num, part))
        return out

    out = []
    for i, c in enumerate(claims, start=1):
        text = c.strip() if isinstance(c, str) else ""
        if not text:
            continue
        m = _CLAIM_NUM_RE.match(text)
        num = int(m.group(1)) if m else i
        out.append((num, text))
    return out


def join_claim_parts(parts: Sequence[tuple[int, str]]) -> str:
    return "\n\n".join(text for _, text in parts)


def build_user_content(
    *,
    instruction: str,
    claims_text: str,
    trailer: str = "",
    repeat_instruction: bool = True,
) -> str:
    """Assemble user content: [instruction] [claims] [instruction?] [trailer?]."""
    instr = instruction.strip()
    claims = claims_text.strip()
    blocks = [instr, "", "Claims:", claims]
    if repeat_instruction:
        blocks.extend(["", instr])
    if trailer.strip():
        blocks.extend(["", trailer.strip()])
    return "\n".join(blocks)


def fit_teacher_prompt(
    *,
    system: str,
    instruction: str,
    claims: Sequence[str] | str,
    trailer: str = "",
    budget: PromptBudget,
    application_number: str,
    task: str,
) -> FittedPrompt:
    """Assemble messages that fit ``budget.input_budget``, trimming claims from the end.

    Fast path: one token count on the full prompt. Trim loop only if oversized.
    Never drops claim 1. Raises OversizedPromptError if claim 1 alone still overflows.
    """
    claim_parts = split_claims(claims)
    if not claim_parts:
        raise ValueError(f"no claims to fit for {application_number}")

    n_original = len(claim_parts)
    input_budget = budget.input_budget

    def _messages(parts: Sequence[tuple[int, str]]) -> list[dict[str, str]]:
        user = build_user_content(
            instruction=instruction,
            claims_text=join_claim_parts(parts),
            trailer=trailer,
            repeat_instruction=budget.repeat_instruction,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # Fast path: full claims.
    full_msgs = _messages(claim_parts)
    full_tokens = count_messages_tokens(full_msgs, tokenizer_id=budget.tokenizer_id)
    if full_tokens <= input_budget:
        return FittedPrompt(
            messages=full_msgs,
            claims_text_sent=join_claim_parts(claim_parts),
            claims_trimmed=False,
            claims_dropped=[],
            n_claims_original=n_original,
            n_claims_sent=n_original,
            prompt_tokens=full_tokens,
            input_budget=input_budget,
        )

    # Slow path: drop from the END, never claim 1.
    kept = list(claim_parts)
    dropped: list[int] = []
    while len(kept) > 1:
        dropped.append(kept[-1][0])
        kept = kept[:-1]
        msgs = _messages(kept)
        n_tok = count_messages_tokens(msgs, tokenizer_id=budget.tokenizer_id)
        if n_tok <= input_budget:
            logger.warning(
                "claims trimmed to fit budget: app=%s task=%s "
                "dropped=%s tokens=%d budget=%d",
                application_number,
                task,
                dropped,
                n_tok,
                input_budget,
            )
            return FittedPrompt(
                messages=msgs,
                claims_text_sent=join_claim_parts(kept),
                claims_trimmed=True,
                claims_dropped=dropped,
                n_claims_original=n_original,
                n_claims_sent=len(kept),
                prompt_tokens=n_tok,
                input_budget=input_budget,
            )

    # Claim 1 alone still too big.
    msgs = _messages(kept)
    n_tok = count_messages_tokens(msgs, tokenizer_id=budget.tokenizer_id)
    raise OversizedPromptError(
        application_number=application_number,
        task=task,
        prompt_tokens=n_tok,
        input_budget=input_budget,
        n_claims_original=n_original,
    )


def append_oversized_record(
    *,
    application_number: str,
    task: str,
    prompt_tokens: int,
    input_budget: int,
    n_claims_original: int,
    path: Path | None = None,
) -> None:
    out = path or OVERSIZED_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "application_number": application_number,
        "task": task,
        "prompt_tokens": prompt_tokens,
        "input_budget": input_budget,
        "n_claims_original": n_claims_original,
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.error(
        "SKIP oversized record (instruction+claim1 > budget): %s",
        row,
    )


def prompt_budget_from_config(llm_cfg: dict[str, Any], prompt_cfg: dict[str, Any] | None = None) -> PromptBudget:
    prompt_cfg = prompt_cfg or {}
    max_out = int(
        prompt_cfg.get("max_output_tokens")
        or llm_cfg.get("max_output_tokens")
        or llm_cfg.get("max_tokens")
        or 1024
    )
    return PromptBudget(
        num_ctx=int(llm_cfg.get("num_ctx") or 8192),
        max_output_tokens=max_out,
        safety_margin=int(prompt_cfg.get("safety_margin") or llm_cfg.get("safety_margin") or 64),
        repeat_instruction=bool(
            prompt_cfg.get(
                "repeat_instruction",
                llm_cfg.get("repeat_instruction", True),
            )
        ),
        tokenizer_id=str(
            prompt_cfg.get("tokenizer_id")
            or llm_cfg.get("tokenizer_id")
            or DEFAULT_TOKENIZER_ID
        ),
    )


def assert_ollama_num_ctx(
    *,
    model: str,
    base_url: str,
    expected_num_ctx: int,
    chat_fn: Callable[[], None],
) -> int:
    """Warm the model with configured num_ctx, then assert /api/ps reports the same.

    Fails loudly on mismatch — never warn-and-continue.
    """
    # base_url is typically http://127.0.0.1:11434/v1 → origin without /v1
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]

    chat_fn()  # load model with options.num_ctx from the client

    req = urllib.request.Request(f"{origin}/api/ps", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to query Ollama /api/ps at {origin}: {exc}") from exc

    models = data.get("models") or []
    if not models:
        raise RuntimeError(
            f"Ollama /api/ps returned no running models after warmup; "
            f"cannot assert num_ctx={expected_num_ctx}"
        )

    # Match by name prefix (tags may vary).
    match = None
    for m in models:
        name = str(m.get("name") or m.get("model") or "")
        if name == model or name.startswith(model.split(":")[0]):
            match = m
            break
    if match is None:
        match = models[0]

    # Ollama versions expose context under different keys.
    actual = (
        match.get("context_length")
        or match.get("context")
        or (match.get("details") or {}).get("context_length")
        or match.get("size_vram")  # never use this — placeholder to detect miss
    )
    # Prefer explicit context fields only.
    actual = match.get("context_length")
    if actual is None and isinstance(match.get("details"), dict):
        actual = match["details"].get("context_length")
    if actual is None:
        # Newer Ollama: nested under 'model' info / options
        actual = match.get("context")

    if actual is None:
        raise RuntimeError(
            f"Ollama /api/ps did not report context_length for {model!r}. "
            f"Raw entry keys: {sorted(match.keys())}. "
            f"Refusing to continue without an asserted num_ctx "
            f"(expected {expected_num_ctx}). Full entry: {match!r}"
        )

    actual_i = int(actual)
    if actual_i != int(expected_num_ctx):
        raise RuntimeError(
            f"Ollama effective num_ctx={actual_i} does not match config "
            f"num_ctx={expected_num_ctx} for model={model!r}. Failing hard."
        )
    logger.info(
        "Asserted Ollama num_ctx=%d matches config for model=%s",
        actual_i,
        model,
    )
    return actual_i
