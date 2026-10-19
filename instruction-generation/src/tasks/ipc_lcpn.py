"""Hop-wise IPC multiple choice along the office primary-IPC path.

A primary symbol N places deep (section = 1) yields one record per edge
from each ancestor to the next place on the path. The gold child is the
next place on that path. When the primary place itself has children, one
further record is written there: the menu is those children, and the gold
choice is the stop token. A primary place with no children is a leaf; the
walk ends without asking, so no stop record is written.

When ``justify`` is true (the default), the teacher writes a short
justification for that gold child. The student input does not contain the
gold symbol. The stored output is the justification, then
``Classification: <symbol>``. When ``justify`` is false, the output is the
symbol only and the teacher is not called.

``options`` (YAML ``tasks.ipc_lcpn``):

- ``include_abstract`` (default true): omit the abstract when false
- ``claims``: ``claim_1`` (default) or ``omit``
- ``justify`` (default true): teacher justification before the symbol
- ``stop_token`` (default ``<STOP>``): last menu option. Gold on a path
  edge is the next place. Gold at the primary place, when it has children,
  is this token.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from patents import PatentText
from prompt_fit import (
    OversizedPromptError,
    append_oversized_record,
    fit_teacher_prompt,
    prompt_budget_from_config,
)

from .base import Task

_IPC_ROOT = Path(__file__).resolve().parents[3]
if str(_IPC_ROOT) not in sys.path:
    sys.path.insert(0, str(_IPC_ROOT))

from ipc.accessor import IPCAccessor, IPCNode  # noqa: E402

def _instruction_label(stop_token: str) -> str:
    return (
        "Select the IPC symbol that correctly continues the classification "
        "from the current place, or "
        f"{stop_token} if none of the children fit. "
        f"Answer with the symbol or {stop_token} only."
    )


def _instruction_justify(stop_token: str) -> str:
    return (
        "Select the IPC symbol that correctly continues the classification "
        "from the current place, or "
        f"{stop_token} if none of the children fit. "
        "Write a short justification grounded in the patent text, then a line "
        f"of the form Classification: <IPC symbol or {stop_token}>."
    )


TEACHER_SYSTEM = "You are an expert Australian Patent Examiner."

TEACHER_TRAILER = """Write a short justification (about 40–90 words) that maps concrete features of this patent onto the correct symbol's title.

Use only the supplied patent text and that title. Do not mention the other choices or any other IPC symbol. Do not write a Classification line. Respond with the justification only."""

TEACHER_TRAILER_STOP = """Write a short justification (about 40–90 words) that explains why this patent is covered by the current place and should remain there rather than move to a more specific child.

Use only the supplied patent text and the current place's title. Do not mention the other choices or any other IPC symbol. Do not write a Classification line. Respond with the justification only."""

_CLAIMS_CLAIM_1 = "claim_1"
_CLAIMS_OMIT = "omit"
_CLAIMS_MODES = frozenset({_CLAIMS_CLAIM_1, _CLAIMS_OMIT})

# Office symbols sometimes drop a trailing subgroup zero (C07K16/3 for C07K16/30).
_HUMAN_GROUP = re.compile(r"^([A-H]\d{2}[A-Z])(\d{1,4})/(\d+)$")


def _pad_short_subgroup(code: str) -> str | None:
    match = _HUMAN_GROUP.match(code.strip().upper().replace(" ", ""))
    if match is None:
        return None
    tail = match.group(3)
    if tail == "00" or len(tail) >= 2:
        return None
    return f"{match.group(1)}{match.group(2)}/{tail.ljust(2, '0')}"


def path_to(accessor: IPCAccessor, node_id: str) -> list[IPCNode] | None:
    """Section-to-node path, or None when the symbol is missing or is a section."""
    node = _lookup(accessor, node_id)
    if node is None:
        return None
    chain = [node]
    seen = {node.node_id}
    while True:
        parents = accessor.get_parent(chain[-1].node_id)
        if not parents:
            break
        parent = parents[0]
        if parent.node_id in seen:
            return None
        seen.add(parent.node_id)
        chain.append(parent)
    chain.reverse()
    if len(chain) < 2:
        return None
    return chain


def _lookup(accessor: IPCAccessor, node_id: str) -> IPCNode | None:
    candidates = [node_id]
    padded = _pad_short_subgroup(node_id)
    if padded is not None and padded not in candidates:
        candidates.append(padded)
    for candidate in candidates:
        try:
            return accessor.get(candidate)
        except KeyError:
            continue
    return None


def _title_path(nodes: list[IPCNode]) -> str:
    parts = [node.title.strip() or node.node_id for node in nodes]
    return " > ".join(parts)


def _choice_lines(children: list[IPCNode], *, stop_token: str) -> str:
    lines = [
        f"{index}. {child.node_id} - {child.title}"
        for index, child in enumerate(children, start=1)
    ]
    lines.append(
        f"{len(children) + 1}. {stop_token} - none of these children; "
        "remain at the current place"
    )
    return "\n".join(lines)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _justification_body(text: str) -> str:
    lines = [
        line
        for line in text.strip().splitlines()
        if not line.strip().lower().startswith("classification:")
    ]
    body = "\n".join(lines).strip()
    if not body:
        raise RuntimeError("teacher returned an empty justification")
    return body


class IPCLCPNTask(Task):
    task_id = "ipc_lcpn"
    needs_llm = False

    @classmethod
    def requires_llm(cls, options: dict[str, Any] | None) -> bool:
        return _as_bool((options or {}).get("justify"), True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._accessor: IPCAccessor | None = None

    def setup(self) -> None:
        self._claims_mode()
        if self._accessor is None:
            self._accessor = IPCAccessor()
            self._accessor.init()

    def eligible(self, patent: PatentText) -> bool:
        if not patent.primary_ipc:
            return False
        self._ensure_accessor()
        assert self._accessor is not None
        return path_to(self._accessor, patent.primary_ipc) is not None

    def generate(self, patent: PatentText) -> list[dict[str, Any]] | None:
        if not patent.primary_ipc:
            return None
        self._ensure_accessor()
        assert self._accessor is not None
        path = path_to(self._accessor, patent.primary_ipc)
        if path is None:
            return None

        stop_token = self._stop_token()
        hops: list[tuple[IPCNode, str, list[IPCNode], bool]] = []
        for index in range(len(path) - 1):
            now = path[index]
            gold = path[index + 1]
            children = self._accessor.get_children(now.node_id)
            if not any(child.node_id == gold.node_id for child in children):
                return None
            hops.append((now, gold.node_id, children, False))
        final = path[-1]
        final_children = self._accessor.get_children(final.node_id)
        if final_children:
            hops.append((final, stop_token, final_children, True))

        document = self._document(patent)
        claims_mode = self._claims_mode()
        include_abstract = self._include_abstract()
        justify = self._justify()
        records: list[dict[str, Any]] = []
        n_hops = len(hops)
        for index, (now, gold_id, children, is_stop) in enumerate(hops):
            ancestors = path if is_stop else path[: index + 1]
            titles = _title_path(ancestors)
            choice_lines = _choice_lines(children, stop_token=stop_token)
            blocks: list[str] = []
            if document:
                blocks.append(document)
            blocks.append(f"Current node: {now.node_id}\nTitles: {titles}")
            blocks.append(f"Children:\n{choice_lines}")
            input_text = "\n\n".join(blocks)
            fitted = None
            if justify:
                justification, fitted = self._justify_hop(
                    patent,
                    now=now,
                    gold_id=gold_id,
                    gold_title="" if is_stop else self._child_title(children, gold_id),
                    titles=titles,
                    choice_lines=choice_lines,
                    stop_token=stop_token,
                    is_stop=is_stop,
                )
                output_text = f"{justification}\nClassification: {gold_id}"
                instruction = _instruction_justify(stop_token)
            else:
                output_text = gold_id
                instruction = _instruction_label(stop_token)
            extra: dict[str, Any] = {}
            if fitted is not None:
                extra.update(self._fit_meta(fitted))
            records.append(
                self._record(
                    patent,
                    instruction=instruction,
                    input_text=input_text,
                    output_text=output_text,
                    now_node_id=now.node_id,
                    gold_node_id=gold_id,
                    is_stop=is_stop,
                    hop_index=index,
                    n_hops=n_hops,
                    depth=len(path),
                    path_node_ids=[node.node_id for node in path],
                    include_abstract=include_abstract,
                    claims=claims_mode,
                    justify=justify,
                    stop_token=stop_token,
                    filed_date=patent.filed_date or None,
                    **extra,
                )
            )
        return records

    def _child_title(self, children: list[IPCNode], gold_id: str) -> str:
        for child in children:
            if child.node_id == gold_id:
                return child.title
        return ""

    def _justify_hop(
        self,
        patent: PatentText,
        *,
        now: IPCNode,
        gold_id: str,
        gold_title: str,
        titles: str,
        choice_lines: str,
        stop_token: str,
        is_stop: bool,
    ) -> tuple[str, Any]:
        if is_stop:
            header = [
                f"The correct choice is {stop_token}.",
                f"Remain at the current place {now.node_id} ({now.title}).",
                "Do not choose a child symbol.",
            ]
            trailer = TEACHER_TRAILER_STOP
        else:
            header = [
                f"The correct next IPC symbol is {gold_id} ({gold_title}).",
                "Do not choose a different symbol.",
            ]
            trailer = TEACHER_TRAILER
        body = [
            f"Current node: {now.node_id}",
            f"Titles: {titles}",
            "",
            f"Children:\n{choice_lines}",
        ]
        if self._include_abstract():
            body[0:0] = [f"Abstract:\n{patent.abstract}", ""]
        instruction = "\n".join(header + ["", *body])
        fitted = None
        if self._claims_mode() == _CLAIMS_CLAIM_1:
            budget = self.prompt_budget or prompt_budget_from_config(
                {
                    "num_ctx": self.client.config.num_ctx,
                    "max_output_tokens": self.client.config.max_output_tokens,
                    "safety_margin": self.client.config.safety_margin,
                    "repeat_instruction": self.client.config.repeat_instruction,
                    "tokenizer_id": self.client.config.tokenizer_id,
                }
            )
            try:
                fitted = fit_teacher_prompt(
                    system=TEACHER_SYSTEM,
                    instruction=instruction,
                    claims=[patent.claim_1],
                    trailer=trailer,
                    budget=budget,
                    application_number=patent.application_number,
                    task=self.task_id,
                )
            except OversizedPromptError as exc:
                append_oversized_record(
                    application_number=exc.application_number,
                    task=exc.task,
                    prompt_tokens=exc.prompt_tokens,
                    input_budget=exc.input_budget,
                    n_claims_original=exc.n_claims_original,
                )
                raise
            messages = fitted.messages
        else:
            messages = [
                {"role": "system", "content": TEACHER_SYSTEM},
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{trailer}",
                },
            ]
        return _justification_body(self.client.chat(messages)), fitted

    def _ensure_accessor(self) -> None:
        if self._accessor is None:
            self.setup()

    def _include_abstract(self) -> bool:
        return _as_bool(self.options.get("include_abstract"), True)

    def _justify(self) -> bool:
        return _as_bool(self.options.get("justify"), True)

    def _stop_token(self) -> str:
        token = str(self.options.get("stop_token") or "<STOP>").strip()
        if not token:
            raise ValueError("tasks.ipc_lcpn.stop_token must be a non-empty string")
        return token

    def _claims_mode(self) -> str:
        mode = str(self.options.get("claims", _CLAIMS_CLAIM_1)).strip().lower()
        if mode not in _CLAIMS_MODES:
            raise ValueError(
                f"tasks.ipc_lcpn.claims must be one of {sorted(_CLAIMS_MODES)}, got {mode!r}"
            )
        return mode

    def _document(self, patent: PatentText) -> str:
        parts: list[str] = []
        if self._include_abstract():
            parts.append(f"Abstract:\n{patent.abstract}")
        if self._claims_mode() == _CLAIMS_CLAIM_1:
            parts.append(f"Claim 1:\n{patent.claim_1}")
        return "\n\n".join(parts)
