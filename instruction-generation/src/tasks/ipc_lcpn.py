"""Hop-wise IPC multiple choice along the office primary-IPC path.

The first on-path record is at ``<ROOT>``: the menu is the sections, and
the gold choice is the section on the office path. Each later edge from an
ancestor to the next place is one record. The gold child is the next place
on that path. When the primary place itself has children, one further
record is written there: the menu is those children, and the gold choice is
the stop token. A primary place with no children is a leaf; the walk ends
without asking, so no stop record is written.

When ``back_token`` is set, every on-path menu except ``<ROOT>`` also offers
that token, and one extra record is appended per on-path step: the current
place is one shuffled non-gold child, and the gold choice is ``back_token``.
The shuffle is seeded by the application number and the office step, so a
repeat of the same filing picks the same child. Those rows follow the
on-path rows and do not change on-path hop indexes.

When ``justify`` is true (the default), the teacher writes a short
justification for that gold child. The student input does not contain the
gold symbol. The stored output is the justification, then
``Classification: <symbol>``. When ``justify`` is false, the output is the
symbol only and the teacher is not called.

``options`` (YAML ``tasks.ipc_lcpn``):

- ``include_abstract`` (default true): omit the abstract when false
- ``claims``: ``claim_1`` (default) or ``omit``
- ``justify`` (default true): teacher justification before the symbol
- ``stop_token`` (default ``<STOP>``): last menu option on a path edge's
  gold is the next place. Gold at the primary place, when it has children,
  is this token.
- ``back_token`` (default empty): when set, offer this token to return to
  the parent, and write the off-path gold-back rows. Empty keeps those off.
"""

from __future__ import annotations

import random
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

ROOT_LABEL = "<ROOT>"


def _instruction(stop_token: str, *, justify: bool, back_token: str = "") -> str:
    if back_token:
        choice = (
            "Select the IPC symbol that correctly continues the classification "
            "from the current place, "
            f"{stop_token} if none of the children fit, or "
            f"{back_token} to return to the parent. "
        )
        answer = f"symbol, {stop_token}, or {back_token}"
    else:
        choice = (
            "Select the IPC symbol that correctly continues the classification "
            "from the current place, or "
            f"{stop_token} if none of the children fit. "
        )
        answer = f"symbol or {stop_token}"
    if justify:
        return (
            choice
            + "Write a short justification grounded in the patent text, then a line "
            + f"of the form Classification: <IPC {answer}>."
        )
    return choice + f"Answer with the {answer} only."


TEACHER_SYSTEM = "You are an expert Australian Patent Examiner."

TEACHER_TRAILER = """Write a short justification (about 40–90 words) that maps concrete features of this patent onto the correct symbol's title.

Use only the supplied patent text and that title. Do not mention the other choices or any other IPC symbol. Do not write a Classification line. Respond with the justification only."""

TEACHER_TRAILER_STOP = """Write a short justification (about 40–90 words) that explains why this patent is covered by the current place and should remain there rather than move to a more specific child.

Use only the supplied patent text and the current place's title. Do not mention the other choices or any other IPC symbol. Do not write a Classification line. Respond with the justification only."""

TEACHER_TRAILER_BACK = """Write a short justification (about 40–90 words) that explains why this patent is not covered by the current place and should return to the parent.

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


def _choice_lines(
    children: list[IPCNode],
    *,
    stop_token: str,
    back_token: str = "",
) -> str:
    lines = [
        f"{index}. {child.node_id} - {child.title}"
        for index, child in enumerate(children, start=1)
    ]
    lines.append(
        f"{len(children) + 1}. {stop_token} - none of these children; "
        "remain at the current place"
    )
    if back_token:
        lines.append(f"{len(children) + 2}. {back_token} - return to the parent")
    return "\n".join(lines)


def _wrong_child(
    children: list[IPCNode],
    gold_id: str,
    *,
    is_stop: bool,
    seed: str,
) -> IPCNode | None:
    """One non-gold child, chosen by a seeded shuffle.

    A stop hop has no gold child, so every child is eligible.
    """
    pool = list(children) if is_stop else [child for child in children if child.node_id != gold_id]
    if not pool:
        return None
    random.Random(seed).shuffle(pool)
    return pool[0]


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
        back_token = self._back_token()
        steps = self._on_path_steps(path, stop_token=stop_token, back_token=back_token)
        if steps is None:
            return None
        if back_token:
            steps.extend(
                self._back_steps(
                    steps,
                    back_token=back_token,
                    seed=patent.application_number,
                )
            )

        document = self._document(patent)
        claims_mode = self._claims_mode()
        include_abstract = self._include_abstract()
        justify = self._justify()
        records: list[dict[str, Any]] = []
        n_hops = len(steps)
        path_node_ids = [ROOT_LABEL, *[node.node_id for node in path]]
        for index, step in enumerate(steps):
            menu_back = back_token if step["offer_back"] else ""
            titles = step["titles"]
            choice_lines = _choice_lines(
                step["children"],
                stop_token=stop_token,
                back_token=menu_back,
            )
            blocks: list[str] = []
            if document:
                blocks.append(document)
            blocks.append(f"Current node: {step['now_id']}\nTitles: {titles}")
            blocks.append(f"Children:\n{choice_lines}")
            input_text = "\n\n".join(blocks)
            fitted = None
            instruction = _instruction(stop_token, justify=justify, back_token=menu_back)
            if justify:
                justification, fitted = self._justify_hop(
                    patent,
                    now_id=step["now_id"],
                    now_title=step["now_title"],
                    gold_id=step["gold_id"],
                    gold_title=step["gold_title"],
                    titles=titles,
                    choice_lines=choice_lines,
                    stop_token=stop_token,
                    back_token=back_token,
                    is_stop=step["is_stop"],
                    is_back=step["is_back"],
                    parent_id=step["parent_id"],
                    parent_title=step["parent_title"],
                )
                output_text = f"{justification}\nClassification: {step['gold_id']}"
            else:
                output_text = step["gold_id"]
            extra: dict[str, Any] = {}
            if fitted is not None:
                extra.update(self._fit_meta(fitted))
            records.append(
                self._record(
                    patent,
                    instruction=instruction,
                    input_text=input_text,
                    output_text=output_text,
                    now_node_id=step["now_id"],
                    gold_node_id=step["gold_id"],
                    is_stop=step["is_stop"],
                    is_back=step["is_back"],
                    parent_node_id=step["parent_id"],
                    hop_index=index,
                    n_hops=n_hops,
                    depth=len(path),
                    path_node_ids=path_node_ids,
                    include_abstract=include_abstract,
                    claims=claims_mode,
                    justify=justify,
                    stop_token=stop_token,
                    back_token=back_token,
                    filed_date=patent.filed_date or None,
                    **extra,
                )
            )
        return records

    def _on_path_steps(
        self,
        path: list[IPCNode],
        *,
        stop_token: str,
        back_token: str,
    ) -> list[dict[str, Any]] | None:
        assert self._accessor is not None
        root_children = self._accessor.get_children("")
        if not any(child.node_id == path[0].node_id for child in root_children):
            return None
        steps: list[dict[str, Any]] = [
            {
                "now_id": ROOT_LABEL,
                "now_title": ROOT_LABEL,
                "gold_id": path[0].node_id,
                "gold_title": path[0].title,
                "children": root_children,
                "is_stop": False,
                "is_back": False,
                "parent_id": "",
                "parent_title": "",
                "offer_back": False,
                "titles": ROOT_LABEL,
            }
        ]
        for index in range(len(path) - 1):
            now = path[index]
            gold = path[index + 1]
            children = self._accessor.get_children(now.node_id)
            if not any(child.node_id == gold.node_id for child in children):
                return None
            parent = path[index - 1] if index else None
            steps.append(
                {
                    "now_id": now.node_id,
                    "now_title": now.title,
                    "gold_id": gold.node_id,
                    "gold_title": gold.title,
                    "children": children,
                    "is_stop": False,
                    "is_back": False,
                    "parent_id": ROOT_LABEL if parent is None else parent.node_id,
                    "parent_title": ROOT_LABEL if parent is None else parent.title,
                    "offer_back": bool(back_token),
                    "titles": _title_path(path[: index + 1]),
                }
            )
        final = path[-1]
        final_children = self._accessor.get_children(final.node_id)
        if final_children:
            parent = path[-2]
            steps.append(
                {
                    "now_id": final.node_id,
                    "now_title": final.title,
                    "gold_id": stop_token,
                    "gold_title": "",
                    "children": final_children,
                    "is_stop": True,
                    "is_back": False,
                    "parent_id": parent.node_id,
                    "parent_title": parent.title,
                    "offer_back": bool(back_token),
                    "titles": _title_path(path),
                }
            )
        return steps

    def _back_steps(
        self,
        on_path: list[dict[str, Any]],
        *,
        back_token: str,
        seed: str,
    ) -> list[dict[str, Any]]:
        assert self._accessor is not None
        extra: list[dict[str, Any]] = []
        for step in on_path:
            wrong = _wrong_child(
                step["children"],
                step["gold_id"],
                is_stop=step["is_stop"],
                seed=f"{seed}|{step['now_id']}|{step['gold_id']}",
            )
            if wrong is None:
                continue
            title = wrong.title.strip() or wrong.node_id
            extra.append(
                {
                    "now_id": wrong.node_id,
                    "now_title": wrong.title,
                    "gold_id": back_token,
                    "gold_title": "",
                    "children": self._accessor.get_children(wrong.node_id),
                    "is_stop": False,
                    "is_back": True,
                    "parent_id": step["now_id"],
                    "parent_title": step["now_title"],
                    "offer_back": True,
                    "titles": f"{step['titles']} > {title}",
                }
            )
        return extra

    def _justify_hop(
        self,
        patent: PatentText,
        *,
        now_id: str,
        now_title: str,
        gold_id: str,
        gold_title: str,
        titles: str,
        choice_lines: str,
        stop_token: str,
        back_token: str,
        is_stop: bool,
        is_back: bool,
        parent_id: str,
        parent_title: str,
    ) -> tuple[str, Any]:
        if is_back:
            header = [
                f"The correct choice is {back_token}.",
                f"Return to the parent place {parent_id} ({parent_title}).",
                f"Do not choose a child symbol or {stop_token}.",
            ]
            trailer = TEACHER_TRAILER_BACK
        elif is_stop:
            header = [
                f"The correct choice is {stop_token}.",
                f"Remain at the current place {now_id} ({now_title}).",
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
            f"Current node: {now_id}",
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

    def _back_token(self) -> str:
        raw = self.options.get("back_token")
        if raw is None:
            return ""
        return str(raw).strip()

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
