# Build the chat messages for one step (same layout as ipc_lcpn training, plus STOP).

from __future__ import annotations

from ipc.accessor import IPCNode

# Kept close to instruction-generation/src/tasks/ipc_lcpn.py wording.
INSTRUCTION_LABEL = (
    "Select the IPC symbol that correctly continues the classification "
    "from the current place, or STOP if none of the children fit. "
    "Answer with the symbol or STOP only."
)

INSTRUCTION_JUSTIFY = (
    "Select the IPC symbol that correctly continues the classification "
    "from the current place, or STOP if none of the children fit. "
    "Write a short justification grounded in the patent text, then a line "
    "of the form Classification: <symbol_or_STOP>."
)

SYSTEM_PROMPT = (
    "You are an expert Australian Patent Examiner. "
    "Classify one IPC step at a time."
)

# Numbered child list, with STOP as the last option.
def choice_lines(children: list[IPCNode], *, stop_token: str) -> str:
    lines = [
        f"{index}. {child.node_id} - {child.title}"
        for index, child in enumerate(children, start=1)
    ]
    lines.append(
        f"{len(children) + 1}. {stop_token} - none of these children; "
        "remain at the current place"
    )
    return "\n".join(lines)

# Join ancestor titles with ' > '.
def title_path(labels: list[str]) -> str:
    parts = [part.strip() or "?" for part in labels]
    return " > ".join(parts)


# Patent text block for the prompt (abstract and/or claim 1).
def build_document(
    *,
    abstract: str,
    claim_1: str,
    include_abstract: bool,
    claims: str,
) -> str:
    blocks: list[str] = []
    if include_abstract and (abstract or "").strip():
        blocks.append(f"Abstract:\n{abstract.strip()}")
    if claims == "claim_1" and (claim_1 or "").strip():
        blocks.append(f"Claim 1:\n{claim_1.strip()}")
    return "\n\n".join(blocks)

# One system message + one user message for a single tree step.
def build_hop_messages(
    *,
    document: str,
    current_id: str,
    titles: str,
    children: list[IPCNode],
    stop_token: str,
    justify: bool,
) -> list[dict[str, str]]:
    instruction = INSTRUCTION_JUSTIFY if justify else INSTRUCTION_LABEL
    blocks: list[str] = [instruction, ""]
    if document:
        blocks.append(document)
        blocks.append("")
    blocks.append(f"Current node: {current_id}")
    blocks.append(f"Titles: {titles}")
    blocks.append("")
    blocks.append(f"Children:\n{choice_lines(children, stop_token=stop_token)}")
    user = "\n".join(blocks).strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
