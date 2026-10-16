# Walk the IPC tree: ask the model at each step, move to the chosen child or stop.

from __future__ import annotations

from dataclasses import dataclass, field

from ipc.accessor import IPCAccessor, IPCNode

from .client import ChatClient
from .config import IPCLCPNConfig
from .parse import HopParseError, ParsedHop, parse_hop_response
from .prompts import build_document, build_hop_messages, title_path

# One step of the walk (for logging / JSON output).
@dataclass(frozen=True)
class HopRecord:

    now_id: str
    titles: str
    child_ids: list[str]
    raw_response: str
    parsed: ParsedHop


@dataclass
class WalkResult:
    path: list[str] = field(default_factory=list)
    hops: list[HopRecord] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def path_str(self) -> str:
        return " | ".join(self.path)


def walk_ipc(
    *,
    accessor: IPCAccessor,
    client: ChatClient,
    cfg: IPCLCPNConfig,
    abstract: str = "",
    claim_1: str = "",
) -> WalkResult:
    """Classify one patent by walking from ROOT down until STOP or a leaf."""
    document = build_document(
        abstract=abstract,
        claim_1=claim_1,
        include_abstract=cfg.include_abstract,
        claims=cfg.claims,
    )

    result = WalkResult()
    # Empty id "" means root in IPCAccessor; children are sections A-H.
    now_id = ""
    ancestor_labels = [cfg.root_label]

    for _ in range(cfg.max_hops):
        try:
            children: list[IPCNode] = accessor.get_children(now_id)
        except KeyError:
            result.stop_reason = "unknown_node"
            return result

        display_now = cfg.root_label if now_id == "" else now_id
        titles = title_path(ancestor_labels)

        if not children:
            result.stop_reason = "leaf"
            return result

        messages = build_hop_messages(
            document=document,
            current_id=display_now,
            titles=titles,
            children=children,
            stop_token=cfg.stop_token,
            justify=cfg.justify,
        )
        raw = client.chat(messages)
        allowed = {child.node_id for child in children}
        try:
            parsed = parse_hop_response(
                raw,
                allowed_codes=allowed,
                stop_token=cfg.stop_token,
                justify=cfg.justify,
            )
        except HopParseError as exc:
            result.stop_reason = f"parse_error: {exc}"
            result.hops.append(
                HopRecord(
                    now_id=display_now,
                    titles=titles,
                    child_ids=sorted(allowed),
                    raw_response=raw,
                    parsed=ParsedHop(justification="", choice="", is_stop=True),
                )
            )
            return result

        result.hops.append(
            HopRecord(
                now_id=display_now,
                titles=titles,
                child_ids=[c.node_id for c in children],
                raw_response=raw,
                parsed=parsed,
            )
        )

        if parsed.is_stop:
            result.stop_reason = "stop"
            return result

        result.path.append(parsed.choice)
        now_id = parsed.choice
        try:
            node = accessor.get(parsed.choice)
            ancestor_labels.append(node.title.strip() or node.node_id)
        except KeyError:
            ancestor_labels.append(parsed.choice)

    result.stop_reason = "max_hops"
    return result
