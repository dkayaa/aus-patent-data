# Walk each test filing from the root and compare the place it stops at to primary_ipc.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ipc.accessor import IPCAccessor

from .client import ChatClient
from .config import IPCLCPNConfig
from .walk import WalkResult, walk_ipc

_COMPLETED = {"stop", "leaf"}


@dataclass
class FilingWalk:
    application_number: str
    primary_ipc: str
    predicted: str
    exact: bool
    path: list[str]
    gold_path: list[str]
    stop_reason: str
    n_hops: int
    hops: list[dict]


@dataclass
class WalkRun:
    filings: list[FilingWalk] = field(default_factory=list)

    @property
    def n_filings(self) -> int:
        return len(self.filings)

    @property
    def n_exact(self) -> int:
        return sum(1 for filing in self.filings if filing.exact)

    @property
    def accuracy(self) -> float:
        if not self.filings:
            return 0.0
        return self.n_exact / len(self.filings)


def patent_text(record: dict) -> tuple[str, str]:
    """Pull abstract and claim 1 out of a stored hop input. Menus stay out."""
    text = str(record.get("input") or "")
    head = text.split("\n\nCurrent node:", 1)[0].strip()
    abstract = ""
    claim = head
    if head.startswith("Abstract:\n"):
        rest = head[len("Abstract:\n") :]
        if "\n\nClaim 1:\n" in rest:
            abstract, claim = rest.split("\n\nClaim 1:\n", 1)
        else:
            abstract, claim = rest, ""
    elif head.startswith("Claim 1:\n"):
        claim = head[len("Claim 1:\n") :]
    return abstract.strip(), claim.strip()


def filings_from_hops(records: list[dict]) -> list[dict]:
    """One record per application. Later hops of the same filing are dropped."""
    chosen: dict[str, dict] = {}
    for record in records:
        app = str(record.get("application_number") or "")
        if app and app not in chosen:
            chosen[app] = record
    return list(chosen.values())


def _gold_path(meta: dict) -> list[str]:
    nodes = [str(node) for node in (meta.get("path_node_ids") or [])]
    root = str(meta.get("root") or "<ROOT>")
    if nodes and nodes[0] == root:
        nodes = nodes[1:]
    return nodes


def _predicted(result: WalkResult) -> str:
    if result.stop_reason in _COMPLETED and result.path:
        return result.path[-1]
    return ""


def score_walks(
    records: list[dict],
    *,
    accessor: IPCAccessor,
    client: ChatClient,
    cfg: IPCLCPNConfig,
    on_filing: Callable[[int, int, FilingWalk], None] | None = None,
) -> WalkRun:
    """Free-run walk_ipc on each filing. Exact means the stopped place is primary_ipc."""
    run = WalkRun()
    filings = filings_from_hops(records)
    total = len(filings)
    for index, record in enumerate(filings, start=1):
        meta = record.get("meta") or {}
        abstract, claim_1 = patent_text(record)
        result = walk_ipc(
            accessor=accessor,
            client=client,
            cfg=cfg,
            abstract=abstract,
            claim_1=claim_1,
        )
        primary = str(meta.get("primary_ipc") or "").strip()
        gold = _gold_path(meta)
        if not primary and gold:
            primary = gold[-1]
        predicted = _predicted(result)
        row = FilingWalk(
            application_number=str(record.get("application_number") or ""),
            primary_ipc=primary,
            predicted=predicted,
            exact=bool(predicted) and predicted == primary,
            path=list(result.path),
            gold_path=gold,
            stop_reason=result.stop_reason,
            n_hops=len(result.hops),
            hops=[
                {
                    "now_id": hop.now_id,
                    "choice": hop.parsed.choice,
                    "is_stop": hop.parsed.is_stop,
                    "is_back": hop.parsed.is_back,
                    "raw_response": hop.raw_response,
                }
                for hop in result.hops
            ],
        )
        run.filings.append(row)
        if on_filing is not None:
            on_filing(index, total, row)
    return run
