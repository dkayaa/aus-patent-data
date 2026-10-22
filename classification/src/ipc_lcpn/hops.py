# Score stored hop records independently. Each row already contains the office path.

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .client import ChatClient
from .config import IPCLCPNConfig
from .parse import HopParseError, parse_hop_response


@dataclass
class HopScore:
    application_number: str
    hop_index: int | None
    gold: str
    choice: str
    correct: bool
    parse_error: str
    raw_response: str


@dataclass
class HopRun:
    scores: list[HopScore] = field(default_factory=list)

    @property
    def n_hops(self) -> int:
        return len(self.scores)

    @property
    def n_correct(self) -> int:
        return sum(1 for score in self.scores if score.correct)

    @property
    def n_parse_errors(self) -> int:
        return sum(1 for score in self.scores if score.parse_error)

    @property
    def accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return self.n_correct / len(self.scores)


def iter_hop_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                yield row


def hop_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Training prompt: instruction, then the stored hop input. No system turn."""
    instruction = str(record.get("instruction") or "").strip()
    input_text = str(record.get("input") or "").strip()
    if instruction and input_text:
        user = f"{instruction}\n\n{input_text}"
    else:
        user = instruction or input_text
    return [{"role": "user", "content": user}]


def menu_codes(
    input_text: str,
    *,
    stop_token: str,
    back_token: str = "",
) -> set[str]:
    codes: set[str] = set()
    in_children = False
    for line in input_text.splitlines():
        if line.startswith("Children:"):
            in_children = True
            continue
        if not in_children or ". " not in line or " - " not in line:
            continue
        code = line.split(". ", 1)[1].split(" - ", 1)[0].strip()
        if code and code != stop_token and code != back_token:
            codes.add(code)
    return codes


def _record_justify(record: dict[str, Any], cfg: IPCLCPNConfig) -> bool:
    meta = record.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("justify"), bool):
        return bool(meta["justify"])
    return cfg.justify


def _record_stop_token(record: dict[str, Any], cfg: IPCLCPNConfig) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict):
        token = str(meta.get("stop_token") or "").strip()
        if token:
            return token
    return cfg.stop_token


def _record_back_token(record: dict[str, Any], cfg: IPCLCPNConfig) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict) and "back_token" in meta:
        return str(meta.get("back_token") or "").strip()
    return cfg.back_token


def score_hops(
    records: list[dict[str, Any]],
    *,
    client: ChatClient,
    cfg: IPCLCPNConfig,
) -> HopRun:
    run = HopRun()
    total = len(records)
    for index, record in enumerate(records, start=1):
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        gold = str(meta.get("gold_node_id") or "").strip()
        stop_token = _record_stop_token(record, cfg)
        back_token = _record_back_token(record, cfg)
        justify = _record_justify(record, cfg)
        allowed = menu_codes(
            str(record.get("input") or ""),
            stop_token=stop_token,
            back_token=back_token,
        )
        raw = client.chat(hop_messages(record))
        choice = ""
        error = ""
        correct = False
        try:
            parsed = parse_hop_response(
                raw,
                allowed_codes=allowed,
                stop_token=stop_token,
                justify=justify,
                back_token=back_token,
            )
            choice = parsed.choice
            correct = bool(gold) and choice == gold
        except HopParseError as exc:
            error = str(exc)
        hop_index = meta.get("hop_index")
        run.scores.append(
            HopScore(
                application_number=str(record.get("application_number") or ""),
                hop_index=int(hop_index) if isinstance(hop_index, int) else None,
                gold=gold,
                choice=choice,
                correct=correct,
                parse_error=error,
                raw_response=raw,
            )
        )
        if index % 50 == 0 or index == total:
            print(
                f"hops {index}/{total} correct={run.n_correct} "
                f"accuracy={run.accuracy:.3f}",
                flush=True,
            )
    return run
