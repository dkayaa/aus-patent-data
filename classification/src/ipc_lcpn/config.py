# Read classification/config/ipc_lcpn.yaml into IPCLCPNConfig.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "classification" / "config" / "ipc_lcpn.yaml"

_CLAIMS_MODES = frozenset({"claim_1", "omit"})
_PROVIDERS = frozenset({"local", "openrouter", "hf"})
_MODES = frozenset({"walk", "hops"})


@dataclass(frozen=True)
class IPCLCPNConfig:
    stop_token: str
    back_token: str
    root_label: str
    include_abstract: bool
    claims: str
    justify: bool
    max_hops: int
    mode: str
    hops_path: Path | None
    llm: dict[str, Any]
    hf: dict[str, Any]
    output_dir: Path
    config_path: Path

    @property
    def provider(self) -> str:
        return str(self.llm.get("provider") or "local").strip().lower()


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path | None = None) -> IPCLCPNConfig:
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")

    claims = str(raw.get("claims") or "claim_1").strip().lower()
    if claims not in _CLAIMS_MODES:
        raise ValueError(f"claims must be one of {sorted(_CLAIMS_MODES)}, got {claims!r}")

    mode = str(raw.get("mode") or "walk").strip().lower()
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    hops_raw = raw.get("hops_path")
    hops_path = None
    if hops_raw is not None and str(hops_raw).strip() not in {"", "null", "None"}:
        hops_path = resolve_path(str(hops_raw))

    llm = dict(raw.get("llm") or {})
    provider = str(llm.get("provider") or "local").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"llm.provider must be one of {sorted(_PROVIDERS)}, got {provider!r}")
    llm["provider"] = provider

    hf = dict(raw.get("hf") or {})
    paths = dict(raw.get("paths") or {})
    output_dir = resolve_path(paths.get("output_dir") or "data/derived/ipc_lcpn")

    back_raw = raw.get("back_token")
    back_token = "" if back_raw is None else str(back_raw).strip()
    if back_token.lower() in {"null", "none"}:
        back_token = ""

    return IPCLCPNConfig(
        stop_token=str(raw.get("stop_token") or "<STOP>").strip() or "<STOP>",
        back_token=back_token,
        root_label=str(raw.get("root_label") or "<ROOT>").strip() or "<ROOT>",
        include_abstract=bool(raw.get("include_abstract", True)),
        claims=claims,
        justify=bool(raw.get("justify", True)),
        max_hops=int(raw.get("max_hops") or 16),
        mode=mode,
        hops_path=hops_path,
        llm=llm,
        hf=hf,
        output_dir=output_dir,
        config_path=cfg_path,
    )
