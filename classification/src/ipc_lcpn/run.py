# Command-line entry: run the LCPN walk on one patent.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASS_SRC = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, CLASS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ipc.accessor import IPCAccessor  # noqa: E402

from ipc_lcpn.client import build_client, client_label  # noqa: E402
from ipc_lcpn.config import DEFAULT_CONFIG, load_config  # noqa: E402
from ipc_lcpn.walk import walk_ipc  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run student IPC LCPN classification on one patent."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--abstract", default="", help="Abstract text")
    p.add_argument("--claim-1", default="", dest="claim_1", help="Claim 1 text")
    p.add_argument(
        "--abstract-file",
        type=Path,
        default=None,
        help="Read abstract from a file",
    )
    p.add_argument(
        "--claim-1-file",
        type=Path,
        default=None,
        dest="claim_1_file",
        help="Read claim 1 from a file",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Save the step-by-step result as JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    abstract = args.abstract
    claim_1 = args.claim_1
    if args.abstract_file is not None:
        abstract = args.abstract_file.read_text(encoding="utf-8")
    if args.claim_1_file is not None:
        claim_1 = args.claim_1_file.read_text(encoding="utf-8")
    if not abstract.strip() and not claim_1.strip():
        print(
            "error: provide --abstract/--claim-1 or the matching *-file flags",
            file=sys.stderr,
        )
        return 1

    accessor = IPCAccessor()
    accessor.init()
    client = build_client(cfg)
    print(f"client={client_label(client)} stop_token={cfg.stop_token!r}", flush=True)

    result = walk_ipc(
        accessor=accessor,
        client=client,
        cfg=cfg,
        abstract=abstract,
        claim_1=claim_1,
    )
    print(f"path: {result.path_str or '(empty)'}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"n_hops: {len(result.hops)}")

    if args.json_out is not None:
        payload = {
            "path": result.path,
            "path_str": result.path_str,
            "stop_reason": result.stop_reason,
            "hops": [
                {
                    "now_id": hop.now_id,
                    "titles": hop.titles,
                    "child_ids": hop.child_ids,
                    "choice": hop.parsed.choice,
                    "is_stop": hop.parsed.is_stop,
                    "justification": hop.parsed.justification,
                    "raw_response": hop.raw_response,
                }
                for hop in result.hops
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
