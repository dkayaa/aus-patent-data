"""Fetch patent records from the Australian Patent Search API.

Reads application numbers from a base CSV, GETs
``{base_url}/patent/{application_number}``, and writes one JSON file per
application under ``data/interim/``. Existing output files are skipped so
re-runs are idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scrape" / "config" / "patent_search.yaml"


@dataclass(frozen=True)
class BackoffConfig:
    initial_seconds: float
    max_seconds: float
    multiplier: float
    max_retries: int


@dataclass(frozen=True)
class FetchConfig:
    base_url: str
    patent_path_template: str
    token_url: str
    client_id: str
    client_secret: str
    input_csv: Path
    application_number_column: str
    output_dir: Path
    max_responses: int | None
    request_timeout_seconds: float
    max_requests_per_minute: int
    rate_limit_headroom: float
    backoff: BackoffConfig

    @property
    def min_interval_seconds(self) -> float:
        """Minimum spacing between API calls from the per-minute cap."""
        effective = max(1, int(self.max_requests_per_minute * self.rate_limit_headroom))
        return 60.0 / effective


class RateLimiter:
    """Space requests so we stay under max_requests_per_minute."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _display_path(path: Path) -> str:
    """Prefer repo-relative path in logs; fall back to absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def redact_token(token: str) -> str:
    """Log-safe token preview: first 3 chars + ... + last 3 chars."""
    if len(token) <= 6:
        return "..."
    return f"{token[:3]}...{token[-3:]}"


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def log_preflight(cfg: FetchConfig) -> None:
    """Log hosts and credential presence without printing secrets."""
    token_host = _host(cfg.token_url)
    api_host = _host(cfg.base_url)
    same_env = token_host == api_host
    logger.info(
        "preflight token_host=%s api_host=%s same_host=%s "
        "client_id_set=%s client_id_len=%s client_secret_set=%s client_secret_len=%s "
        "max_requests_per_minute=%s headroom=%s min_interval_seconds=%.3f "
        "token_url=%s base_url=%s",
        token_host,
        api_host,
        same_env,
        bool(cfg.client_id),
        len(cfg.client_id),
        bool(cfg.client_secret),
        len(cfg.client_secret),
        cfg.max_requests_per_minute,
        cfg.rate_limit_headroom,
        cfg.min_interval_seconds,
        cfg.token_url,
        cfg.base_url,
    )
    if not same_env:
        logger.warning(
            "token host and API host differ; use matching test/prod credentials "
            "for the token URL"
        )


def load_config(
    config_path: Path,
    *,
    client_id_override: str | None = None,
    client_secret_override: str | None = None,
) -> FetchConfig:
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    auth = raw["auth"]
    client_id_env = auth["client_id_env"]
    client_secret_env = auth["client_secret_env"]
    client_id = (client_id_override or _env(client_id_env)).strip()
    client_secret = (client_secret_override or _env(client_secret_env)).strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Missing OAuth credentials. Set "
            f"{client_id_env} and {client_secret_env}, or pass "
            "--client-id / --client-secret. "
            "JWT is obtained automatically from the External Token API."
        )

    fetch = raw["fetch"]
    backoff = fetch["backoff"]
    max_responses = fetch.get("max_responses")
    if max_responses is not None:
        max_responses = int(max_responses)

    # Steady-state pacing from published per-minute cap.
    max_rpm = int(fetch.get("max_requests_per_minute", 500))
    headroom = float(fetch.get("rate_limit_headroom", 0.9))
    if not 0.0 < headroom <= 1.0:
        raise SystemExit("fetch.rate_limit_headroom must be in (0, 1]")

    return FetchConfig(
        base_url=raw["api"]["base_url"].rstrip("/"),
        patent_path_template=raw["api"]["patent_path_template"],
        token_url=auth["token_url"],
        client_id=client_id,
        client_secret=client_secret,
        input_csv=_repo_path(raw["paths"]["input_csv"]),
        application_number_column=raw["paths"]["application_number_column"],
        output_dir=_repo_path(raw["paths"]["output_dir"]),
        max_responses=max_responses,
        request_timeout_seconds=float(fetch["request_timeout_seconds"]),
        max_requests_per_minute=max_rpm,
        rate_limit_headroom=headroom,
        backoff=BackoffConfig(
            initial_seconds=float(backoff["initial_seconds"]),
            max_seconds=float(backoff["max_seconds"]),
            multiplier=float(backoff["multiplier"]),
            max_retries=int(backoff["max_retries"]),
        ),
    )


def fetch_access_token(cfg: FetchConfig) -> str:
    """Exchange client_id/client_secret for a Bearer JWT."""
    body = urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
        }
    ).encode("utf-8")
    req = Request(
        cfg.token_url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=cfg.request_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise SystemExit(
            f"Token request failed: HTTP {exc.code} {exc.reason}. {detail}"
        ) from exc
    except URLError as exc:
        raise SystemExit(f"Token request failed: {exc}") from exc

    token = payload.get("access_token")
    if not token or not isinstance(token, str):
        raise SystemExit(
            "Token response missing access_token. "
            f"Keys present: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )
    token_type = payload.get("token_type", "Bearer")
    expires_in = payload.get("expires_in")
    logger.info(
        "obtained access token (type=%s expires_in=%s token=%s)",
        token_type,
        expires_in,
        redact_token(token),
    )
    return token


def read_application_numbers(csv_path: Path, column: str) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header row in {csv_path}")
        if column not in reader.fieldnames:
            raise SystemExit(
                f"Column {column!r} not in {csv_path}; "
                f"found {reader.fieldnames}"
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for row in reader:
            value = (row.get(column) or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return ordered


def output_path_for(output_dir: Path, application_number: str) -> Path:
    # Application numbers are numeric; keep filename safe anyway.
    safe = "".join(c for c in application_number if c.isalnum() or c in "-_")
    return output_dir / f"{safe}.json"


def already_fetched(output_dir: Path, application_number: str) -> bool:
    return output_path_for(output_dir, application_number).is_file()


def patent_url(cfg: FetchConfig, application_number: str) -> str:
    path = cfg.patent_path_template.format(ip_right_identifier=application_number)
    if not path.startswith("/"):
        path = "/" + path
    return f"{cfg.base_url.rstrip('/')}{path}"


def _should_retry(status: int | None, exc: BaseException | None) -> bool:
    if isinstance(exc, URLError) and not isinstance(exc, HTTPError):
        return True
    if status is None:
        return True
    return status in {408, 425, 429, 500, 502, 503, 504}


def fetch_patent(
    cfg: FetchConfig,
    application_number: str,
    access_token: str,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    url = patent_url(cfg, application_number)
    delay = cfg.backoff.initial_seconds
    last_error: BaseException | None = None

    for attempt in range(cfg.backoff.max_retries + 1):
        rate_limiter.wait()
        logger.info("GET %s (token=%s)", url, redact_token(access_token))
        req = Request(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=cfg.request_timeout_seconds) as resp:
                body = resp.read()
                status = getattr(resp, "status", 200)
                if status >= 400:
                    raise HTTPError(url, status, resp.reason, resp.headers, None)
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected JSON object for {application_number}, "
                        f"got {type(payload).__name__}"
                    )
                return payload
        except HTTPError as exc:
            last_error = exc
            status = exc.code
            if status == 404:
                raise
            if attempt >= cfg.backoff.max_retries or not _should_retry(status, exc):
                raise
            logger.warning(
                "HTTP %s for %s url=%s (attempt %s/%s); backing off %.1fs",
                status,
                application_number,
                url,
                attempt + 1,
                cfg.backoff.max_retries + 1,
                delay,
            )
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= cfg.backoff.max_retries or not _should_retry(None, exc):
                raise
            logger.warning(
                "Error fetching %s url=%s (attempt %s/%s): %s; backing off %.1fs",
                application_number,
                url,
                attempt + 1,
                cfg.backoff.max_retries + 1,
                exc,
                delay,
            )

        time.sleep(delay)
        delay = min(delay * cfg.backoff.multiplier, cfg.backoff.max_seconds)

    assert last_error is not None
    raise last_error


def write_response(
    output_dir: Path,
    application_number: str,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path_for(output_dir, application_number)
    # Write via temp file then rename for crash-safe idempotency.
    tmp = path.with_suffix(".json.tmp")
    record = {
        "application_number": application_number,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": payload,
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
    return path


def run(cfg: FetchConfig) -> int:
    log_preflight(cfg)
    numbers = read_application_numbers(cfg.input_csv, cfg.application_number_column)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pending = [n for n in numbers if not already_fetched(cfg.output_dir, n)]
    skipped = len(numbers) - len(pending)
    if cfg.max_responses is not None:
        pending = pending[: cfg.max_responses]

    logger.info(
        "applications=%s skipped_existing=%s to_fetch=%s output=%s",
        len(numbers),
        skipped,
        len(pending),
        cfg.output_dir,
    )

    if not pending:
        logger.info("nothing to fetch")
        return 0

    access_token = fetch_access_token(cfg)
    rate_limiter = RateLimiter(cfg.min_interval_seconds)

    fetched = 0
    failures = 0
    for application_number in pending:
        # Re-check in case of concurrent runs / partial prior write.
        if already_fetched(cfg.output_dir, application_number):
            logger.info("skip existing %s", application_number)
            continue
        url = patent_url(cfg, application_number)
        try:
            payload = fetch_patent(
                cfg, application_number, access_token, rate_limiter
            )
            path = write_response(cfg.output_dir, application_number, payload)
            fetched += 1
            logger.info("wrote %s (url=%s)", _display_path(path), url)
        except HTTPError as exc:
            failures += 1
            logger.error(
                "failed %s url=%s: HTTP %s %s",
                application_number,
                url,
                exc.code,
                exc.reason,
            )
        except Exception as exc:  # noqa: BLE001 — keep run going on per-row errors
            failures += 1
            logger.error("failed %s url=%s: %s", application_number, url, exc)

    logger.info("done fetched=%s failures=%s skipped_existing=%s", fetched, failures, skipped)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Idempotent Australian Patent Search API enrichment."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--client-id",
        default=None,
        help="OAuth client_id override (else env from config auth.client_id_env)",
    )
    p.add_argument(
        "--client-secret",
        default=None,
        help="OAuth client_secret override (else env from config auth.client_secret_env)",
    )
    p.add_argument(
        "--max-responses",
        type=int,
        default=None,
        help="Optional cap on new fetches this run (overrides config)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = load_config(
        args.config.resolve(),
        client_id_override=args.client_id,
        client_secret_override=args.client_secret,
    )
    if args.max_responses is not None:
        cfg = replace(cfg, max_responses=args.max_responses)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
