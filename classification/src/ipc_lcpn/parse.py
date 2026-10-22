# Read the model reply and pull out the chosen child (or STOP).

from __future__ import annotations

import re
from dataclasses import dataclass

_CLASSIFICATION_RE = re.compile(
    r"(?im)^\s*Classification:\s*(.+?)\s*$"
)

class HopParseError(ValueError):
    # Raised when the reply has no usable Classification line / choice.
    pass

@dataclass(frozen=True)
class ParsedHop:
    justification: str
    choice: str
    is_stop: bool
    is_back: bool = False


def _norm_choice(text: str) -> str:
    return text.strip().upper().replace(" ", "")

# If justify is true: find the Classification: line; text above it is the justification.
# If justify is false: treat the first line as the choice.
# Choice must be stop_token or one of allowed_codes.
def parse_hop_response(
    text: str,
    *,
    allowed_codes: set[str],
    stop_token: str,
    justify: bool = True,
    back_token: str = "",
) -> ParsedHop:
  
    raw = (text or "").strip()
    if not raw:
        raise HopParseError("empty model response")

    stop_norm = _norm_choice(stop_token)
    back_norm = _norm_choice(back_token) if back_token else ""
    allowed_norm = {_norm_choice(code): code for code in allowed_codes}

    if justify:
        match = None
        for match in _CLASSIFICATION_RE.finditer(raw):
            pass
        if match is None:
            raise HopParseError("missing Classification: line")
        choice_raw = match.group(1).strip()
        justification = raw[: match.start()].strip()
    else:
        choice_raw = raw.splitlines()[0].strip()
        justification = ""

    choice_norm = _norm_choice(choice_raw)
    choice_norm = choice_norm.rstrip(".,;")

    if choice_norm == stop_norm:
        return ParsedHop(
            justification=justification,
            choice=stop_token,
            is_stop=True,
        )

    if back_norm and choice_norm == back_norm:
        return ParsedHop(
            justification=justification,
            choice=back_token,
            is_stop=False,
            is_back=True,
        )

    if choice_norm in allowed_norm:
        return ParsedHop(
            justification=justification,
            choice=allowed_norm[choice_norm],
            is_stop=False,
        )

    if back_token:
        raise HopParseError(
            f"choice {choice_raw!r} is not STOP, not BACK, and not in the child menu"
        )
    raise HopParseError(
        f"choice {choice_raw!r} is not STOP and not in the child menu"
    )
