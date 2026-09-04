"""Guard IPC teacher/pool prompts against memo-style and train/test mismatch."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.ipc_reasoning import (  # noqa: E402
    INSTRUCTION_POOL_PROMPT,
    JUSTIFICATION_INSTRUCTION,
    JUSTIFICATION_TRAILER,
)

POOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "derived"
    / "instruction_generation"
    / "_pools"
    / "ipc_reasoning.json"
)


class IPCPromptTests(unittest.TestCase):
    def test_teacher_is_paragraph_mapping_not_two_sentences(self) -> None:
        text = f"{JUSTIFICATION_INSTRUCTION}\n{JUSTIFICATION_TRAILER}".lower()
        self.assertNotIn("two-sentence", text)
        self.assertNotIn("two sentences", text)
        self.assertNotIn("one sentence stating what independent claim 1", text)
        self.assertNotIn("do not start with", text)
        self.assertIn("GOLD", JUSTIFICATION_INSTRUCTION)
        self.assertIn("do not mention other ipc codes", text)
        self.assertIn("do not follow a fixed outline", text)

    def test_pool_meta_prompt_is_assign_and_justify(self) -> None:
        text = INSTRUCTION_POOL_PROMPT.lower()
        self.assertIn("assign an ipc code", text)
        self.assertIn("justifying", text)
        self.assertNotIn("already-assigned", text)
        self.assertNotIn("do not ask to re-classify", text)
        self.assertIn("must not request", text)
        self.assertIn("multi-section", text)

    def test_cached_pool_matches_target(self) -> None:
        data = json.loads(POOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 40)
        blob = "\n".join(data).lower()
        self.assertNotIn("examiner memorandum that systematically", blob)
        self.assertNotIn("hierarchical position", blob)
        # Student task is assign+justify, not "code is fixed / do not re-classify".
        fixed_markers = (
            "the code is fixed",
            "do not re-classify",
            "do not reclassify",
            "already-assigned",
            "already assigned",
            "treat the assigned code as given",
            "do not suggest a different code",
            "do not propose a different code",
            "do not recommend a change of classification",
            "do not ask to re-classify",
        )
        for marker in fixed_markers:
            self.assertNotIn(marker, blob, msg=f"stale marker in pool: {marker!r}")
        assignish = 0
        for item in data:
            self.assertTrue(str(item).strip())
            low = str(item).lower()
            if any(
                w in low
                for w in (
                    "assign",
                    "select",
                    "choose",
                    "determine",
                    "identify",
                    "propose",
                    "classify",
                )
            ):
                assignish += 1
        self.assertGreaterEqual(assignish, 30)


if __name__ == "__main__":
    unittest.main()
