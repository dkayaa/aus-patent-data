"""Shared Evol-Instruct instruction pool builder."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llm import LLMClient, chat_json


def load_or_build_pool(
    *,
    client: LLMClient,
    pools_dir: Path,
    pool_name: str,
    user_prompt: str,
    pool_size: int,
    batch_size: int,
    force_rebuild: bool = False,
) -> list[str]:
    pools_dir.mkdir(parents=True, exist_ok=True)
    path = pools_dir / f"{pool_name}.json"
    if path.is_file() and not force_rebuild:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            instructions = [str(x).strip() for x in data if str(x).strip()]
            if instructions:
                return instructions

    instructions: list[str] = []
    seen: set[str] = set()
    while len(instructions) < pool_size:
        need = min(batch_size, pool_size - len(instructions))
        prompt = user_prompt
        if need != 5:
            prompt = user_prompt.replace("Generate 5 ", f"Generate {need} ").replace(
                "generate 5 ", f"generate {need} "
            )
        batch = chat_json(
            client,
            [{"role": "user", "content": prompt}],
            expect=list,
        )
        for item in batch:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            instructions.append(text)
            if len(instructions) >= pool_size:
                break

    if not instructions:
        raise RuntimeError(f"Failed to build Evol-Instruct pool: {pool_name}")

    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
    return instructions


def sample_instruction(pool: list[str], rng: random.Random | None = None) -> str:
    if not pool:
        raise ValueError("Empty instruction pool")
    chooser = rng.choice if rng is not None else random.choice
    return chooser(pool)
