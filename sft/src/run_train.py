"""QLoRA SFT via TRL / PEFT (CUDA). Trains on train+val only; never fits on test."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    DEFAULT_GENERATOR,
    SFT_DATASETS,
    load_config,
    resolve_path,
    user_turn,
)
from holdings import resolve_generator_dir
from io_local import iter_jsonl_gz


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "error: torch is required. On a GPU droplet install CUDA torch, then "
            "pip install -r requirements-sft.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "error: CUDA is not available. run_sft.py targets DigitalOcean / NVIDIA "
            "GPUs (QLoRA + bitsandbytes). Use a GPU droplet, or prepare data only "
            "with scripts/prepare_sft_data.py on CPU."
        )


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(iter_jsonl_gz(path))


def _maybe_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit < 0:
        return rows
    return rows[:limit]


def _dtype_from_name(name: str) -> Any:
    import torch

    key = (name or "bfloat16").lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QLoRA SFT on a prepared flat dataset (train/val only)."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--generator",
        default=None,
        help=(
            "Generator model id or slug under data/derived/sft/ "
            f"(default: config generator, currently {DEFAULT_GENERATOR})"
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=SFT_DATASETS,
        help="Prepared SFT dataset id",
    )
    p.add_argument("--model", default=None, help="Override model.name")
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap train rows for a smoke run",
    )
    p.add_argument("--run-name", default=None, help="Subdir under runs_root")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override paths.output_root (prepared data root)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_cuda()

    cfg_path = resolve_path(args.config)
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)
    paths = cfg.get("paths") or {}
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("train") or {}
    qlora_cfg = cfg.get("qlora") or {}
    lora_cfg = cfg.get("lora") or {}
    generator = (
        args.generator
        or str(cfg.get("generator") or "").strip()
        or DEFAULT_GENERATOR
    )

    data_root = resolve_path(
        args.data_root
        or Path(paths.get("output_root") or "data/derived/sft")
    )
    runs_root = resolve_path(Path(paths.get("runs_root") or "data/derived/sft/runs"))

    try:
        gen_dir = resolve_generator_dir(data_root, generator=generator)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: run scripts/prepare_sft_data.py --generator … --dataset all",
            file=sys.stderr,
        )
        return 1

    dataset = args.dataset
    ds_dir = gen_dir / dataset
    train_path = ds_dir / "train.jsonl.gz"
    val_path = ds_dir / "val.jsonl.gz"
    if not train_path.is_file():
        print(f"error: missing {train_path}", file=sys.stderr)
        return 1

    train_rows = _maybe_limit(_load_jsonl_gz(train_path), args.limit)
    val_rows = _load_jsonl_gz(val_path) if val_path.is_file() else []
    if not train_rows:
        print("error: empty train set", file=sys.stderr)
        return 1

    model_name = (args.model or model_cfg.get("name") or "").strip()
    if not model_name:
        print("error: model name missing", file=sys.stderr)
        return 1

    max_seq = int(
        args.max_seq_length
        if args.max_seq_length is not None
        else (train_cfg.get("max_seq_length") or 4096)
    )
    epochs = float(
        args.epochs if args.epochs is not None else (train_cfg.get("num_train_epochs") or 2)
    )

    run_name = args.run_name or (
        f"{gen_dir.name}__{dataset}__{model_name.replace('/', '-')}"
    )
    out_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else (runs_root / run_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Late imports so CPU prepare runs without GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    def to_text(row: dict[str, Any]) -> dict[str, str]:
        messages = [
            {
                "role": "user",
                "content": user_turn(
                    str(row.get("instruction") or ""),
                    str(row.get("input") or ""),
                ),
            },
            {
                "role": "assistant",
                "content": str(row.get("output") or ""),
            },
        ]
        return {"messages": messages}

    train_ds = Dataset.from_list([to_text(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_text(r) for r in val_rows]) if val_rows else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = _dtype_from_name(str(qlora_cfg.get("bnb_4bit_compute_dtype") or "bfloat16"))
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qlora_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qlora_cfg.get("bnb_4bit_quant_type") or "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r") or 16),
        lora_alpha=int(lora_cfg.get("lora_alpha") or 32),
        lora_dropout=float(lora_cfg.get("lora_dropout") or 0.05),
        target_modules=list(
            lora_cfg.get("target_modules")
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ),
        bias=str(lora_cfg.get("bias") or "none"),
        task_type=str(lora_cfg.get("task_type") or "CAUSAL_LM"),
    )

    use_bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_bf16_supported()
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": int(
            train_cfg.get("per_device_train_batch_size") or 1
        ),
        "per_device_eval_batch_size": int(
            train_cfg.get("per_device_train_batch_size") or 1
        ),
        "gradient_accumulation_steps": int(
            train_cfg.get("gradient_accumulation_steps") or 8
        ),
        "learning_rate": float(train_cfg.get("learning_rate") or 2e-4),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type") or "cosine"),
        "warmup_ratio": float(train_cfg.get("warmup_ratio") or 0.03),
        "logging_steps": int(train_cfg.get("logging_steps") or 10),
        "save_steps": int(train_cfg.get("save_steps") or 200),
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "packing": bool(train_cfg.get("packing", False)),
        "seed": int(train_cfg.get("seed") or 42),
        "report_to": [],
        "completion_only_loss": True,
    }
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = str(train_cfg.get("eval_strategy") or "steps")
        sft_kwargs["eval_steps"] = int(train_cfg.get("eval_steps") or 200)
    else:
        sft_kwargs["eval_strategy"] = "no"

    # TRL renamed max_seq_length → max_length across versions.
    try:
        sft_args = SFTConfig(max_length=max_seq, **sft_kwargs)
    except TypeError:
        sft_kwargs.pop("completion_only_loss", None)
        try:
            sft_args = SFTConfig(max_seq_length=max_seq, **sft_kwargs)
        except TypeError:
            sft_kwargs.pop("eval_strategy", None)
            sft_kwargs["evaluation_strategy"] = (
                str(train_cfg.get("eval_strategy") or "steps")
                if eval_ds is not None
                else "no"
            )
            sft_args = SFTConfig(max_seq_length=max_seq, **sft_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": peft_config,
    }
    # TRL API: processing_class (newer) vs tokenizer (older).
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    print(
        f"Training dataset={dataset} model={model_name} "
        f"train_n={len(train_rows)} val_n={len(val_rows)} "
        f"max_seq={max_seq} epochs={epochs} → {out_dir}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    train_config = {
        "generator_slug": gen_dir.name,
        "dataset": dataset,
        "model": model_name,
        "max_seq_length": max_seq,
        "num_train_epochs": epochs,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path.is_file() else None,
        "output_dir": str(out_dir),
        "qlora": qlora_cfg,
        "lora": {
            "r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "target_modules": list(peft_config.target_modules or []),
        },
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Trained on train/val only; test split must stay held out.",
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved adapter + train_config.json → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
