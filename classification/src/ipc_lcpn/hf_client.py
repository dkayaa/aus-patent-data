# Load a Hugging Face model (and optional LoRA adapter from sft/) and chat.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HFClientConfig:
    model: str
    adapter_path: Path | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0

# Chat with a Hugging Face model locally via transformers.
class HFClient:
    def __init__(self, config: HFClientConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "error: hf provider needs torch and transformers. "
                "On a GPU machine: install CUDA torch, then "
                "pip install -r requirements-sft.txt"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        adapter = self.config.adapter_path
        if adapter is not None:
            adapter = Path(adapter)
            if not adapter.is_dir():
                raise FileNotFoundError(f"adapter_path not found: {adapter}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise SystemExit(
                    "error: adapter_path is set but peft is missing "
                    "(pip install -r requirements-sft.txt)"
                ) from exc
            model = PeftModel.from_pretrained(model, str(adapter))

        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._ensure_loaded()
        import torch

        temp = self.config.temperature if temperature is None else temperature
        max_new = self.config.max_new_tokens if max_tokens is None else max_tokens

        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        do_sample = temp is not None and float(temp) > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new),
                do_sample=do_sample,
                temperature=float(temp) if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("HF model returned empty content")
        return text.strip()
