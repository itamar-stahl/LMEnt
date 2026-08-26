"""Local Transformers connector for a hosted Gemma instruction model."""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_RATING_RE = re.compile(r"Rating\s*:\s*\[\[\s*([012])\s*\]\]", re.IGNORECASE)


class GemmaJudge:
    """Expose one loaded Gemma model through EMBER's four callback seams."""

    def __init__(self, *, model: Any, tokenizer: Any, device: str = "cuda",
                 max_new_tokens: int = 256) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str = "cuda",
        max_new_tokens: int = 256,
        local_files_only: bool = False,
        cache_dir: Optional[str | Path] = None,
    ) -> "GemmaJudge":
        """Load one text-only Gemma judge, normally in bf16 on an H100."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("The Gemma judge requested CUDA, but CUDA is unavailable")
        source = str(model_name_or_path)
        common = {
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": bool(local_files_only),
        }
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
            **common,
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Gemma judge prompt must be a non-empty string")
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_length = int(inputs["input_ids"].shape[-1])
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        response = self.tokenizer.decode(
            generated[0, input_length:], skip_special_tokens=True)
        if not response.strip():
            raise ValueError("Gemma judge returned an empty response")
        return response.strip()

    def describe_feature(self, prompt: str) -> str:
        return self.generate(prompt)

    @staticmethod
    def _classification_json(response: str) -> str:
        decoder = json.JSONDecoder()
        payload = None
        for offset, character in enumerate(response):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(response[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            raise ValueError("Gemma classification response contains no JSON object")
        is_member = payload.get("is_member")
        confidence = payload.get("confidence")
        if not isinstance(is_member, bool):
            raise ValueError("Gemma classification JSON needs boolean 'is_member'")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Gemma classification JSON needs numeric 'confidence'")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Gemma classification confidence must be in [0, 1]")
        return json.dumps({"is_member": is_member, "confidence": confidence})

    def classify_feature(self, prompt: str) -> str:
        return self._classification_json(self.generate(prompt))

    @staticmethod
    def _rating(response: str) -> str:
        match = _RATING_RE.search(response)
        if match is None:
            raise ValueError(
                "Gemma Alpaca response must contain Rating: [[0]], [[1]], or [[2]]")
        return f"Rating: [[{match.group(1)}]]"

    def score_alpaca_relevance(self, prompt: str) -> str:
        return self._rating(self.generate(prompt))

    def score_alpaca_fluency(self, prompt: str) -> str:
        return self._rating(self.generate(prompt))

    def close(self) -> None:
        """Release the judge before another large model is loaded."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["GemmaJudge"]
