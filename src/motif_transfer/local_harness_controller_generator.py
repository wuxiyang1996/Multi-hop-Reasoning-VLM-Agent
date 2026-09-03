"""Local greedy Qwen/PEFT generator for the neural Harness controller."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalHarnessControllerGenerator:
    """Load one frozen adapter and reproduce its JSON-only greedy inference."""

    def __init__(
        self, *, model_name: str, adapter: Path,
        expected_adapter_sha256: str, device: str = "cuda:0",
        max_input_length: int = 1792, max_new_tokens: int = 256,
    ):
        adapter = adapter.resolve()
        adapter_file = adapter / "adapter_model.safetensors"
        actual = _sha256(adapter_file)
        if actual != str(expected_adapter_sha256):
            raise ValueError("local Harness adapter hash mismatch")
        if max_input_length < 1 or max_new_tokens < 1:
            raise ValueError("generation token limits must be positive")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        self.artifact_sha256 = actual
        self.model_name = str(model_name)
        self.adapter = adapter
        self.device = device
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        model = self._model
        encoded = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_input_length,
        ).to(model.device)
        with self._torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        suffix = generated[:, encoded["input_ids"].shape[1]:]
        return tokenizer.batch_decode(suffix, skip_special_tokens=True)[0]


__all__ = ["LocalHarnessControllerGenerator"]
