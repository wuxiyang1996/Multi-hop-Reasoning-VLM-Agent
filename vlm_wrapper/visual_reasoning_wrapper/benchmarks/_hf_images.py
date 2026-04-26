"""Decode HuggingFace ``datasets`` image fields to ``PIL.Image``."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image


def decode_hf_image(value: Any) -> Image.Image:
    """Turn a HF row image cell (PIL, ``{path, bytes}``, or filesystem path) into RGB."""
    if value is None:
        raise ValueError("image field is None")
    if hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            p = Path(value["path"])
            if p.exists():
                return Image.open(p).convert("RGB")
    if isinstance(value, str):
        p = Path(value)
        if p.is_file():
            return Image.open(p).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")
