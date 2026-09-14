"""Inference-only Anima tokenization and reference preprocessing.

The full training strategy imports dataset, parser, and cache infrastructure
that is irrelevant to the fail-closed V7 Comfy runtime.  This audited overlay
keeps only the two operations used at inference and has no prompt-rewrite or
clause-mask API.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageOps
import torch


def preprocess_anima_reference_image(
    image,
    *,
    max_area: Optional[int],
    multiple_of: int,
    target_size_hw: Optional[Tuple[int, int]] = None,
    flipped: bool = False,
) -> Image.Image:
    """Apply the canonical independent-reference or aligned-canvas transform."""

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(image[:, :, :3]).convert("RGB")

    if target_size_hw is not None:
        target_h, target_w = (int(target_size_hw[0]), int(target_size_hw[1]))
        image = ImageOps.fit(
            image,
            (target_w, target_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        if flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    if max_area is not None and max_area > 0 and image.width * image.height > max_area:
        scale = math.sqrt(max_area / (image.width * image.height))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    width = (image.width // int(multiple_of)) * int(multiple_of)
    height = (image.height // int(multiple_of)) * int(multiple_of)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Reference image is too small after alignment: {image.width}x{image.height}. "
            f"Both dimensions must be at least {multiple_of}px."
        )
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


class AnimaTokenizeStrategy:
    """Exact dual tokenizer used by the standard Anima text path."""

    def __init__(
        self,
        qwen3_tokenizer,
        t5_tokenizer,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
    ) -> None:
        if qwen3_tokenizer is None or t5_tokenizer is None:
            raise ValueError("Both preloaded tokenizers are required by the V7 runtime.")
        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = int(qwen3_max_length)
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = int(t5_max_length)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        qwen = self.qwen3_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.qwen3_max_length,
        )
        neutral_t5 = self.t5_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.t5_max_length,
        )
        return [
            qwen["input_ids"],
            qwen["attention_mask"],
            neutral_t5["input_ids"],
            neutral_t5["attention_mask"],
        ]


__all__ = ["AnimaTokenizeStrategy", "preprocess_anima_reference_image"]
