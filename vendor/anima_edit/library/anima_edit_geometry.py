"""Canonical registered-canvas geometry for Anima source editing.

The source and target of an edit pair are spatial supervision, not independent
reference photographs.  They must therefore share one integer resize/crop plan.
This module intentionally mirrors the existing target dataset's
``resize -> integer centre crop -> shared flip`` convention and is used by
training reference caching, online reference encoding, and inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from _anima_native_ref_vendor.library import utils


ANIMA_EDIT_CANVAS_TRANSFORM_V1 = (
    "resize_then_integer_center_crop_then_flip_v1"
)


@dataclass(frozen=True)
class EditCanvasPlanV1:
    """Serializable geometry contract with explicitly named tuple ordering."""

    input_wh: Tuple[int, int]
    output_hw: Tuple[int, int]
    resized_wh: Tuple[int, int]
    crop_xyxy: Tuple[int, int, int, int]
    flip_x: bool = False
    resize_interpolation: Optional[str] = None
    transform_version: str = ANIMA_EDIT_CANVAS_TRANSFORM_V1

    def validate(self) -> "EditCanvasPlanV1":
        input_w, input_h = (int(v) for v in self.input_wh)
        output_h, output_w = (int(v) for v in self.output_hw)
        resized_w, resized_h = (int(v) for v in self.resized_wh)
        left, top, right, bottom = (int(v) for v in self.crop_xyxy)
        if self.transform_version != ANIMA_EDIT_CANVAS_TRANSFORM_V1:
            raise ValueError(
                f"Unsupported edit canvas transform {self.transform_version!r}."
            )
        if min(input_w, input_h, output_w, output_h, resized_w, resized_h) <= 0:
            raise ValueError("Edit canvas dimensions must all be positive.")
        if (right - left, bottom - top) != (output_w, output_h):
            raise ValueError(
                "Edit crop size does not match output canvas: "
                f"crop={(left, top, right, bottom)}, "
                f"output_hw={(output_h, output_w)}."
            )
        if left < 0 or top < 0 or right > resized_w or bottom > resized_h:
            raise ValueError(
                "Edit crop lies outside the resized canvas: "
                f"crop={(left, top, right, bottom)}, "
                f"resized_wh={(resized_w, resized_h)}."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_edit_canvas_plan_v1(
    *,
    input_wh: Tuple[int, int],
    output_hw: Tuple[int, int],
    resized_wh: Optional[Tuple[int, int]] = None,
    flip_x: bool = False,
    resize_interpolation: Optional[str] = None,
) -> EditCanvasPlanV1:
    """Create the target-compatible positive-half-up centre-crop plan."""

    input_w, input_h = (int(v) for v in input_wh)
    output_h, output_w = (int(v) for v in output_hw)
    if min(input_w, input_h, output_w, output_h) <= 0:
        raise ValueError("Edit canvas input/output dimensions must be positive.")
    if resized_wh is None:
        source_ratio = float(input_w) / float(input_h)
        target_ratio = float(output_w) / float(output_h)
        if source_ratio > target_ratio:
            scale = float(output_h) / float(input_h)
        else:
            scale = float(output_w) / float(input_w)
        resized_w = int(input_w * scale + 0.5)
        resized_h = int(input_h * scale + 0.5)
    else:
        resized_w, resized_h = (int(v) for v in resized_wh)
    if resized_w < output_w or resized_h < output_h:
        raise ValueError(
            "Edit resized canvas cannot be smaller than the output crop: "
            f"resized_wh={(resized_w, resized_h)}, "
            f"output_hw={(output_h, output_w)}."
        )
    left = (resized_w - output_w) // 2
    top = (resized_h - output_h) // 2
    return EditCanvasPlanV1(
        input_wh=(input_w, input_h),
        output_hw=(output_h, output_w),
        resized_wh=(resized_w, resized_h),
        crop_xyxy=(left, top, left + output_w, top + output_h),
        flip_x=bool(flip_x),
        resize_interpolation=resize_interpolation,
    ).validate()


def _as_rgb_uint8(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Edit canvas input must be an RGB-like HWC image.")
        image = image[:, :, :3]
        if image.dtype != np.uint8:
            raise ValueError("Edit canvas input must use uint8 pixels.")
    return np.ascontiguousarray(image)


def _as_mask_uint8(mask: Any) -> np.ndarray:
    """Normalize a raster mask without changing its grayscale values.

    Mask polarity and raster-to-latent reduction are deliberately outside the
    geometry contract.  This helper only establishes a single-channel uint8
    canvas so training and inference replay the same resize/crop/flip plan.
    """

    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError(
                "Edit mask canvas input must be a single-channel HxW raster."
            )
        if mask.dtype != np.uint8:
            raise ValueError("Edit mask canvas input must use uint8 pixels.")
    return np.ascontiguousarray(mask)


def apply_edit_canvas_plan_v1(
    image: Any,
    plan: EditCanvasPlanV1,
    *,
    is_mask: bool = False,
) -> np.ndarray:
    """Apply one canonical registered-canvas plan.

    RGB rasters retain the target dataset's configured interpolation.  Masks
    always use bit-exact nearest-neighbour resizing and remain grayscale;
    raster-to-latent reduction is handled by its own shared soft-mask contract.
    """

    plan.validate()
    pixels = _as_mask_uint8(image) if is_mask else _as_rgb_uint8(image)
    input_w, input_h = plan.input_wh
    if (pixels.shape[1], pixels.shape[0]) != (input_w, input_h):
        raise ValueError(
            "Registered edit input does not match the plan's original canvas: "
            f"actual_wh={(pixels.shape[1], pixels.shape[0])}, "
            f"plan_wh={(input_w, input_h)}."
        )
    resized_w, resized_h = plan.resized_wh
    if (input_w, input_h) != (resized_w, resized_h):
        pixels = utils.resize_image(
            pixels,
            input_w,
            input_h,
            resized_w,
            resized_h,
            "nearest" if is_mask else plan.resize_interpolation,
        )
    left, top, right, bottom = plan.crop_xyxy
    pixels = pixels[top:bottom, left:right]
    if plan.flip_x:
        pixels = pixels[:, ::-1]
    pixels = np.ascontiguousarray(pixels)
    expected_h, expected_w = plan.output_hw
    expected_shape = (
        (expected_h, expected_w) if is_mask else (expected_h, expected_w, 3)
    )
    if pixels.shape != expected_shape:
        raise RuntimeError(
            "Canonical edit transform produced an illegal output shape: "
            f"{pixels.shape} != {expected_shape}."
        )
    return pixels


__all__ = [
    "ANIMA_EDIT_CANVAS_TRANSFORM_V1",
    "EditCanvasPlanV1",
    "apply_edit_canvas_plan_v1",
    "make_edit_canvas_plan_v1",
]
