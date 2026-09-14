"""Minimal inference utilities for the vendored Anima V7 runtime.

The upstream training utility module also imports diffusers, torchvision and
OpenCV and contains scheduler/training helpers.  None of those are part of the
published ComfyUI inference path.  Keeping only the two functions consumed by
the frozen runtime prevents an unrelated optional dependency from deciding
whether the node can be imported.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def setup_logging(args=None, log_level=None, reset: bool = False) -> None:
    """Configure the root logger without pulling in training dependencies."""

    if logging.root.handlers:
        if not reset:
            return
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    if log_level is None and args is not None:
        log_level = getattr(args, "console_log_level", None)
    if log_level is None:
        log_level = "INFO"
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    log_file = getattr(args, "console_log_file", None) if args is not None else None
    if log_file:
        handler: logging.Handler = logging.FileHandler(log_file, mode="w")
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.propagate = False

    handler.setFormatter(
        logging.Formatter(fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)


_PIL_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "area": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "lanczos4": Image.Resampling.LANCZOS,
}


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    resize_interpolation: Optional[str] = None,
) -> np.ndarray:
    """Resize one uint8 RGB/grayscale raster with the training geometry API.

    ``width`` and ``height`` are checked instead of silently trusting stale
    metadata.  PIL supplies every interpolation mode needed by V7, including
    BOX as the dependency-free equivalent of area downsampling.
    """

    pixels = np.asarray(image)
    width, height = int(width), int(height)
    resized_width, resized_height = int(resized_width), int(resized_height)
    if pixels.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image array, got {pixels.shape!r}.")
    if (pixels.shape[1], pixels.shape[0]) != (width, height):
        raise ValueError(
            "Image dimensions do not match resize metadata: "
            f"actual={(pixels.shape[1], pixels.shape[0])}, declared={(width, height)}."
        )
    if min(width, height, resized_width, resized_height) <= 0:
        raise ValueError("All resize dimensions must be positive integers.")

    if resize_interpolation is None:
        resize_interpolation = (
            "area" if width >= resized_width and height >= resized_height else "lanczos"
        )
    mode = str(resize_interpolation).lower()
    try:
        resample = _PIL_RESAMPLING[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported resize interpolation {resize_interpolation!r}; "
            f"expected one of {sorted(_PIL_RESAMPLING)}."
        ) from exc

    result = Image.fromarray(pixels).resize(
        (resized_width, resized_height), resample=resample
    )
    logger.debug("resize image using %s (PIL)", mode)
    return np.asarray(result).copy()
