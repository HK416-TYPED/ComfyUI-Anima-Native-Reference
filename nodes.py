"""Stable ComfyUI V1 node schema for Anima Native Reference V2 E180."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

import folder_paths

from .runtime import AnimaNativeReferenceV2Runtime, get_or_create_runtime


PIPELINE_TYPE = "ANIMA_NATIVE_REF_V2_PIPELINE"
CATEGORY = "Anima/Native Reference V2"

# Reuse ComfyUI's standard model directories.  The E180 file is a diffusion
# transformer, not a stock all-in-one checkpoint; it therefore belongs under
# models/diffusion_models rather than models/checkpoints.
CHECKPOINT_FOLDER = "diffusion_models"
TEXT_ENCODER_FOLDER = "text_encoders"
VAE_FOLDER = "vae"


def _filenames(folder_name: str) -> list[str]:
    names = folder_paths.get_filename_list(folder_name)
    # Comfy requires a non-empty COMBO.  The sentinel gives a clear loader
    # error while still allowing the node to be created before models arrive.
    return names if names else [f"<no files in models/{folder_name}>"]


def _resolve_model(folder_name: str, filename: str) -> str:
    if filename.startswith("<no files in models/"):
        raise FileNotFoundError(
            f"No model was found for {folder_name!r}. See this custom node's README for the required files."
        )
    # get_full_path_or_raise rejects missing files and prevents arbitrary paths.
    return folder_paths.get_full_path_or_raise(folder_name, filename)


def _stat_fingerprint(folder_name: str, filename: str) -> tuple[str, int, int]:
    path = Path(_resolve_model(folder_name, filename)).resolve(strict=True)
    stat = path.stat()
    return (os.path.normcase(str(path)), int(stat.st_size), int(stat.st_mtime_ns))


class AnimaNativeRefV2Loader:
    """Load and cache the one-file integrated E180 reference model runtime."""

    DESCRIPTION = (
        "Loads the integrated Anima Native Reference V2 E180 checkpoint, the "
        "Anima Qwen3-0.6B text encoder, and the Qwen-Image VAE. No external "
        "LoRA or reference adapter is loaded."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "checkpoint": (
                    _filenames(CHECKPOINT_FOLDER),
                    {"tooltip": "E180 integrated V2 .safetensors in models/diffusion_models."},
                ),
                "text_encoder": (
                    _filenames(TEXT_ENCODER_FOLDER),
                    {"tooltip": "qwen_3_06b_base.safetensors in models/text_encoders."},
                ),
                "vae": (
                    _filenames(VAE_FOLDER),
                    {"tooltip": "qwen_image_vae.safetensors in models/vae."},
                ),
                "offload_mode": (
                    ["balanced", "high_vram", "text_encoder_cpu"],
                    {
                        "default": "balanced",
                        "tooltip": (
                            "balanced keeps DiT on GPU and stages Qwen/VAE; high_vram keeps all models on GPU; "
                            "text_encoder_cpu is the slowest low-VRAM fallback."
                        ),
                    },
                ),
            },
            "optional": {
                "verify_release_sha256": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Read and verify SHA256 for all three release weight files once while loading.",
                    },
                ),
            },
        }

    RETURN_TYPES = (PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(
        cls,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        offload_mode: str,
        verify_release_sha256: bool = False,
    ) -> tuple[Any, ...]:
        """Invalidate Comfy's loader cache if an on-disk model is replaced."""

        return (
            _stat_fingerprint(CHECKPOINT_FOLDER, checkpoint),
            _stat_fingerprint(TEXT_ENCODER_FOLDER, text_encoder),
            _stat_fingerprint(VAE_FOLDER, vae),
            str(offload_mode),
            bool(verify_release_sha256),
        )

    def load(
        self,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        offload_mode: str,
        verify_release_sha256: bool = False,
    ) -> tuple[AnimaNativeReferenceV2Runtime]:
        checkpoint_path = _resolve_model(CHECKPOINT_FOLDER, checkpoint)
        text_encoder_path = _resolve_model(TEXT_ENCODER_FOLDER, text_encoder)
        vae_path = _resolve_model(VAE_FOLDER, vae)

        pipeline = get_or_create_runtime(
            checkpoint_path,
            text_encoder_path,
            vae_path,
            offload_mode=offload_mode,
            attn_mode="torch",
            vae_chunk_size=64,
            vae_disable_cache=True,
            prompt_cache_entries=64,
            reference_cache_entries=16,
            verify_checkpoint_sha256=bool(verify_release_sha256),
        )
        return (pipeline,)


class AnimaNativeRefV2Generate:
    """Generate one image from two ordered reference-image slots and a prompt."""

    DESCRIPTION = (
        "Native two-reference generation from pure noise. Reference Image 1 is ordered slot 0 and Reference "
        "Image 2 is ordered slot 1; neither slot has a hard-coded scene/identity/style role. Prompt text defines "
        "the requested use of both references. This is not img2img, so there is intentionally no denoise-strength input."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pipeline": (PIPELINE_TYPE,),
                "reference_image_1": (
                    "IMAGE",
                    {"tooltip": "Ordered reference slot 0. The initial release requires exactly one image (B=1)."},
                ),
                "reference_image_2": (
                    "IMAGE",
                    {"tooltip": "Ordered reference slot 1. The initial release requires exactly one image (B=1)."},
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Create a new anime illustration using both reference images, following their visual information and the requested composition.",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "width": (
                    "INT",
                    {"default": 256, "min": 64, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 256, "min": 64, "max": 4096, "step": 16},
                ),
                "steps": (
                    "INT",
                    {"default": 40, "min": 1, "max": 200, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01},
                ),
                "flow_shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "reference_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "round": 0.001,
                        "tooltip": "Native V2 reference-route strength. Formal E180 evaluation used 1.0.",
                    },
                ),
                "reference_max_area": (
                    "INT",
                    {
                        "default": 65536,
                        "min": 256,
                        "max": 4194304,
                        "step": 256,
                        "tooltip": "Each reference is downscaled only if its pixel area exceeds this limit.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        pipeline: AnimaNativeReferenceV2Runtime,
        reference_image_1: torch.Tensor,
        reference_image_2: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        flow_shift: float,
        reference_scale: float,
        reference_max_area: int,
    ) -> tuple[torch.Tensor]:
        if not isinstance(pipeline, AnimaNativeReferenceV2Runtime):
            raise TypeError(
                "pipeline must come from Anima Reference V2 Loader; received "
                f"{type(pipeline).__name__}."
            )

        progress_bar = None
        interrupt_callback = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(steps))
        except Exception:
            progress_bar = None
        try:
            import comfy.model_management as model_management

            interrupt_callback = model_management.throw_exception_if_processing_interrupted
        except Exception:
            interrupt_callback = None

        def report_progress(done: int, total: int) -> None:
            if progress_bar is not None:
                progress_bar.update_absolute(int(done), int(total))

        image = pipeline.generate(
            reference_image_1,
            reference_image_2,
            prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            seed=int(seed),
            steps=int(steps),
            guidance_scale=float(cfg),
            flow_shift=float(flow_shift),
            native_reference_scale=float(reference_scale),
            reference_max_area=int(reference_max_area),
            progress_callback=report_progress,
            interrupt_callback=interrupt_callback,
        )
        return (image,)


NODE_CLASS_MAPPINGS = {
    "AnimaNativeRefV2Loader": AnimaNativeRefV2Loader,
    "AnimaNativeRefV2Generate": AnimaNativeRefV2Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaNativeRefV2Loader": "Anima Reference V2 Loader (E180)",
    "AnimaNativeRefV2Generate": "Anima Reference V2 Generate (2 Refs)",
}


__all__ = [
    "AnimaNativeRefV2Generate",
    "AnimaNativeRefV2Loader",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
