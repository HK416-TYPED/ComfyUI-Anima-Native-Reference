"""Stable ComfyUI nodes for legacy V2 E180 and final competitive-router V4."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

import folder_paths

from .runtime import (
    AnimaNativeReferenceV2Runtime,
    AnimaNativeReferenceV4Runtime,
    get_or_create_runtime,
    get_or_create_v4_runtime,
)


PIPELINE_TYPE = "ANIMA_NATIVE_REF_V2_PIPELINE"
CATEGORY = "Anima/Native Reference V2"
V4_PIPELINE_TYPE = "ANIMA_NATIVE_REF_V4_PIPELINE"
V4_CATEGORY = "Anima/Native Reference V4"
V4_SLOT_OPTIONS = ("Image 1 (slot 0)", "Image 2 (slot 1)")
V4_PREPROCESS_OPTIONS = ("independent_reference", "match_output_edit")

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


def _v4_sampling_inputs(*, prompt: str) -> dict[str, Any]:
    return {
        "prompt": (
            "STRING",
            {
                "default": prompt,
                "multiline": True,
                "dynamicPrompts": True,
                "tooltip": (
                    "Must be byte-exact canonical text with an explicit clause for "
                    "every selected Image 1/Image 2 slot. Ambiguous, missing, or "
                    "truncated bindings fail before denoising."
                ),
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
            {
                "default": 30,
                "min": 1,
                "max": 200,
                "step": 1,
                "tooltip": "Validated final-V4 release default: 30 denoising steps.",
            },
        ),
        "cfg": (
            "FLOAT",
            {
                "default": 3.5,
                "min": 0.0,
                "max": 20.0,
                "step": 0.1,
                "round": 0.01,
                "tooltip": (
                    "Validated final-V4 release default is CFG 3.5. "
                    "At CFG != 1 the negative branch keeps negative ordinary text "
                    "conditioning but shares the positive raw-Qwen router clauses."
                ),
            },
        ),
        "flow_shift": (
            "FLOAT",
            {
                "default": 5.0,
                "min": 0.1,
                "max": 100.0,
                "step": 0.1,
                "round": 0.01,
            },
        ),
        "reference_scale": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 0.05,
                "round": 0.001,
                "tooltip": "Integrated native reference-route strength; formal default is 1.0.",
            },
        ),
        "reference_max_area": (
            "INT",
            {
                "default": 65536,
                "min": 256,
                "max": 4194304,
                "step": 256,
                "tooltip": (
                    "Area limit for independent references. match_output_edit uses "
                    "the requested output geometry instead."
                ),
            },
        ),
    }


def _comfy_progress_hooks(steps: int):
    progress_bar = None
    interrupt_callback = None
    try:
        import comfy.utils

        progress_bar = comfy.utils.ProgressBar(int(steps))
    except Exception:
        progress_bar = None
    try:
        import comfy.model_management as model_management

        interrupt_callback = (
            model_management.throw_exception_if_processing_interrupted
        )
    except Exception:
        interrupt_callback = None

    def report_progress(done: int, total: int) -> None:
        if progress_bar is not None:
            progress_bar.update_absolute(int(done), int(total))

    return report_progress, interrupt_callback


class AnimaNativeRefV4Loader:
    """Load the exact final 49k self-contained competitive-router model."""

    DESCRIPTION = (
        "Loads the final integrated V4 scaled-mixture checkpoint, the raw "
        "Qwen3-0.6B text encoder, and Qwen-Image VAE. The checkpoint contains "
        "all native reference/router weights; no LoRA or adapter sidecar is used."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "checkpoint": (
                    _filenames(CHECKPOINT_FOLDER),
                    {
                        "tooltip": (
                            "Exact final 49k integrated V4 file "
                            "anima-v4-scaled-mix-50-30-20-e2-step49000-"
                            "256area.safetensors in models/diffusion_models "
                            "(SHA256 starts 4500a4aa)."
                        )
                    },
                ),
                "text_encoder": (
                    _filenames(TEXT_ENCODER_FOLDER),
                    {
                        "tooltip": (
                            "qwen_3_06b_base.safetensors. Raw Qwen states feed both "
                            "the native LLM adapter and the V4 text-slot router."
                        )
                    },
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
                            "balanced stages Qwen/VAE; high_vram keeps all models on "
                            "GPU; text_encoder_cpu is the lowest-VRAM fallback."
                        ),
                    },
                ),
            },
            "optional": {
                "verify_release_sha256": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Read and verify exact SHA256 for the final checkpoint, "
                            "Qwen encoder, and VAE once while loading."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (V4_PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = V4_CATEGORY

    @classmethod
    def IS_CHANGED(
        cls,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        offload_mode: str,
        verify_release_sha256: bool = False,
    ) -> tuple[Any, ...]:
        return (
            "v4-final49k",
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
    ) -> tuple[AnimaNativeReferenceV4Runtime]:
        pipeline = get_or_create_v4_runtime(
            _resolve_model(CHECKPOINT_FOLDER, checkpoint),
            _resolve_model(TEXT_ENCODER_FOLDER, text_encoder),
            _resolve_model(VAE_FOLDER, vae),
            offload_mode=offload_mode,
            attn_mode="torch",
            vae_chunk_size=64,
            vae_disable_cache=True,
            prompt_cache_entries=64,
            reference_cache_entries=16,
            verify_checkpoint_sha256=bool(verify_release_sha256),
        )
        return (pipeline,)


class AnimaNativeRefV4Generate1Ref:
    """Generate from one real reference assigned to logical slot 0 or slot 1."""

    DESCRIPTION = (
        "Pure-noise one-reference generation with an explicit logical slot. "
        "Only one VAE latent is carried—no duplicate/fake second image. Use "
        "independent_reference for character/reference generation or "
        "match_output_edit for one-image edit geometry."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required: dict[str, Any] = {
            "pipeline": (V4_PIPELINE_TYPE,),
            "reference_image": (
                "IMAGE",
                {
                    "tooltip": (
                        "Exactly one Comfy IMAGE (B=1). It is physically encoded "
                        "once and assigned to the selected logical slot."
                    )
                },
            ),
            "logical_slot": (
                list(V4_SLOT_OPTIONS),
                {
                    "default": V4_SLOT_OPTIONS[0],
                    "tooltip": (
                        "The prompt must explicitly mention this same Image 1 or "
                        "Image 2 index."
                    ),
                },
            ),
            "preprocess_mode": (
                list(V4_PREPROCESS_OPTIONS),
                {
                    "default": "independent_reference",
                    "tooltip": (
                        "independent_reference preserves the reference aspect ratio "
                        "under the area limit; match_output_edit center-fits it to "
                        "the requested output width/height."
                    ),
                },
            ),
        }
        required.update(
            _v4_sampling_inputs(
                prompt=(
                    "Use Image 1 as the character identity reference. "
                    "Generate a new illustration."
                )
            )
        )
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = V4_CATEGORY

    def generate(
        self,
        pipeline: AnimaNativeReferenceV4Runtime,
        reference_image: torch.Tensor,
        logical_slot: str,
        preprocess_mode: str,
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
        if not isinstance(pipeline, AnimaNativeReferenceV4Runtime):
            raise TypeError(
                "pipeline must come from Anima Reference V4 Loader; received "
                f"{type(pipeline).__name__}."
            )
        try:
            slot_id = V4_SLOT_OPTIONS.index(str(logical_slot))
        except ValueError as exc:
            raise ValueError(
                f"logical_slot must be one of {V4_SLOT_OPTIONS!r}; got {logical_slot!r}."
            ) from exc
        report_progress, interrupt_callback = _comfy_progress_hooks(int(steps))
        image = pipeline.generate(
            [reference_image],
            [slot_id],
            prompt,
            negative_prompt=negative_prompt,
            preprocess_mode=preprocess_mode,
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


class AnimaNativeRefV4Generate2Refs:
    """Generate from Image 1/slot 0 and Image 2/slot 1."""

    DESCRIPTION = (
        "Pure-noise dual-reference generation. Reference Image 1 is logical "
        "slot 0 and Reference Image 2 is logical slot 1. Neither has a fixed "
        "semantic role: separate canonical prompt clauses specify what to use "
        "from each image. Both references use independent preprocessing."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required: dict[str, Any] = {
            "pipeline": (V4_PIPELINE_TYPE,),
            "reference_image_1": (
                "IMAGE",
                {"tooltip": "Exactly one physical image for Image 1 / slot 0."},
            ),
            "reference_image_2": (
                "IMAGE",
                {"tooltip": "Exactly one physical image for Image 2 / slot 1."},
            ),
        }
        required.update(
            _v4_sampling_inputs(
                prompt=(
                    "Use the pose and composition from Image 1; "
                    "use the character appearance from Image 2."
                )
            )
        )
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = V4_CATEGORY

    def generate(
        self,
        pipeline: AnimaNativeReferenceV4Runtime,
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
        if not isinstance(pipeline, AnimaNativeReferenceV4Runtime):
            raise TypeError(
                "pipeline must come from Anima Reference V4 Loader; received "
                f"{type(pipeline).__name__}."
            )
        report_progress, interrupt_callback = _comfy_progress_hooks(int(steps))
        image = pipeline.generate(
            [reference_image_1, reference_image_2],
            [0, 1],
            prompt,
            negative_prompt=negative_prompt,
            preprocess_mode="independent_reference",
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
    "AnimaNativeRefV4Loader": AnimaNativeRefV4Loader,
    "AnimaNativeRefV4Generate1Ref": AnimaNativeRefV4Generate1Ref,
    "AnimaNativeRefV4Generate2Refs": AnimaNativeRefV4Generate2Refs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaNativeRefV2Loader": "Anima Reference V2 Loader (E180)",
    "AnimaNativeRefV2Generate": "Anima Reference V2 Generate (2 Refs)",
    "AnimaNativeRefV4Loader": "Anima Reference V4 Loader (Final 49k)",
    "AnimaNativeRefV4Generate1Ref": "Anima Reference V4 Generate (1 Ref)",
    "AnimaNativeRefV4Generate2Refs": "Anima Reference V4 Generate (2 Refs)",
}


__all__ = [
    "AnimaNativeRefV2Generate",
    "AnimaNativeRefV2Loader",
    "AnimaNativeRefV4Generate1Ref",
    "AnimaNativeRefV4Generate2Refs",
    "AnimaNativeRefV4Loader",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
