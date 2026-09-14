"""ComfyUI nodes for the V7 exact-prompt native-context checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

import folder_paths

from .runtime import (
    AnimaNativeContextV7Runtime,
    V7_DEFAULT_HEIGHT,
    V7_DEFAULT_REFERENCE_MAX_AREA,
    V7_DEFAULT_WIDTH,
    V7_EDIT_SCALE,
    V7_REFERENCE_SCALE,
    get_or_create_v7_runtime,
)


PIPELINE_TYPE = "ANIMA_NATIVE_CONTEXT_V7_PIPELINE"
CATEGORY = "Anima/Native Context V7"
CHECKPOINT_FOLDER = "diffusion_models"
TEXT_ENCODER_FOLDER = "text_encoders"
VAE_FOLDER = "vae"


def _filenames(folder_name: str) -> list[str]:
    names = folder_paths.get_filename_list(folder_name)
    return names if names else [f"<no files in models/{folder_name}>"]


def _resolve_model(folder_name: str, filename: str) -> str:
    if filename.startswith("<no files in models/"):
        raise FileNotFoundError(
            f"No model was found in models/{folder_name}. See this node's README."
        )
    return folder_paths.get_full_path_or_raise(folder_name, filename)


def _stat_fingerprint(folder_name: str, filename: str) -> tuple[str, int, int]:
    path = Path(_resolve_model(folder_name, filename)).resolve(strict=True)
    stat = path.stat()
    return os.path.normcase(str(path)), int(stat.st_size), int(stat.st_mtime_ns)


def _comfy_progress_hooks(steps: int):
    progress_bar = None
    interrupt_callback = None
    try:
        import comfy.utils

        progress_bar = comfy.utils.ProgressBar(int(steps))
    except Exception:
        pass
    try:
        import comfy.model_management as model_management

        interrupt_callback = model_management.throw_exception_if_processing_interrupted
    except Exception:
        pass

    def report_progress(done: int, total: int) -> None:
        if progress_bar is not None:
            progress_bar.update_absolute(int(done), int(total))

    return report_progress, interrupt_callback


def _sampling_inputs(default_prompt: str) -> dict[str, Any]:
    return {
        "prompt": (
            "STRING",
            {
                "default": default_prompt,
                "multiline": True,
                "dynamicPrompts": True,
                "tooltip": (
                    "Passed byte-for-byte to the normal Anima text path. The node does "
                    "not add image labels, role clauses, or hidden instructions."
                ),
            },
        ),
        "negative_prompt": (
            "STRING",
            {"default": "", "multiline": True, "dynamicPrompts": True},
        ),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        "width": ("INT", {"default": V7_DEFAULT_WIDTH, "min": 64, "max": 4096, "step": 16}),
        "height": ("INT", {"default": V7_DEFAULT_HEIGHT, "min": 64, "max": 4096, "step": 16}),
        "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
        "cfg": (
            "FLOAT",
            {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01},
        ),
        "flow_shift": (
            "FLOAT",
            {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": 0.01},
        ),
    }


def _reference_controls(default_scale: float, *, edit: bool = False) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "reference_scale": (
            "FLOAT",
            {
                "default": float(default_scale),
                "min": 0.0,
                "max": 4.0,
                "step": 0.05,
                "round": 0.001,
            },
        )
    }
    if not edit:
        controls["reference_max_area"] = (
            "INT",
            {"default": V7_DEFAULT_REFERENCE_MAX_AREA, "min": 256, "max": 4194304, "step": 256},
        )
    return controls


def _require_pipeline(pipeline: Any) -> AnimaNativeContextV7Runtime:
    if not isinstance(pipeline, AnimaNativeContextV7Runtime):
        raise TypeError("pipeline must come from Anima Native Context V7 Loader.")
    return pipeline


class AnimaNativeContextV7Loader:
    """Load a self-contained V7 checkpoint and the unchanged Anima auxiliaries."""

    DESCRIPTION = (
        "Loads native_context_v1. It rejects old attribute/index/pointer routers, "
        "external LoRAs, runtime masks, and prompt-rewrite checkpoints."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "checkpoint": (_filenames(CHECKPOINT_FOLDER),),
                "text_encoder": (_filenames(TEXT_ENCODER_FOLDER),),
                "vae": (_filenames(VAE_FOLDER),),
                "offload_mode": (
                    ["balanced", "high_vram", "text_encoder_cpu"],
                    {"default": "balanced"},
                ),
            },
            "optional": {
                "verify_file_sha256": ("BOOLEAN", {"default": False}),
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
        verify_file_sha256: bool = False,
    ) -> tuple[Any, ...]:
        return (
            "native-context-v7",
            _stat_fingerprint(CHECKPOINT_FOLDER, checkpoint),
            _stat_fingerprint(TEXT_ENCODER_FOLDER, text_encoder),
            _stat_fingerprint(VAE_FOLDER, vae),
            str(offload_mode),
            bool(verify_file_sha256),
        )

    def load(
        self,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        offload_mode: str,
        verify_file_sha256: bool = False,
    ) -> tuple[AnimaNativeContextV7Runtime]:
        runtime = get_or_create_v7_runtime(
            _resolve_model(CHECKPOINT_FOLDER, checkpoint),
            _resolve_model(TEXT_ENCODER_FOLDER, text_encoder),
            _resolve_model(VAE_FOLDER, vae),
            offload_mode=str(offload_mode),
            attn_mode="torch",
            vae_chunk_size=0,
            vae_disable_cache=False,
            prompt_cache_entries=64,
            reference_cache_entries=16,
            verify_checkpoint_sha256=bool(verify_file_sha256),
        )
        return (runtime,)


class AnimaNativeContextV7T2I:
    DESCRIPTION = "True T2I: no reference latent, no empty-image substitute, no visual route."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required = {"pipeline": (PIPELINE_TYPE,)}
        required.update(_sampling_inputs(""))
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, pipeline, prompt, negative_prompt, seed, width, height, steps, cfg, flow_shift):
        runtime = _require_pipeline(pipeline)
        progress, interrupt = _comfy_progress_hooks(int(steps))
        return (
            runtime.generate_t2i(
                prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(cfg),
                flow_shift=float(flow_shift),
                progress_callback=progress,
                interrupt_callback=interrupt,
            ),
        )


class AnimaNativeContextV7SingleReference:
    DESCRIPTION = (
        "One physical reference is always structural instance 0. There is no slot selector "
        "and no prompt text is generated by the node."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "reference_image": ("IMAGE", {"tooltip": "One reference image, B=1."}),
        }
        required.update(_sampling_inputs(""))
        required.update(_reference_controls(V7_REFERENCE_SCALE))
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self, pipeline, reference_image, prompt, negative_prompt, seed, width, height,
        steps, cfg, flow_shift, reference_scale, reference_max_area,
    ):
        runtime = _require_pipeline(pipeline)
        progress, interrupt = _comfy_progress_hooks(int(steps))
        return (
            runtime.generate(
                [reference_image],
                [0],
                prompt,
                task_mode="reference",
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(cfg),
                flow_shift=float(flow_shift),
                native_reference_scale=float(reference_scale),
                reference_max_area=int(reference_max_area),
                progress_callback=progress,
                interrupt_callback=interrupt,
            ),
        )


class AnimaNativeContextV7MultiReference:
    DESCRIPTION = (
        "Two images retain physical order as structural instances 0 and 1. Roles are not "
        "hard-coded; the exact user prompt and visual candidates determine routing."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "reference_image_1": ("IMAGE",),
            "reference_image_2": ("IMAGE",),
        }
        required.update(_sampling_inputs(""))
        required.update(_reference_controls(V7_REFERENCE_SCALE))
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self, pipeline, reference_image_1, reference_image_2, prompt, negative_prompt,
        seed, width, height, steps, cfg, flow_shift, reference_scale, reference_max_area,
    ):
        runtime = _require_pipeline(pipeline)
        progress, interrupt = _comfy_progress_hooks(int(steps))
        return (
            runtime.generate(
                [reference_image_1, reference_image_2],
                [0, 1],
                prompt,
                task_mode="reference",
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(cfg),
                flow_shift=float(flow_shift),
                native_reference_scale=float(reference_scale),
                reference_max_area=int(reference_max_area),
                progress_callback=progress,
                interrupt_callback=interrupt,
            ),
        )


class AnimaNativeContextV7Edit:
    DESCRIPTION = (
        "Maskless edit. The sole source is structurally marked aligned_source in instance 0; "
        "the node never adds source/image wording to the prompt."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "source_image": ("IMAGE", {"tooltip": "Editable source image, B=1."}),
        }
        required.update(_sampling_inputs(""))
        required.update(_reference_controls(V7_EDIT_SCALE, edit=True))
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self, pipeline, source_image, prompt, negative_prompt, seed, width, height,
        steps, cfg, flow_shift, reference_scale,
    ):
        runtime = _require_pipeline(pipeline)
        progress, interrupt = _comfy_progress_hooks(int(steps))
        return (
            runtime.generate(
                [source_image],
                [0],
                prompt,
                task_mode="edit",
                aligned_source_slot_id=0,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(cfg),
                flow_shift=float(flow_shift),
                native_reference_scale=float(reference_scale),
                reference_max_area=V7_DEFAULT_REFERENCE_MAX_AREA,
                progress_callback=progress,
                interrupt_callback=interrupt,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "AnimaNativeContextV7Loader": AnimaNativeContextV7Loader,
    "AnimaNativeContextV7T2I": AnimaNativeContextV7T2I,
    "AnimaNativeContextV7SingleReference": AnimaNativeContextV7SingleReference,
    "AnimaNativeContextV7MultiReference": AnimaNativeContextV7MultiReference,
    "AnimaNativeContextV7Edit": AnimaNativeContextV7Edit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaNativeContextV7Loader": "Anima Native Context V7 Loader",
    "AnimaNativeContextV7T2I": "Anima Native Context V7 T2I (No Images)",
    "AnimaNativeContextV7SingleReference": "Anima Native Context V7 (1 Reference)",
    "AnimaNativeContextV7MultiReference": "Anima Native Context V7 (2 References)",
    "AnimaNativeContextV7Edit": "Anima Native Context V7 Edit (No Mask)",
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
