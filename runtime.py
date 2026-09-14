"""Fail-closed ComfyUI runtime for integrated Anima Native Context V7.

The user prompt is tokenized exactly as supplied. This runtime never appends an
``Image 1``/``Image 2`` clause, never parses prompt clauses, and never constructs
reference masks. Physical reference order is compacted to structural instance
indices ``[0]`` or ``[0, 1]``. Edit source identity is a separate non-text role
bit passed through ``aligned_source_slot_ids``. T2I supplies no visual sequence.

Only a self-contained ``native_context_v1`` checkpoint is accepted. Legacy
attribute routers, index conditioners, pointer heads, external LoRAs, masks, and
adapter sidecars fail closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Optional, Sequence

from PIL import Image
from safetensors import safe_open
import torch

from . import _runtime_common as base


V7_DEFAULT_STEPS = 30
V7_DEFAULT_GUIDANCE_SCALE = 3.5
V7_DEFAULT_WIDTH = base.DEFAULT_WIDTH
V7_DEFAULT_HEIGHT = base.DEFAULT_HEIGHT
V7_DEFAULT_REFERENCE_MAX_AREA = base.DEFAULT_REFERENCE_MAX_AREA
V7_REFERENCE_SCALE = 1.0
V7_EDIT_SCALE = 1.4
V7_TASK_MODES = frozenset({"reference", "edit"})
V7_EXPECTED_TOTAL_TENSORS = 1569
V7_EXPECTED_NATIVE_ATTN_TENSORS = 868

V7_EXPECTED_CONFIG: Mapping[str, Any] = {
    "architecture": "v2",
    "context_alpha": 1.0,
    "context_index_dim": 64,
    "context_routing_version": "native_reference_context_v1",
    "gate_dim": 4,
    "initial_gate": 0.5,
    "max_reference_images": 2,
    "rank": 64,
    "router_dim": 64,
    "router_null_enabled": True,
    "router_temperature": 1.0,
    "routing_alpha": 1.0,
    "routing_mode": "native_context_v1",
    "scope_alpha": 1.0,
    "scope_hidden_dim": 32,
    "scope_initial_change_probability": 0.95,
    "scope_version": "implicit_source_scope_v1",
}

V7_REQUIRED_METADATA: Mapping[str, str] = {
    "anima_checkpoint_layout": "integrated_single_checkpoint",
    "anima_external_adapter_required": "false",
    "anima_native_reference_conditioning": "true",
    "anima_native_reference_version": "4",
    "anima_native_reference_routing_mode": "native_context_v1",
    "anima_prompt_contract": "exact_user_prompt_no_hidden_reference_rewrite",
    "anima_reference_slot_semantics": "structural_image_instance_not_semantic_role",
    "anima_reference_routing_inputs": "full_prompt+visual_candidate+target+instance+source_role",
}

V7_REQUIRED_NATIVE_SHAPES: Mapping[str, tuple[int, ...]] = {
    "native_reference_context_conditioner.instance_conditioner.router_proj.weight": (64, 64),
    "native_reference_context_conditioner.instance_conditioner.visual_proj.weight": (2048, 64),
    "native_reference_context_conditioner.role_router_proj.weight": (64, 2),
    "native_reference_context_conditioner.role_visual_proj.weight": (2048, 2),
    "edit_scope_predictor.auxiliary_proj.weight": (32, 2048),
    "edit_scope_predictor.output.bias": (1,),
    "edit_scope_predictor.output.weight": (1, 32),
    "edit_scope_predictor.prompt_proj.weight": (32, 1024),
    "edit_scope_predictor.source_clause_proj.weight": (32, 1024),
    "edit_scope_predictor.source_proj.weight": (32, 2048),
    "edit_scope_predictor.timestep_proj.weight": (32, 2048),
    "reference_slot_embeddings.weight": (2, 2048),
    "reference_type_embedding": (2048,),
}

_RETIRED_PREFIXES = (
    "native_reference_attribute_router.",
    "native_reference_index_conditioner.",
)


@dataclass(frozen=True)
class V7CheckpointInfo:
    fingerprint: base.FileFingerprint
    metadata: Mapping[str, str]
    config: Mapping[str, Any]
    tensor_count: int
    native_tensor_count: int
    dtype_counts: Mapping[str, int]
    sha256: Optional[str] = None


def _is_pointer_key(key: str) -> bool:
    key = base._normalise_state_key(key)
    return "pointer_head" in key or "reference_pointer" in key


def validate_v7_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    verify_sha256: bool = False,
) -> V7CheckpointInfo:
    """Validate the architecture and prompt contract, not one training step."""

    fingerprint = base.FileFingerprint.from_path(checkpoint_path)
    path = Path(fingerprint.path)
    if path.suffix.lower() != ".safetensors":
        raise base.CheckpointValidationError(f"V7 model must be .safetensors: {path}")

    dtype_counts: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    native_attn_count = 0
    retired: list[str] = []
    external: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        if len(keys) != V7_EXPECTED_TOTAL_TENSORS:
            raise base.CheckpointValidationError(
                f"V7 tensor count mismatch: {len(keys)} != {V7_EXPECTED_TOTAL_TENSORS}."
            )
        for raw_key in keys:
            key = base._normalise_state_key(raw_key)
            if base._is_external_lora_key(raw_key):
                external.append(raw_key)
            if key.startswith(_RETIRED_PREFIXES) or _is_pointer_key(raw_key):
                retired.append(raw_key)
            tensor_slice = handle.get_slice(raw_key)
            dtype_name = str(tensor_slice.get_dtype())
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            if dtype_name != "BF16":
                raise base.CheckpointValidationError(
                    f"V7 contains non-BF16 tensor {raw_key!r}: {dtype_name}."
                )
            if ".native_reference_attn." in key:
                native_attn_count += 1
            if key in V7_REQUIRED_NATIVE_SHAPES:
                shapes[key] = tuple(int(value) for value in tensor_slice.get_shape())

    if external:
        raise base.CheckpointValidationError(
            f"External LoRA tensors are forbidden; first keys: {external[:5]!r}."
        )
    if retired:
        raise base.CheckpointValidationError(
            "Retired parser/attribute/index/pointer state is forbidden; "
            f"first keys: {retired[:5]!r}."
        )
    if native_attn_count != V7_EXPECTED_NATIVE_ATTN_TENSORS:
        raise base.CheckpointValidationError(
            "V7 read-only reference-attention tensor count mismatch: "
            f"{native_attn_count} != {V7_EXPECTED_NATIVE_ATTN_TENSORS}."
        )
    missing = sorted(set(V7_REQUIRED_NATIVE_SHAPES) - set(shapes))
    if missing:
        raise base.CheckpointValidationError(f"V7 native state is incomplete: {missing!r}.")
    for key, expected in V7_REQUIRED_NATIVE_SHAPES.items():
        if shapes[key] != expected:
            raise base.CheckpointValidationError(
                f"V7 shape mismatch for {key!r}: expected {expected}, got {shapes[key]}."
            )
    for key, expected in V7_REQUIRED_METADATA.items():
        base._expect_metadata(metadata, key, expected)

    # The training saver can omit three redundant booleans. Keep validation
    # fail-closed: an explicit conflict always fails, and a missing key is
    # accepted only when its stricter equivalent contract is present.
    compatibility_aliases = (
        ("runtime_prompt_rewrite", "anima_prompt_contract", "exact_user_prompt_no_hidden_reference_rewrite"),
        ("runtime_mask_input", "anima_edit_interface", "source_refs_prompt_no_mask"),
        ("semantic_slot_roles", "anima_reference_slot_semantics", "structural_image_instance_not_semantic_role"),
    )
    for key, equivalent_key, equivalent_value in compatibility_aliases:
        actual = metadata.get(key)
        if actual is not None:
            if actual != "false":
                raise base.CheckpointValidationError(
                    f"Checkpoint metadata mismatch for {key!r}: expected 'false', got {actual!r}."
                )
        else:
            base._expect_metadata(metadata, equivalent_key, equivalent_value)
    try:
        config = json.loads(metadata["anima_native_reference_config"])
    except Exception as exc:
        raise base.CheckpointValidationError("Invalid V7 native-reference config JSON.") from exc
    if config != dict(V7_EXPECTED_CONFIG):
        raise base.CheckpointValidationError(
            f"V7 native-reference config mismatch: expected {dict(V7_EXPECTED_CONFIG)!r}, got {config!r}."
        )

    checksum = base._sha256_file(path) if verify_sha256 else None
    return V7CheckpointInfo(
        fingerprint=fingerprint,
        metadata=metadata,
        config=config,
        tensor_count=V7_EXPECTED_TOTAL_TENSORS,
        native_tensor_count=native_attn_count + len(V7_REQUIRED_NATIVE_SHAPES),
        dtype_counts=dtype_counts,
        sha256=checksum,
    )


class AnimaNativeContextV7Runtime(base.NativeContextRuntimePrimitives):
    """Native-context V7 runtime for T2I, reference generation, and Edit."""

    def __init__(
        self,
        checkpoint_path: os.PathLike[str] | str,
        text_encoder_path: os.PathLike[str] | str,
        vae_path: os.PathLike[str] | str,
        *,
        runtime_root: os.PathLike[str] | str | None = None,
        device: str | torch.device | None = None,
        offload_mode: str = "balanced",
        attn_mode: str = "torch",
        vae_chunk_size: int = 0,
        vae_disable_cache: bool = False,
        prompt_cache_entries: int = 64,
        reference_cache_entries: int = 16,
        verify_checkpoint_sha256: bool = False,
    ) -> None:
        if offload_mode not in self.SUPPORTED_OFFLOAD_MODES:
            raise ValueError(f"unsupported offload_mode {offload_mode!r}")
        if attn_mode != "torch":
            raise ValueError("V7 publication runtime supports attn_mode='torch' only.")
        if int(vae_chunk_size) < 0:
            raise ValueError("vae_chunk_size must be zero (exact unchunked mode) or positive")

        self.lock = threading.RLock()
        self.checkpoint_info = validate_v7_checkpoint(
            checkpoint_path, verify_sha256=verify_checkpoint_sha256
        )
        self.checkpoint_path = self.checkpoint_info.fingerprint.path
        self.text_encoder_fingerprint = base.validate_qwen_text_encoder(
            text_encoder_path, verify_sha256=verify_checkpoint_sha256
        )
        self.vae_fingerprint = base.validate_qwen_image_vae(
            vae_path, verify_sha256=verify_checkpoint_sha256
        )
        self._release_sha256_verified = bool(verify_checkpoint_sha256)
        self.text_encoder_path = self.text_encoder_fingerprint.path
        self.vae_path = self.vae_fingerprint.path
        self.runtime_root = str(
            Path(runtime_root or base.default_runtime_root()).expanduser().resolve(strict=True)
        )
        self.device = base._resolve_device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise base.AnimaRuntimeError(f"V7 runtime requires a CUDA BF16 device; got {self.device}.")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise base.AnimaRuntimeError(f"CUDA device {self.device} does not report BF16 support.")

        self.offload_mode = offload_mode
        self.attention_mode = attn_mode
        self.vae_chunk_size = int(vae_chunk_size)
        self.vae_disable_cache = bool(vae_disable_cache)
        self.dtype = torch.bfloat16
        self._prompt_cache = base._BoundedLRU(prompt_cache_entries)
        self._reference_cache = base._BoundedLRU(reference_cache_entries)
        self._prompt_hits = self._prompt_misses = 0
        self._reference_hits = self._reference_misses = 0
        self._model_loads = 0
        self._closed = False
        self._modules = base.load_runtime_modules(self.runtime_root)
        self._anima = None
        self._vae = None
        self._text_encoder = None
        self._qwen_tokenizer = None
        self._t5_tokenizer = None
        self._tokenize_strategy = None
        self._load_models_once()

    def _load_models_once(self) -> None:
        with self.lock:
            if self._closed:
                raise base.AnimaRuntimeError("Runtime is closed.")
            if self._anima is not None:
                return
            self._anima = self._modules.anima_utils.load_anima_model(
                self.device,
                self.checkpoint_path,
                self.attention_mode,
                True,
                self.device,
                self.dtype,
                False,
                lora_weights_list=None,
                lora_multipliers=None,
                enable_ip_adapter=False,
                enable_native_reference_conditioning=None,
                native_reference_scale=V7_REFERENCE_SCALE,
            )
            self._anima.to(self.device, dtype=self.dtype)
            self._anima.eval().requires_grad_(False)
            checks = {
                "native reference conditioning": bool(
                    getattr(self._anima, "native_reference_conditioning_enabled", False)
                ),
                "native context routing mode": (
                    getattr(self._anima, "native_reference_routing_mode", None)
                    == "native_context_v1"
                ),
                "structural context conditioner": (
                    getattr(self._anima, "native_reference_context_conditioner", None)
                    is not None
                ),
                "retired attribute router absent": (
                    getattr(self._anima, "native_reference_attribute_router", None)
                    is None
                ),
                "retired index conditioner absent": (
                    getattr(self._anima, "native_reference_index_conditioner", None)
                    is None
                ),
                "implicit edit scope": (
                    getattr(self._anima, "edit_scope_predictor", None) is not None
                ),
                "native LLM adapter": bool(getattr(self._anima, "use_llm_adapter", False)),
            }
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                raise base.AnimaRuntimeError(
                    f"Loaded V7 model violates the native-context graph contract: {failed!r}."
                )
            if getattr(self._anima, "native_reference_architecture", None) != "v2":
                raise base.AnimaRuntimeError("Loaded V7 model has the wrong native carrier architecture.")
            if int(getattr(self._anima, "native_reference_max_images", -1)) != 2:
                raise base.AnimaRuntimeError("Loaded V7 model must expose exactly two logical slots.")
            setter = getattr(self._anima, "set_native_reference_fixed12ref_vectorized", None)
            if setter is None:
                raise base.AnimaRuntimeError("Vendored V7 model lacks fixed-one/two carrier control.")
            setter(True)
            if self._modules.anima_text_conditioning.reference_router_requires_clause_masks(
                self._anima, [[0, 1]], use_reference_sequence=True
            ):
                raise base.AnimaRuntimeError(
                    "V7 model still exposes the retired clause-mask route."
                )

            text_device: torch.device | str = self.device if self.offload_mode == "high_vram" else "cpu"
            self._text_encoder, self._qwen_tokenizer = self._modules.anima_utils.load_qwen3_text_encoder(
                self.text_encoder_path,
                dtype=self.dtype,
                device=text_device,
                lora_weights=None,
                lora_multipliers=None,
            )
            self._text_encoder.eval().requires_grad_(False)
            self._t5_tokenizer = self._modules.anima_utils.load_t5_tokenizer(None)
            self._tokenize_strategy = self._modules.strategy_anima.AnimaTokenizeStrategy(
                qwen3_tokenizer=self._qwen_tokenizer,
                t5_tokenizer=self._t5_tokenizer,
                qwen3_max_length=base.DEFAULT_TEXT_MAX_LENGTH,
                t5_max_length=base.DEFAULT_TEXT_MAX_LENGTH,
            )

            vae_device: torch.device | str = self.device if self.offload_mode == "high_vram" else "cpu"
            self._vae = self._modules.qwen_image_autoencoder_kl.load_vae(
                self.vae_path,
                device=vae_device,
                disable_mmap=True,
                spatial_chunk_size=None if self.vae_chunk_size == 0 else self.vae_chunk_size,
                disable_cache=self.vae_disable_cache,
            )
            self._vae.to(dtype=self.dtype)
            self._vae.eval().requires_grad_(False)
            self._model_loads += 1

    def _preprocess_reference(
        self,
        image: torch.Tensor,
        *,
        name: str,
        preprocess_mode: str,
        reference_max_area: int,
        output_width: int,
        output_height: int,
    ) -> tuple[Image.Image, str]:
        """Use the exact V7 edit-canvas transform for the aligned source."""

        pil = self._comfy_image_to_pil(image, name=name).convert("RGB")
        multiple_of = int(self._modules.qwen_image_autoencoder_kl.SCALE_FACTOR) * int(self._anima.patch_spatial)
        if preprocess_mode == "match_output_edit":
            geometry = importlib.import_module(
                f"{base._VENDOR_PACKAGE_NAME}.library.anima_edit_geometry"
            )
            plan = geometry.make_edit_canvas_plan_v1(
                input_wh=(pil.width, pil.height),
                output_hw=(int(output_height), int(output_width)),
                flip_x=False,
                resize_interpolation="lanczos",
            )
            prepared = Image.fromarray(
                geometry.apply_edit_canvas_plan_v1(pil, plan), mode="RGB"
            )
        elif preprocess_mode == "independent_reference":
            prepared = self._modules.strategy_anima.preprocess_anima_reference_image(
                pil,
                max_area=int(reference_max_area),
                multiple_of=multiple_of,
                target_size_hw=None,
                flipped=False,
            )
        else:
            raise base.InputValidationError(f"Unknown V7 preprocess mode {preprocess_mode!r}.")

        digest = hashlib.sha256()
        digest.update(b"anima-v7-unified-reference-latent-v1\0")
        digest.update(
            f"mode={preprocess_mode}|prepared={prepared.width}x{prepared.height}|output={output_width}x{output_height}|multiple={multiple_of}|max_area={reference_max_area}|".encode("ascii")
        )
        digest.update(self.vae_fingerprint.path.encode("utf-8"))
        digest.update(str(self.vae_fingerprint.size).encode("ascii"))
        digest.update(str(self.vae_fingerprint.mtime_ns).encode("ascii"))
        digest.update(prepared.tobytes())
        return prepared, digest.hexdigest()

    def _prepare_t2i_conditionings(self, prompt: str, negative_prompt: str) -> tuple[Any, Any]:
        positive_raw, negative_raw = self._get_raw_text_encodings(prompt, negative_prompt)
        prepare = self._modules.anima_text_conditioning.prepare_anima_prompt_conditioning
        positive = prepare(
            prompt,
            positive_raw.as_sequence(),
            tokenize_strategy=self._tokenize_strategy,
            device=self.device,
            dtype=self.dtype,
            reference_slot_ids=None,
            require_reference_binding=False,
            max_slots=2,
        )
        negative = prepare(
            negative_prompt,
            negative_raw.as_sequence(),
            tokenize_strategy=self._tokenize_strategy,
            device=self.device,
            dtype=self.dtype,
            reference_slot_ids=None,
            require_reference_binding=False,
            max_slots=2,
        )
        if (
            positive.reference_clause_masks is not None
            or negative.reference_clause_masks is not None
        ):
            raise base.AnimaRuntimeError(
                "V7 T2I unexpectedly constructed visual clause masks."
            )
        return positive, negative

    def _prepare_text_conditionings(
        self,
        prompt: str,
        negative_prompt: str,
        *,
        reference_slot_ids: tuple[int, ...],
    ) -> tuple[Any, Any]:
        """Encode the exact prompt; slots are metadata, never generated clauses."""

        slot_rows = [list(reference_slot_ids)]
        requires_masks = (
            self._modules.anima_text_conditioning.reference_router_requires_clause_masks(
                self._anima,
                slot_rows,
                use_reference_sequence=True,
            )
        )
        if requires_masks:
            raise base.AnimaRuntimeError(
                "V7 checkpoint unexpectedly requested parser-generated clause masks."
            )
        positive_raw, negative_raw = self._get_raw_text_encodings(
            prompt, negative_prompt
        )
        prepare = self._modules.anima_text_conditioning.prepare_anima_prompt_conditioning
        try:
            positive = prepare(
                prompt,
                positive_raw.as_sequence(),
                tokenize_strategy=self._tokenize_strategy,
                device=self.device,
                dtype=self.dtype,
                reference_slot_ids=slot_rows,
                require_reference_binding=False,
                max_slots=2,
            )
            negative = prepare(
                negative_prompt,
                negative_raw.as_sequence(),
                tokenize_strategy=self._tokenize_strategy,
                device=self.device,
                dtype=self.dtype,
                reference_slot_ids=None,
                require_reference_binding=False,
                max_slots=2,
            )
        except (TypeError, ValueError) as exc:
            raise base.InputValidationError(
                f"Invalid V7 prompt conditioning: {exc}"
            ) from exc
        if (
            positive.reference_clause_masks is not None
            or negative.reference_clause_masks is not None
        ):
            raise base.AnimaRuntimeError(
                "V7 exact-prompt runtime forbids reference_clause_masks."
            )
        return positive, negative

    def _denoise_v7(
        self,
        *,
        positive_conditioning: Any,
        negative_conditioning: Any,
        reference_latents: Optional[list[list[torch.Tensor]]],
        reference_slot_ids: tuple[int, ...],
        aligned_source_slot_id: Optional[int],
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance_scale: float,
        flow_shift: float,
        progress_callback: Optional[Callable[[int, int], None]],
        interrupt_callback: Optional[Callable[[], None]],
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        shape = (1, int(self._modules.anima_models.Anima.LATENT_CHANNELS), 1, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device="cpu", dtype=self.dtype).to(self.device)
        padding_mask = torch.zeros(1, 1, height // 8, width // 8, dtype=self.dtype, device=self.device)
        timesteps, sigmas = self._modules.hunyuan_image_utils.get_timesteps_sigmas(
            int(steps), float(flow_shift), self.device
        )
        timesteps = (timesteps / 1000).to(self.device, dtype=self.dtype)

        aligned_tensor = None
        if aligned_source_slot_id is not None:
            if reference_latents is None or aligned_source_slot_id not in reference_slot_ids:
                raise base.InputValidationError("Edit source slot is absent from reference_slot_ids.")
            physical = reference_slot_ids.index(int(aligned_source_slot_id))
            source = reference_latents[0][physical]
            if source.ndim == 4:
                source = source.unsqueeze(2)
            if tuple(source.shape) != tuple(latents.shape):
                raise base.InputValidationError(
                    f"Aligned source latent must match target canvas: {tuple(source.shape)} != {tuple(latents.shape)}."
                )
            aligned_tensor = torch.tensor([int(aligned_source_slot_id)], device=self.device, dtype=torch.long)

        use_llm_adapter = bool(getattr(self._anima, "use_llm_adapter", False))
        positive_kwargs = positive_conditioning.model_text_kwargs(
            self.device, self.dtype, use_llm_adapter=use_llm_adapter
        )
        negative_kwargs = negative_conditioning.model_text_kwargs(
            self.device,
            self.dtype,
            router_conditioning=positive_conditioning,
            use_llm_adapter=use_llm_adapter,
        )
        use_refs = reference_latents is not None
        common_kwargs = {
            "padding_mask": padding_mask,
            "reference_latents": reference_latents,
            "reference_slot_ids": [list(reference_slot_ids)] if use_refs else None,
            "reference_t_offset_scale": 10,
            "ip_adapter_latents": None,
            "ip_adapter_embeds": None,
            "use_ip_adapter": False,
            "use_reference_sequence": use_refs,
            "aligned_source_slot_ids": aligned_tensor,
        }
        do_cfg = float(guidance_scale) != 1.0
        total = len(timesteps)
        for index, timestep in enumerate(timesteps):
            if interrupt_callback is not None:
                interrupt_callback()
            timestep_batch = timestep.expand(latents.shape[0])
            noise_pred = self._anima(latents, timestep_batch, **common_kwargs, **positive_kwargs)
            if do_cfg:
                uncond = self._anima(latents, timestep_batch, **common_kwargs, **negative_kwargs)
                noise_pred = uncond + float(guidance_scale) * (noise_pred - uncond)
            latents = self._modules.hunyuan_image_utils.step(latents, noise_pred, sigmas, index).to(latents.dtype)
            if progress_callback is not None:
                progress_callback(index + 1, total)
        return latents

    def generate(
        self,
        reference_images: Sequence[torch.Tensor],
        reference_slot_ids: Sequence[int],
        prompt: str,
        *,
        task_mode: str = "reference",
        aligned_source_slot_id: Optional[int] = None,
        negative_prompt: str = "",
        width: int = V7_DEFAULT_WIDTH,
        height: int = V7_DEFAULT_HEIGHT,
        seed: int = 0,
        steps: int = V7_DEFAULT_STEPS,
        guidance_scale: float = V7_DEFAULT_GUIDANCE_SCALE,
        flow_shift: float = base.DEFAULT_FLOW_SHIFT,
        native_reference_scale: float = V7_REFERENCE_SCALE,
        reference_max_area: int = V7_DEFAULT_REFERENCE_MAX_AREA,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        interrupt_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        mode = str(task_mode)
        if mode not in V7_TASK_MODES:
            raise base.InputValidationError(f"task_mode must be one of {sorted(V7_TASK_MODES)}")
        if mode == "edit":
            if len(reference_images) != 1:
                raise base.InputValidationError("The V7 Edit node accepts exactly one aligned source image.")
            if aligned_source_slot_id is None:
                raise base.InputValidationError("Edit requires aligned_source_slot_id.")
            preprocess_mode = "match_output_edit"
        else:
            if aligned_source_slot_id is not None:
                raise base.InputValidationError("Reference generation must not set an aligned source slot.")
            preprocess_mode = "independent_reference"

        images, slots = self._normalise_reference_request(
            reference_images, reference_slot_ids, preprocess_mode=preprocess_mode
        )
        expected_slots = tuple(range(len(images)))
        if slots != expected_slots:
            raise base.InputValidationError(
                "V7 compacts active physical images to structural instances "
                f"{expected_slots!r}; semantic/sparse slot request {slots!r} is forbidden."
            )
        if aligned_source_slot_id is not None and int(aligned_source_slot_id) not in slots:
            raise base.InputValidationError("aligned_source_slot_id must identify the supplied logical slot.")
        self._validate_sample_inputs(
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            native_reference_scale=native_reference_scale,
            reference_max_area=reference_max_area,
        )
        with self.lock, torch.inference_mode():
            self._ensure_open()
            self._anima.to(self.device, dtype=self.dtype)
            self._anima.set_native_reference_fixed12ref_vectorized(True)
            # Set on every call: Comfy caches the runtime, so this prevents an
            # Edit preset from leaking into a later reference-generation call.
            self._anima.set_native_reference_scale(float(native_reference_scale))
            positive, negative = self._prepare_text_conditionings(
                prompt, negative_prompt, reference_slot_ids=slots
            )
            refs = self._encode_references(
                images,
                preprocess_mode=preprocess_mode,
                reference_max_area=int(reference_max_area),
                output_width=int(width),
                output_height=int(height),
            )
            latent = self._denoise_v7(
                positive_conditioning=positive,
                negative_conditioning=negative,
                reference_latents=refs,
                reference_slot_ids=slots,
                aligned_source_slot_id=None if aligned_source_slot_id is None else int(aligned_source_slot_id),
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                flow_shift=float(flow_shift),
                progress_callback=progress_callback,
                interrupt_callback=interrupt_callback,
            )
            return self._decode_latent(latent)

    def generate_t2i(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = V7_DEFAULT_WIDTH,
        height: int = V7_DEFAULT_HEIGHT,
        seed: int = 0,
        steps: int = V7_DEFAULT_STEPS,
        guidance_scale: float = V7_DEFAULT_GUIDANCE_SCALE,
        flow_shift: float = base.DEFAULT_FLOW_SHIFT,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        interrupt_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        self._validate_sample_inputs(
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            native_reference_scale=V7_REFERENCE_SCALE,
            reference_max_area=V7_DEFAULT_REFERENCE_MAX_AREA,
        )
        with self.lock, torch.inference_mode():
            self._ensure_open()
            self._anima.to(self.device, dtype=self.dtype)
            self._anima.set_native_reference_scale(V7_REFERENCE_SCALE)
            positive, negative = self._prepare_t2i_conditionings(prompt, negative_prompt)
            latent = self._denoise_v7(
                positive_conditioning=positive,
                negative_conditioning=negative,
                reference_latents=None,
                reference_slot_ids=(),
                aligned_source_slot_id=None,
                width=int(width),
                height=int(height),
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                flow_shift=float(flow_shift),
                progress_callback=progress_callback,
                interrupt_callback=interrupt_callback,
            )
            return self._decode_latent(latent)

    def verify_release_sha256(self) -> None:
        with self.lock:
            self._ensure_open()
            if self._release_sha256_verified:
                return
            self.checkpoint_info = validate_v7_checkpoint(self.checkpoint_path, verify_sha256=True)
            base.validate_qwen_text_encoder(self.text_encoder_path, verify_sha256=True)
            base.validate_qwen_image_vae(self.vae_path, verify_sha256=True)
            self._release_sha256_verified = True


def get_or_create_v7_runtime(
    checkpoint_path: os.PathLike[str] | str,
    text_encoder_path: os.PathLike[str] | str,
    vae_path: os.PathLike[str] | str,
    *,
    runtime_root: os.PathLike[str] | str | None = None,
    device: str | torch.device | None = None,
    offload_mode: str = "balanced",
    attn_mode: str = "torch",
    vae_chunk_size: int = 0,
    vae_disable_cache: bool = False,
    prompt_cache_entries: int = 64,
    reference_cache_entries: int = 16,
    verify_checkpoint_sha256: bool = False,
) -> AnimaNativeContextV7Runtime:
    key = base._make_runtime_cache_key(
        checkpoint_path,
        text_encoder_path,
        vae_path,
        runtime_root=runtime_root,
        device=device,
        offload_mode=offload_mode,
        attn_mode=attn_mode,
        vae_chunk_size=vae_chunk_size,
        vae_disable_cache=vae_disable_cache,
        runtime_kind="v7-native-context-v1",
    )
    with base._MODEL_CACHE_LOCK:
        runtime = base._MODEL_CACHE.get(key)
        if runtime is not None and not runtime._closed:
            if not isinstance(runtime, AnimaNativeContextV7Runtime):
                raise base.AnimaRuntimeError("Runtime cache collision for V7.")
            runtime.set_condition_cache_limits(
                prompt_cache_entries=prompt_cache_entries,
                reference_cache_entries=reference_cache_entries,
            )
            if verify_checkpoint_sha256:
                runtime.verify_release_sha256()
            base._MODEL_CACHE.move_to_end(key)
            return runtime
        if runtime is not None:
            del base._MODEL_CACHE[key]
        created = AnimaNativeContextV7Runtime(
            key.checkpoint.path,
            key.text_encoder.path,
            key.vae.path,
            runtime_root=key.runtime_root,
            device=key.device,
            offload_mode=offload_mode,
            attn_mode=attn_mode,
            vae_chunk_size=vae_chunk_size,
            vae_disable_cache=vae_disable_cache,
            prompt_cache_entries=prompt_cache_entries,
            reference_cache_entries=reference_cache_entries,
            verify_checkpoint_sha256=verify_checkpoint_sha256,
        )
        base._MODEL_CACHE[key] = created
        base._MODEL_CACHE.move_to_end(key)
        while len(base._MODEL_CACHE) > base.PROCESS_MODEL_CACHE_CAPACITY:
            base._MODEL_CACHE.popitem(last=False)
        return created


__all__ = [
    "AnimaNativeContextV7Runtime",
    "V7CheckpointInfo",
    "V7_EDIT_SCALE",
    "V7_REFERENCE_SCALE",
    "get_or_create_v7_runtime",
    "validate_v7_checkpoint",
]
