"""In-process runtimes for integrated Anima Native Reference checkpoints.

The runtime is deliberately independent from ComfyUI at import time.  A node can
pass/receive ordinary Comfy ``IMAGE`` tensors (float ``[B,H,W,C]`` in ``[0,1]``)
without writing temporary files.  The implementation is pinned to the vendored
Anima runtime snapshot and fails closed when a checkpoint is not one of the
exact, audited integrated-native-reference layouts.

The legacy V2 E180 runtime remains available so existing workflows keep their
class IDs.  The V4 publication runtime is a separate fail-closed path for the
final competitive text-slot-router checkpoint.  In that path raw Qwen states
and masks are retained for routing, neutral-T5 supplies token IDs/masks only,
the model's own LLM adapter supplies ordinary cross-attention, and the CFG
negative branch reuses the *positive* router-only conditioning.

There are two cache layers:

1. a process-wide strong-reference LRU model cache, keyed by model file
   fingerprints and runtime options; and
2. bounded per-runtime condition caches for prompt embeddings and reference VAE
   latents.

No external LoRA, IP-Adapter, or adapter sidecar is accepted or loaded.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch


VENDORED_ANIMA_COMMIT = "2ae811d296ff4159c6024c4a86415d19961a388c"
VENDORED_ANIMA_OVERLAY = "v4-scaled-mixture-final49k-20260724"

E180_EXPECTED_SHA256 = (
    "1f970a7867dd7b65858d30b58135134ce84fd3b552ab27fc9f07e7f15209c6dd"
)
E180_EXPECTED_FILE_SIZE = 4_271_362_542
E180_EXPECTED_TOTAL_TENSORS = 1_222
E180_EXPECTED_NATIVE_TENSORS = 534
QWEN_EXPECTED_SHA256 = (
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
)
QWEN_EXPECTED_FILE_SIZE = 1_192_135_096
QWEN_EXPECTED_TOTAL_TENSORS = 310
VAE_EXPECTED_SHA256 = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
VAE_EXPECTED_FILE_SIZE = 253_806_246
VAE_EXPECTED_TOTAL_TENSORS = 194
E180_EXPECTED_CONFIG: Mapping[str, Any] = {
    "architecture": "v2",
    "gate_dim": 4,
    "initial_gate": 0.5,
    "max_reference_images": 2,
    "rank": 64,
    "router_dim": 64,
}
E180_REQUIRED_METADATA: Mapping[str, str] = {
    "anima_checkpoint_layout": "integrated_single_checkpoint",
    "anima_external_adapter_required": "false",
    "anima_native_reference_conditioning": "true",
    "anima_native_reference_version": "2",
    "anima_prediction_objective": "rectified_flow_velocity",
    "anima_reference_slot_semantics": "ordered_image_index",
    "modelspec.architecture": "anima-preview",
    "ss_epoch": "180",
    "ss_steps": "64080",
    # E180 is a numbered milestone from the planned E200 run.
    "ss_num_epochs": "200",
    "ss_max_train_steps": "71200",
}

# Final self-contained scaled-mixture checkpoint produced by the formal 49k
# run.  The underlying integrated reference architecture remains "v2"; V4 is
# the project/release generation that adds the competitive text-slot router.
V4_EXPECTED_SHA256 = (
    "4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f"
)
V4_EXPECTED_FILE_SIZE = 4_302_295_014
V4_EXPECTED_TOTAL_TENSORS = 1_614
V4_EXPECTED_NATIVE_TENSORS = 926
V4_EXPECTED_CONFIG: Mapping[str, Any] = {
    "architecture": "v2",
    "gate_dim": 4,
    "initial_gate": 0.5,
    "max_reference_images": 2,
    "rank": 64,
    "router_dim": 64,
    "router_null_enabled": True,
    "router_temperature": 1.0,
    "routing_alpha": 1.0,
    "routing_mode": "competitive_text_slot_v1",
}
V4_REQUIRED_METADATA: Mapping[str, str] = {
    "anima_checkpoint_layout": "integrated_single_checkpoint",
    "anima_external_adapter_required": "false",
    "anima_native_reference_conditioning": "true",
    # This is intentionally "2": V4 did not rename the underlying native
    # reference architecture serialized by the official implementation.
    "anima_native_reference_version": "2",
    "anima_prediction_objective": "rectified_flow_velocity",
    "anima_reference_slot_semantics": "ordered_image_index",
    "anima_native_reference_routing_mode": "competitive_text_slot_v1",
    "anima_native_reference_routing_alpha": "1.0",
    "anima_native_reference_router_temperature": "1.0",
    "anima_native_reference_router_null_enabled": "true",
    "anima_v4_scaled_mixture": "true",
    "anima_v4_stage": "scaled_edit_character_dualref",
    "modelspec.architecture": "anima-preview",
    "ss_epoch": "2",
    "ss_steps": "49000",
    "ss_num_epochs": "2",
    "ss_max_train_steps": "49000",
}

DEFAULT_REFERENCE_MAX_AREA = 65_536
DEFAULT_VAE_CHUNK_SIZE = 64
DEFAULT_TEXT_MAX_LENGTH = 512
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 1.0
# Final V4 publication defaults.  Keep the legacy constants above unchanged so
# existing E180 callers retain their accepted 40-step / CFG-1 behavior.
V4_DEFAULT_STEPS = 30
V4_DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_FLOW_SHIFT = 5.0
DEFAULT_NATIVE_REFERENCE_SCALE = 1.0
ORDERED_SLOT_IDS = ((0, 1),)
SINGLE_SLOT_IDS = ((0,), (1,))
V4_PREPROCESS_MODES = frozenset({"independent_reference", "match_output_edit"})
# Keep the two most recently used heavyweight runtimes alive even when ComfyUI
# runs with ``--cache-none`` or evicts a loader-node output.  Two entries allow
# one alternate loader configuration without making model residency unbounded.
PROCESS_MODEL_CACHE_CAPACITY = 2


class AnimaRuntimeError(RuntimeError):
    """Base exception for the custom runtime."""


class CheckpointValidationError(AnimaRuntimeError):
    """Raised when a file is not the exact integrated E180 checkpoint."""


class RuntimeImportError(AnimaRuntimeError):
    """Raised when the pinned vendored runtime cannot be imported safely."""


class InputValidationError(AnimaRuntimeError, ValueError):
    """Raised for invalid Comfy tensor or sampling inputs."""


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: os.PathLike[str] | str) -> "FileFingerprint":
        resolved = Path(path).expanduser().resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file():
            raise FileNotFoundError(f"Expected a file: {resolved}")
        return cls(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


@dataclass(frozen=True)
class E180CheckpointInfo:
    fingerprint: FileFingerprint
    metadata: Mapping[str, str]
    config: Mapping[str, Any]
    tensor_count: int
    native_tensor_count: int
    dtype_counts: Mapping[str, int]
    sha256: Optional[str] = None


@dataclass(frozen=True)
class V4CheckpointInfo:
    fingerprint: FileFingerprint
    metadata: Mapping[str, str]
    config: Mapping[str, Any]
    tensor_count: int
    native_tensor_count: int
    dtype_counts: Mapping[str, int]
    sha256: Optional[str] = None


@dataclass(frozen=True)
class RawPromptEncoding:
    """CPU-resident raw Qwen plus neutral-T5-token conditioning."""

    raw_qwen_context: torch.Tensor
    source_attention_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_attention_mask: torch.Tensor

    def as_sequence(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.raw_qwen_context,
            self.source_attention_mask,
            self.target_input_ids,
            self.target_attention_mask,
        )


@dataclass(frozen=True)
class RuntimeModules:
    anima_utils: ModuleType
    anima_models: ModuleType
    hunyuan_image_utils: ModuleType
    qwen_image_autoencoder_kl: ModuleType
    strategy_anima: ModuleType
    anima_text_conditioning: ModuleType
    anima_reference_binding: ModuleType


@dataclass(frozen=True)
class RuntimeCacheKey:
    runtime_kind: str
    checkpoint: FileFingerprint
    text_encoder: FileFingerprint
    vae: FileFingerprint
    runtime_root: str
    device: str
    offload_mode: str
    attention_mode: str
    vae_chunk_size: int
    vae_disable_cache: bool


@dataclass(frozen=True)
class CacheStatistics:
    model_loads: int
    prompt_hits: int
    prompt_misses: int
    reference_hits: int
    reference_misses: int
    prompt_entries: int
    reference_entries: int


def _normalise_state_key(key: str) -> str:
    return key[len("net.") :] if key.startswith("net.") else key


def _is_native_reference_key(raw_key: str) -> bool:
    key = _normalise_state_key(raw_key)
    return (
        key in {"reference_slot_embeddings.weight", "reference_type_embedding"}
        or ".native_reference_attn." in key
    )


def _is_external_lora_key(raw_key: str) -> bool:
    # Do not match the base model's legitimate ``adaln_lora`` modules.
    key = _normalise_state_key(raw_key)
    return key.startswith(("lora_unet_", "lora_te_", "lycoris_", "network_lora_"))


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expect_metadata(metadata: Mapping[str, str], key: str, expected: str) -> None:
    actual = metadata.get(key)
    if actual != expected:
        raise CheckpointValidationError(
            f"Checkpoint metadata mismatch for {key!r}: expected {expected!r}, got {actual!r}."
        )


def validate_e180_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    verify_sha256: bool = False,
) -> E180CheckpointInfo:
    """Validate E180 identity and architecture without loading its tensors.

    The default performs an exact file-size, metadata, key-count, dtype and
    authoritative-shape validation.  ``verify_sha256=True`` additionally reads
    all 4.27 GB once and compares the release checksum.
    """

    fingerprint = FileFingerprint.from_path(checkpoint_path)
    path = Path(fingerprint.path)
    if path.suffix.lower() != ".safetensors":
        raise CheckpointValidationError(f"E180 must be a .safetensors file: {path}")
    if fingerprint.size != E180_EXPECTED_FILE_SIZE:
        raise CheckpointValidationError(
            f"E180 byte size mismatch: expected {E180_EXPECTED_FILE_SIZE}, got {fingerprint.size}."
        )

    dtype_counts: Dict[str, int] = {}
    native_keys: list[str] = []
    key_shapes: Dict[str, Tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        if len(keys) != E180_EXPECTED_TOTAL_TENSORS:
            raise CheckpointValidationError(
                f"E180 tensor count mismatch: expected {E180_EXPECTED_TOTAL_TENSORS}, got {len(keys)}."
            )
        external_lora = [key for key in keys if _is_external_lora_key(key)]
        if external_lora:
            raise CheckpointValidationError(
                f"External LoRA tensors are forbidden; first keys: {external_lora[:5]!r}."
            )
        for raw_key in keys:
            tensor_slice = handle.get_slice(raw_key)
            dtype_name = str(tensor_slice.get_dtype())
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            if dtype_name != "BF16":
                raise CheckpointValidationError(
                    f"E180 contains non-BF16 tensor {raw_key!r} with dtype {dtype_name}."
                )
            if _is_native_reference_key(raw_key):
                native_keys.append(raw_key)
                key_shapes[_normalise_state_key(raw_key)] = tuple(
                    int(v) for v in tensor_slice.get_shape()
                )

    if len(native_keys) != E180_EXPECTED_NATIVE_TENSORS:
        raise CheckpointValidationError(
            "E180 native-reference tensor count mismatch: "
            f"expected {E180_EXPECTED_NATIVE_TENSORS}, got {len(native_keys)}."
        )
    for key, expected in E180_REQUIRED_METADATA.items():
        _expect_metadata(metadata, key, expected)

    raw_config = metadata.get("anima_native_reference_config")
    try:
        config = json.loads(raw_config) if raw_config is not None else None
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"Invalid anima_native_reference_config JSON: {raw_config!r}."
        ) from exc
    if config != dict(E180_EXPECTED_CONFIG):
        raise CheckpointValidationError(
            f"E180 native-reference config mismatch: expected {dict(E180_EXPECTED_CONFIG)!r}, got {config!r}."
        )

    required_shapes: Dict[str, Tuple[int, ...]] = {
        "reference_slot_embeddings.weight": (2, 2048),
        "reference_type_embedding": (2048,),
    }
    for block_index in range(28):
        stem = f"blocks.{block_index}.native_reference_attn"
        required_shapes.update(
            {
                f"{stem}.k_down.weight": (64, 2048),
                f"{stem}.prompt_key.weight": (64, 1024),
                f"{stem}.head_gate.weight": (16, 64),
                f"{stem}.target_spatial.weight": (64, 2048),
            }
        )
    for key, expected_shape in required_shapes.items():
        actual_shape = key_shapes.get(key)
        if actual_shape != expected_shape:
            raise CheckpointValidationError(
                f"E180 tensor shape mismatch for {key!r}: expected {expected_shape}, got {actual_shape}."
            )

    checksum: Optional[str] = None
    if verify_sha256:
        checksum = _sha256_file(path)
        if checksum != E180_EXPECTED_SHA256:
            raise CheckpointValidationError(
                f"E180 SHA-256 mismatch: expected {E180_EXPECTED_SHA256}, got {checksum}."
            )

    return E180CheckpointInfo(
        fingerprint=fingerprint,
        metadata=metadata,
        config=dict(config),
        tensor_count=E180_EXPECTED_TOTAL_TENSORS,
        native_tensor_count=len(native_keys),
        dtype_counts=dict(dtype_counts),
        sha256=checksum,
    )


def _v4_required_native_shapes() -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {
        "reference_slot_embeddings.weight": (2, 2048),
        "reference_type_embedding": (2048,),
    }
    per_block: Mapping[str, Tuple[int, ...]] = {
        "clause_pooler.key_proj.weight": (64, 1024),
        "clause_pooler.pool_queries": (4, 64),
        "clause_pooler.value_proj.weight": (64, 1024),
        "competitive_film.weight": (128, 64),
        "competitive_output_gain": (),
        "competitive_router.candidate_key.weight": (64, 128),
        "competitive_router.context_key.weight": (64, 64),
        "competitive_router.target_query.weight": (64, 2048),
        "condition_spatial.weight": (64, 64),
        "head_gate.bias": (16,),
        "head_gate.weight": (16, 64),
        "k_down.weight": (64, 2048),
        "k_up.weight": (2048, 64),
        "output_down.weight": (64, 2048),
        "output_up.weight": (2048, 64),
        "pointer_head.bias": (2,),
        "pointer_head.weight": (2, 64),
        "prompt_key.weight": (64, 1024),
        "prompt_value.weight": (64, 1024),
        "query_correction.condition_proj.weight": (64, 64),
        "query_correction.output_proj.bias": (2048,),
        "query_correction.output_proj.weight": (2048, 64),
        "query_correction.query_proj.weight": (64, 2048),
        "reference_summary.weight": (64, 2048),
        "router_mlp.1.weight": (64, 64),
        "router_to_film.weight": (128, 64),
        "slot_prompt_query.weight": (64, 2048),
        "slot_summary.weight": (64, 2048),
        "spatial_bias": (16,),
        "target_spatial.weight": (64, 2048),
        "timestep_summary.weight": (64, 2048),
        "v_down.weight": (64, 2048),
        "v_up.weight": (2048, 64),
    }
    for block_index in range(28):
        stem = f"blocks.{block_index}.native_reference_attn"
        for suffix, shape in per_block.items():
            shapes[f"{stem}.{suffix}"] = shape
    if len(shapes) != V4_EXPECTED_NATIVE_TENSORS:
        raise AssertionError(
            "Internal V4 native tensor-shape contract is incomplete: "
            f"{len(shapes)} != {V4_EXPECTED_NATIVE_TENSORS}."
        )
    return shapes


def validate_v4_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    verify_sha256: bool = False,
) -> V4CheckpointInfo:
    """Validate the exact final 49k V4 checkpoint without loading tensor data."""

    fingerprint = FileFingerprint.from_path(checkpoint_path)
    path = Path(fingerprint.path)
    if path.suffix.lower() != ".safetensors":
        raise CheckpointValidationError(
            f"Final V4 model must be a .safetensors file: {path}"
        )
    if fingerprint.size != V4_EXPECTED_FILE_SIZE:
        raise CheckpointValidationError(
            "Final V4 byte size mismatch: "
            f"expected {V4_EXPECTED_FILE_SIZE}, got {fingerprint.size}."
        )

    dtype_counts: Dict[str, int] = {}
    native_shapes: Dict[str, Tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        if len(keys) != V4_EXPECTED_TOTAL_TENSORS:
            raise CheckpointValidationError(
                "Final V4 tensor count mismatch: "
                f"expected {V4_EXPECTED_TOTAL_TENSORS}, got {len(keys)}."
            )
        external_lora = [key for key in keys if _is_external_lora_key(key)]
        if external_lora:
            raise CheckpointValidationError(
                "External LoRA tensors are forbidden; "
                f"first keys: {external_lora[:5]!r}."
            )
        for raw_key in keys:
            tensor_slice = handle.get_slice(raw_key)
            dtype_name = str(tensor_slice.get_dtype())
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            if dtype_name != "BF16":
                raise CheckpointValidationError(
                    "Final V4 contains non-BF16 tensor "
                    f"{raw_key!r} with dtype {dtype_name}."
                )
            if _is_native_reference_key(raw_key):
                native_shapes[_normalise_state_key(raw_key)] = tuple(
                    int(value) for value in tensor_slice.get_shape()
                )

    if len(native_shapes) != V4_EXPECTED_NATIVE_TENSORS:
        raise CheckpointValidationError(
            "Final V4 native-reference tensor count mismatch: "
            f"expected {V4_EXPECTED_NATIVE_TENSORS}, got {len(native_shapes)}."
        )
    for key, expected in V4_REQUIRED_METADATA.items():
        _expect_metadata(metadata, key, expected)

    raw_config = metadata.get("anima_native_reference_config")
    try:
        config = json.loads(raw_config) if raw_config is not None else None
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"Invalid anima_native_reference_config JSON: {raw_config!r}."
        ) from exc
    if config != dict(V4_EXPECTED_CONFIG):
        raise CheckpointValidationError(
            "Final V4 native-reference config mismatch: "
            f"expected {dict(V4_EXPECTED_CONFIG)!r}, got {config!r}."
        )

    raw_router_config = metadata.get("anima_native_reference_router_config")
    expected_router_config = {
        "router_null_enabled": True,
        "router_temperature": 1.0,
        "routing_alpha": 1.0,
        "routing_mode": "competitive_text_slot_v1",
    }
    try:
        router_config = (
            json.loads(raw_router_config) if raw_router_config is not None else None
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"Invalid anima_native_reference_router_config JSON: {raw_router_config!r}."
        ) from exc
    if router_config != expected_router_config:
        raise CheckpointValidationError(
            "Final V4 router config mismatch: "
            f"expected {expected_router_config!r}, got {router_config!r}."
        )

    expected_shapes = _v4_required_native_shapes()
    missing = sorted(set(expected_shapes) - set(native_shapes))
    unexpected = sorted(set(native_shapes) - set(expected_shapes))
    if missing or unexpected:
        raise CheckpointValidationError(
            "Final V4 native key set mismatch: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}."
        )
    for key, expected_shape in expected_shapes.items():
        actual_shape = native_shapes[key]
        if actual_shape != expected_shape:
            raise CheckpointValidationError(
                f"Final V4 tensor shape mismatch for {key!r}: "
                f"expected {expected_shape}, got {actual_shape}."
            )

    checksum: Optional[str] = None
    if verify_sha256:
        checksum = _sha256_file(path)
        if checksum != V4_EXPECTED_SHA256:
            raise CheckpointValidationError(
                "Final V4 SHA-256 mismatch: "
                f"expected {V4_EXPECTED_SHA256}, got {checksum}."
            )

    return V4CheckpointInfo(
        fingerprint=fingerprint,
        metadata=metadata,
        config=dict(config),
        tensor_count=V4_EXPECTED_TOTAL_TENSORS,
        native_tensor_count=len(native_shapes),
        dtype_counts=dict(dtype_counts),
        sha256=checksum,
    )


def _validate_auxiliary_safetensors(
    model_path: os.PathLike[str] | str,
    *,
    label: str,
    expected_file_size: int,
    expected_tensor_count: int,
    required_shapes: Mapping[str, Tuple[int, ...]],
    expected_sha256: str,
    verify_sha256: bool,
) -> FileFingerprint:
    """Fail closed on a wrong Qwen/VAE selection before allocating a model."""

    fingerprint = FileFingerprint.from_path(model_path)
    path = Path(fingerprint.path)
    if path.suffix.lower() != ".safetensors":
        raise CheckpointValidationError(f"{label} must be a .safetensors file: {path}")
    if fingerprint.size != int(expected_file_size):
        raise CheckpointValidationError(
            f"{label} byte size mismatch: expected {expected_file_size}, got {fingerprint.size}."
        )

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != int(expected_tensor_count):
            raise CheckpointValidationError(
                f"{label} tensor count mismatch: expected {expected_tensor_count}, got {len(keys)}."
            )
        for key in keys:
            dtype_name = str(handle.get_slice(key).get_dtype())
            if dtype_name != "BF16":
                raise CheckpointValidationError(
                    f"{label} contains non-BF16 tensor {key!r} with dtype {dtype_name}."
                )
        for key, expected_shape in required_shapes.items():
            if key not in keys:
                raise CheckpointValidationError(
                    f"{label} is missing required tensor {key!r}."
                )
            actual_shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
            if actual_shape != tuple(expected_shape):
                raise CheckpointValidationError(
                    f"{label} tensor shape mismatch for {key!r}: "
                    f"expected {tuple(expected_shape)}, got {actual_shape}."
                )

    if verify_sha256:
        checksum = _sha256_file(path)
        if checksum != expected_sha256:
            raise CheckpointValidationError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {checksum}."
            )
    return fingerprint


def validate_qwen_text_encoder(
    model_path: os.PathLike[str] | str, *, verify_sha256: bool = False
) -> FileFingerprint:
    return _validate_auxiliary_safetensors(
        model_path,
        label="Qwen3-0.6B text encoder",
        expected_file_size=QWEN_EXPECTED_FILE_SIZE,
        expected_tensor_count=QWEN_EXPECTED_TOTAL_TENSORS,
        required_shapes={
            "model.embed_tokens.weight": (151_936, 1_024),
            "model.layers.0.self_attn.q_proj.weight": (2_048, 1_024),
            "model.layers.27.mlp.down_proj.weight": (1_024, 3_072),
            "model.layers.27.self_attn.q_proj.weight": (2_048, 1_024),
            "model.norm.weight": (1_024,),
        },
        expected_sha256=QWEN_EXPECTED_SHA256,
        verify_sha256=verify_sha256,
    )


def validate_qwen_image_vae(
    model_path: os.PathLike[str] | str, *, verify_sha256: bool = False
) -> FileFingerprint:
    return _validate_auxiliary_safetensors(
        model_path,
        label="Qwen Image VAE",
        expected_file_size=VAE_EXPECTED_FILE_SIZE,
        expected_tensor_count=VAE_EXPECTED_TOTAL_TENSORS,
        required_shapes={
            "conv1.weight": (32, 32, 1, 1, 1),
            "conv2.weight": (16, 16, 1, 1, 1),
            "decoder.conv1.weight": (384, 16, 3, 3, 3),
            "decoder.head.2.weight": (3, 96, 3, 3, 3),
            "encoder.conv1.weight": (96, 3, 3, 3, 3),
            "encoder.head.2.weight": (32, 384, 3, 3, 3),
        },
        expected_sha256=VAE_EXPECTED_SHA256,
        verify_sha256=verify_sha256,
    )


_VENDOR_IMPORT_LOCK = threading.RLock()
_RUNTIME_MODULES_BY_ROOT: Dict[str, RuntimeModules] = {}
_VENDOR_PACKAGE_NAME = "_anima_native_ref_vendor"


def default_runtime_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "anima_edit"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_module_origin(module: ModuleType, root: Path, name: str) -> None:
    origin = getattr(module, "__file__", None)
    if not origin:
        namespace_paths = [
            Path(value).resolve() for value in (getattr(module, "__path__", None) or [])
        ]
        if namespace_paths and all(
            _is_relative_to(value, root) for value in namespace_paths
        ):
            return
        raise RuntimeImportError(
            f"Imported module {name!r} has no trusted filesystem origin; namespace paths={namespace_paths!r}."
        )
    resolved = Path(origin).resolve()
    if not _is_relative_to(resolved, root):
        raise RuntimeImportError(
            f"Import collision for {name!r}: resolved to {resolved}, outside pinned runtime {root}."
        )


def _install_vendored_package(root: Path) -> ModuleType:
    """Install the private vendor root without modifying ``sys.path``.

    The physical directory intentionally remains ``vendor/anima_edit`` so the
    audited source/config layout stays recognizable.  It is registered only as
    ``_anima_native_ref_vendor``; generic ``library`` and ``networks`` packages
    in the hosting ComfyUI process are therefore never read or overwritten.
    """

    existing = sys.modules.get(_VENDOR_PACKAGE_NAME)
    if existing is not None:
        _validate_module_origin(existing, root, _VENDOR_PACKAGE_NAME)
        return existing

    package_init = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _VENDOR_PACKAGE_NAME,
        package_init,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeImportError(
            f"Cannot create an import specification for vendored runtime {root}."
        )

    package = importlib.util.module_from_spec(spec)
    sys.modules[_VENDOR_PACKAGE_NAME] = package
    try:
        spec.loader.exec_module(package)
    except Exception:
        sys.modules.pop(_VENDOR_PACKAGE_NAME, None)
        raise
    _validate_module_origin(package, root, _VENDOR_PACKAGE_NAME)
    return package


def validate_vendored_runtime(
    runtime_root: os.PathLike[str] | str | None = None,
) -> str:
    """Verify every vendored source/config file against the release manifest."""

    root = (
        Path(runtime_root or default_runtime_root()).expanduser().resolve(strict=True)
    )
    manifest_path = root.parent / "VENDOR_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeImportError(
            f"Cannot read vendor manifest {manifest_path}: {exc}"
        ) from exc
    commit = manifest.get("source_commit")
    if commit != VENDORED_ANIMA_COMMIT:
        raise RuntimeImportError(
            f"Vendor commit mismatch: expected {VENDORED_ANIMA_COMMIT}, got {commit!r}."
        )
    overlay = manifest.get("source_overlay")
    if overlay != VENDORED_ANIMA_OVERLAY:
        raise RuntimeImportError(
            f"Vendor overlay mismatch: expected {VENDORED_ANIMA_OVERLAY}, got {overlay!r}."
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or int(manifest.get("file_count", -1)) != len(
        entries
    ):
        raise RuntimeImportError("Vendor manifest has an invalid file list/count.")

    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeImportError(
                "Vendor manifest contains a non-object file entry."
            )
        relative = str(entry.get("path", ""))
        candidate = (root / relative).resolve(strict=True)
        if not _is_relative_to(candidate, root) or not candidate.is_file():
            raise RuntimeImportError(f"Unsafe or missing vendored path: {relative!r}.")
        canonical = candidate.relative_to(root).as_posix()
        if canonical in listed:
            raise RuntimeImportError(f"Duplicate vendor manifest path: {canonical!r}.")
        listed.add(canonical)
        actual_size = candidate.stat().st_size
        if actual_size != int(entry.get("size", -1)):
            raise RuntimeImportError(
                f"Vendored file size mismatch for {canonical}: expected {entry.get('size')}, got {actual_size}."
            )
        actual_sha = _sha256_file(candidate, chunk_size=1024 * 1024)
        expected_sha = str(entry.get("sha256", ""))
        if actual_sha != expected_sha:
            raise RuntimeImportError(
                f"Vendored file SHA-256 mismatch for {canonical}: expected {expected_sha}, got {actual_sha}."
            )

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
    }
    extra = sorted(actual_files - listed)
    missing = sorted(listed - actual_files)
    if extra or missing:
        raise RuntimeImportError(
            f"Vendor tree/manifest mismatch: extra={extra!r}, missing={missing!r}."
        )
    return str(commit)


def load_runtime_modules(
    runtime_root: os.PathLike[str] | str | None = None,
) -> RuntimeModules:
    """Import the pinned runtime under its collision-free private namespace."""

    root = (
        Path(runtime_root or default_runtime_root()).expanduser().resolve(strict=True)
    )
    required = (
        root / "__init__.py",
        root / "library" / "__init__.py",
        root / "library" / "anima_utils.py",
        root / "library" / "anima_models.py",
        root / "library" / "anima_reference_binding.py",
        root / "library" / "anima_reference_router.py",
        root / "library" / "anima_text_conditioning.py",
        root / "library" / "strategy_anima.py",
        root / "library" / "hunyuan_image_utils.py",
        root / "library" / "qwen_image_autoencoder_kl.py",
        root / "configs" / "qwen3_06b" / "config.json",
        root / "configs" / "t5_old" / "spiece.model",
        root / "networks" / "__init__.py",
        root / "networks" / "loha.py",
        root / "networks" / "lokr.py",
        root / "LICENSE.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeImportError(
            f"Vendored runtime is incomplete; missing: {missing!r}"
        )

    cache_key = str(root)
    with _VENDOR_IMPORT_LOCK:
        validate_vendored_runtime(root)
        cached = _RUNTIME_MODULES_BY_ROOT.get(cache_key)
        if cached is not None:
            vendor_package = sys.modules.get(_VENDOR_PACKAGE_NAME)
            if vendor_package is None:
                raise RuntimeImportError(
                    f"Pinned runtime cache lost required package {_VENDOR_PACKAGE_NAME!r}."
                )
            _validate_module_origin(vendor_package, root, _VENDOR_PACKAGE_NAME)
            for name, module in (
                (f"{_VENDOR_PACKAGE_NAME}.library.anima_utils", cached.anima_utils),
                (f"{_VENDOR_PACKAGE_NAME}.library.anima_models", cached.anima_models),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.hunyuan_image_utils",
                    cached.hunyuan_image_utils,
                ),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.qwen_image_autoencoder_kl",
                    cached.qwen_image_autoencoder_kl,
                ),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.strategy_anima",
                    cached.strategy_anima,
                ),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning",
                    cached.anima_text_conditioning,
                ),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.anima_reference_binding",
                    cached.anima_reference_binding,
                ),
            ):
                _validate_module_origin(module, root, name)
            for name in (
                f"{_VENDOR_PACKAGE_NAME}.networks.loha",
                f"{_VENDOR_PACKAGE_NAME}.networks.lokr",
            ):
                module = sys.modules.get(name)
                if module is None:
                    raise RuntimeImportError(
                        f"Pinned runtime cache lost required module {name!r}."
                    )
                _validate_module_origin(module, root, name)
            return cached

        preexisting_vendor_modules = {
            name
            for name in sys.modules
            if name == _VENDOR_PACKAGE_NAME
            or name.startswith(f"{_VENDOR_PACKAGE_NAME}.")
        }
        try:
            _install_vendored_package(root)
            anima_utils = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.anima_utils"
            )
            anima_models = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.anima_models"
            )
            hunyuan_image_utils = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.hunyuan_image_utils"
            )
            qwen_vae = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.qwen_image_autoencoder_kl"
            )
            strategy_anima = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.strategy_anima"
            )
            anima_text_conditioning = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning"
            )
            anima_reference_binding = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.anima_reference_binding"
            )
            loha = importlib.import_module(f"{_VENDOR_PACKAGE_NAME}.networks.loha")
            lokr = importlib.import_module(f"{_VENDOR_PACKAGE_NAME}.networks.lokr")
        except Exception as exc:
            for name in tuple(sys.modules):
                if (
                    name == _VENDOR_PACKAGE_NAME
                    or name.startswith(f"{_VENDOR_PACKAGE_NAME}.")
                ) and name not in preexisting_vendor_modules:
                    sys.modules.pop(name, None)
            raise RuntimeImportError(
                f"Failed to import pinned Anima runtime from {root}: {exc}"
            ) from exc

        for name, module in (
            (f"{_VENDOR_PACKAGE_NAME}.library.anima_utils", anima_utils),
            (f"{_VENDOR_PACKAGE_NAME}.library.anima_models", anima_models),
            (
                f"{_VENDOR_PACKAGE_NAME}.library.hunyuan_image_utils",
                hunyuan_image_utils,
            ),
            (f"{_VENDOR_PACKAGE_NAME}.library.qwen_image_autoencoder_kl", qwen_vae),
            (f"{_VENDOR_PACKAGE_NAME}.library.strategy_anima", strategy_anima),
            (
                f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning",
                anima_text_conditioning,
            ),
            (
                f"{_VENDOR_PACKAGE_NAME}.library.anima_reference_binding",
                anima_reference_binding,
            ),
            (f"{_VENDOR_PACKAGE_NAME}.networks.loha", loha),
            (f"{_VENDOR_PACKAGE_NAME}.networks.lokr", lokr),
        ):
            _validate_module_origin(module, root, name)

        modules = RuntimeModules(
            anima_utils,
            anima_models,
            hunyuan_image_utils,
            qwen_vae,
            strategy_anima,
            anima_text_conditioning,
            anima_reference_binding,
        )
        _RUNTIME_MODULES_BY_ROOT[cache_key] = modules
        return modules


K = TypeVar("K")
V = TypeVar("V")
_MISSING = object()


class _BoundedLRU(Generic[K, V]):
    def __init__(self, max_entries: int) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._items: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K, default: Any = _MISSING) -> V | Any:
        if key not in self._items:
            return default
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: K, value: V) -> None:
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def resize(self, max_entries: int) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        try:
            import comfy.model_management as model_management  # type: ignore

            resolved = torch.device(model_management.get_torch_device())
        except Exception:
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _module_device(module: Any) -> torch.device:
    declared = getattr(module, "device", None)
    if declared is not None:
        return torch.device(declared)
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _module_dtype(module: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    declared = getattr(module, "dtype", None)
    if isinstance(declared, torch.dtype):
        return declared
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        return default


class AnimaNativeReferenceV2Runtime:
    """Cached E180 inference runtime for exactly two ordered reference images."""

    SUPPORTED_OFFLOAD_MODES = frozenset({"balanced", "high_vram", "text_encoder_cpu"})

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
        vae_chunk_size: int = DEFAULT_VAE_CHUNK_SIZE,
        vae_disable_cache: bool = True,
        prompt_cache_entries: int = 64,
        reference_cache_entries: int = 16,
        verify_checkpoint_sha256: bool = False,
    ) -> None:
        if offload_mode not in self.SUPPORTED_OFFLOAD_MODES:
            raise ValueError(
                f"Unsupported offload_mode {offload_mode!r}; expected one of {sorted(self.SUPPORTED_OFFLOAD_MODES)}."
            )
        if attn_mode != "torch":
            raise ValueError(
                "The formal E180 release runtime supports attn_mode='torch' only."
            )
        if int(vae_chunk_size) <= 0:
            raise ValueError("vae_chunk_size must be positive")

        self.lock = threading.RLock()
        self.checkpoint_info = validate_e180_checkpoint(
            checkpoint_path, verify_sha256=verify_checkpoint_sha256
        )
        self.checkpoint_path = self.checkpoint_info.fingerprint.path
        self.text_encoder_fingerprint = validate_qwen_text_encoder(
            text_encoder_path, verify_sha256=verify_checkpoint_sha256
        )
        self.vae_fingerprint = validate_qwen_image_vae(
            vae_path, verify_sha256=verify_checkpoint_sha256
        )
        self._release_sha256_verified = bool(verify_checkpoint_sha256)
        self.text_encoder_path = self.text_encoder_fingerprint.path
        self.vae_path = self.vae_fingerprint.path
        self.runtime_root = str(
            Path(runtime_root or default_runtime_root())
            .expanduser()
            .resolve(strict=True)
        )
        self.device = _resolve_device(device)
        if self.device.type != "cuda":
            raise AnimaRuntimeError(
                f"E180 publication runtime currently requires a CUDA BF16 device; got {self.device}."
            )
        if not torch.cuda.is_available():
            raise AnimaRuntimeError(
                "CUDA device was selected but torch.cuda.is_available() is false."
            )
        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            raise AnimaRuntimeError(
                f"CUDA device {self.device} does not report BF16 support."
            )

        self.offload_mode = offload_mode
        self.attention_mode = attn_mode
        self.vae_chunk_size = int(vae_chunk_size)
        self.vae_disable_cache = bool(vae_disable_cache)
        self.dtype = torch.bfloat16
        self._prompt_cache: _BoundedLRU[str, torch.Tensor] = _BoundedLRU(
            prompt_cache_entries
        )
        self._reference_cache: _BoundedLRU[str, torch.Tensor] = _BoundedLRU(
            reference_cache_entries
        )
        self._prompt_hits = 0
        self._prompt_misses = 0
        self._reference_hits = 0
        self._reference_misses = 0
        self._model_loads = 0
        self._closed = False

        self._modules = load_runtime_modules(self.runtime_root)
        self._anima: Any = None
        self._vae: Any = None
        self._text_encoder: Any = None
        self._qwen_tokenizer: Any = None
        self._t5_tokenizer: Any = None
        self._load_models_once()

    def _load_models_once(self) -> None:
        with self.lock:
            if self._closed:
                raise AnimaRuntimeError("Runtime is closed.")
            if self._anima is not None:
                return

            self._anima = self._modules.anima_utils.load_anima_model(
                self.device,
                self.checkpoint_path,
                self.attention_mode,
                True,  # split attention, matching formal inference
                self.device,
                self.dtype,
                False,  # fp8_scaled
                lora_weights_list=None,
                lora_multipliers=None,
                enable_ip_adapter=False,
                enable_native_reference_conditioning=None,
                native_reference_scale=DEFAULT_NATIVE_REFERENCE_SCALE,
            )
            self._anima.to(self.device, dtype=self.dtype)
            self._anima.eval().requires_grad_(False)
            if not bool(
                getattr(self._anima, "native_reference_conditioning_enabled", False)
            ):
                raise AnimaRuntimeError(
                    "Loaded model did not enable integrated native reference conditioning."
                )
            if getattr(self._anima, "native_reference_architecture", None) != "v2":
                raise AnimaRuntimeError(
                    "Loaded model is not native-reference architecture V2."
                )
            if int(getattr(self._anima, "native_reference_max_images", -1)) != 2:
                raise AnimaRuntimeError(
                    "Loaded model does not have exactly two ordered reference slots."
                )
            # Formal held-out evaluation used the generic (non-vectorized) path.
            self._anima.set_native_reference_fixed2ref_vectorized(False)

            text_device: torch.device | str = (
                self.device if self.offload_mode == "high_vram" else "cpu"
            )
            self._text_encoder, self._qwen_tokenizer = (
                self._modules.anima_utils.load_qwen3_text_encoder(
                    self.text_encoder_path,
                    dtype=self.dtype,
                    device=text_device,
                    lora_weights=None,
                    lora_multipliers=None,
                )
            )
            self._text_encoder.eval().requires_grad_(False)
            self._t5_tokenizer = self._modules.anima_utils.load_t5_tokenizer(None)

            vae_device: torch.device | str = (
                self.device if self.offload_mode == "high_vram" else "cpu"
            )
            self._vae = self._modules.qwen_image_autoencoder_kl.load_vae(
                self.vae_path,
                device=vae_device,
                disable_mmap=True,
                spatial_chunk_size=self.vae_chunk_size,
                disable_cache=self.vae_disable_cache,
            )
            self._vae.to(dtype=self.dtype)
            self._vae.eval().requires_grad_(False)
            self._model_loads += 1

    def _ensure_open(self) -> None:
        if self._closed or self._anima is None:
            raise AnimaRuntimeError("Runtime is closed or not loaded.")

    @staticmethod
    def _validate_sample_inputs(
        *,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance_scale: float,
        flow_shift: float,
        native_reference_scale: float,
        reference_max_area: int,
    ) -> None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise InputValidationError(
                f"Output width/height must be positive multiples of 16; got {width}x{height}."
            )
        if not 0 <= int(seed) <= 2**64 - 1:
            raise InputValidationError("seed must be in [0, 2**64-1].")
        if int(steps) <= 0:
            raise InputValidationError("steps must be positive.")
        for name, value in (
            ("guidance_scale", guidance_scale),
            ("flow_shift", flow_shift),
            ("native_reference_scale", native_reference_scale),
        ):
            if not math.isfinite(float(value)):
                raise InputValidationError(f"{name} must be finite.")
        if float(guidance_scale) < 0:
            raise InputValidationError("guidance_scale must be non-negative.")
        if float(flow_shift) <= 0:
            raise InputValidationError("flow_shift must be positive.")
        if float(native_reference_scale) < 0:
            raise InputValidationError("native_reference_scale must be non-negative.")
        if int(reference_max_area) <= 0:
            raise InputValidationError("reference_max_area must be positive.")

    @staticmethod
    def _comfy_image_to_pil(image: torch.Tensor, *, name: str) -> Image.Image:
        if not isinstance(image, torch.Tensor):
            raise InputValidationError(
                f"{name} must be a torch.Tensor in Comfy IMAGE format."
            )
        if image.ndim != 4 or int(image.shape[0]) != 1:
            raise InputValidationError(
                f"{name} must have shape [1,H,W,C]; initial release is fixed B=1, got {tuple(image.shape)}."
            )
        if int(image.shape[1]) <= 0 or int(image.shape[2]) <= 0:
            raise InputValidationError(f"{name} has an empty spatial dimension.")
        channels = int(image.shape[3])
        if channels not in (1, 3, 4):
            raise InputValidationError(
                f"{name} must have 1, 3, or 4 channels; got {channels}."
            )

        sample = image[0].detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(sample).all()):
            raise InputValidationError(f"{name} contains NaN or Inf.")
        if channels == 1:
            sample = sample.expand(-1, -1, 3)
        else:
            sample = sample[..., :3]
        sample = sample.clamp(0.0, 1.0)
        array = (sample * 255.0).round().to(torch.uint8).contiguous().numpy()
        return Image.fromarray(array).convert("RGB")

    def _preprocess_reference(
        self,
        image: torch.Tensor,
        *,
        name: str,
        reference_max_area: int,
    ) -> Tuple[Image.Image, str]:
        pil = self._comfy_image_to_pil(image, name=name).convert("RGB")
        if pil.width * pil.height > int(reference_max_area):
            scale = math.sqrt(int(reference_max_area) / float(pil.width * pil.height))
            pil = pil.resize(
                (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                Image.Resampling.LANCZOS,
            )

        multiple_of = int(self._modules.qwen_image_autoencoder_kl.SCALE_FACTOR) * int(
            self._anima.patch_spatial
        )
        width = (pil.width // multiple_of) * multiple_of
        height = (pil.height // multiple_of) * multiple_of
        if width <= 0 or height <= 0:
            raise InputValidationError(
                f"{name} is too small after alignment ({pil.width}x{pil.height}); "
                f"both sides must be at least {multiple_of}."
            )
        left = (pil.width - width) // 2
        top = (pil.height - height) // 2
        pil = pil.crop((left, top, left + width, top + height))

        pixels = pil.tobytes()
        digest = hashlib.sha256()
        digest.update(b"anima-ref-v2-e180-reference-latent-v1\0")
        digest.update(
            f"{pil.width}x{pil.height}|{multiple_of}|{reference_max_area}|".encode(
                "ascii"
            )
        )
        digest.update(self.vae_fingerprint.path.encode("utf-8"))
        digest.update(str(self.vae_fingerprint.size).encode("ascii"))
        digest.update(str(self.vae_fingerprint.mtime_ns).encode("ascii"))
        digest.update(pixels)
        return pil, digest.hexdigest()

    def _move_vae_to(self, device: torch.device | str) -> None:
        self._vae.to(device)

    def _offload_vae_if_needed(self) -> None:
        if self.offload_mode != "high_vram":
            self._vae.to("cpu")

    def _offload_text_encoder_if_needed(self) -> None:
        if self.offload_mode != "high_vram":
            self._text_encoder.to("cpu")

    def _encode_reference_pair(
        self,
        reference_image_1: torch.Tensor,
        reference_image_2: torch.Tensor,
        *,
        reference_max_area: int,
    ) -> list[list[torch.Tensor]]:
        prepared = [
            self._preprocess_reference(
                reference_image_1,
                name="reference_image_1",
                reference_max_area=reference_max_area,
            ),
            self._preprocess_reference(
                reference_image_2,
                name="reference_image_2",
                reference_max_area=reference_max_area,
            ),
        ]
        cpu_latents: list[Optional[torch.Tensor]] = [None, None]
        misses: list[int] = []
        for index, (_pil, key) in enumerate(prepared):
            cached = self._reference_cache.get(key)
            if cached is _MISSING:
                self._reference_misses += 1
                misses.append(index)
            else:
                self._reference_hits += 1
                cpu_latents[index] = cached

        if misses:
            self._move_vae_to(self.device)
            try:
                vae_dtype = _module_dtype(self._vae, self.dtype)
                for index in misses:
                    pil, key = prepared[index]
                    array = np.asarray(pil, dtype=np.uint8).copy()
                    pixels = (
                        torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
                    )
                    pixels = (
                        pixels.mul_(2.0)
                        .sub_(1.0)
                        .unsqueeze(0)
                        .to(device=self.device, dtype=vae_dtype)
                    )
                    latent = self._vae.encode_pixels_to_latents(pixels).to(
                        device="cpu", dtype=self.dtype
                    )
                    if latent.ndim == 5:
                        latent = latent.squeeze(2)
                    latent = latent.contiguous()
                    self._reference_cache.put(key, latent)
                    cpu_latents[index] = latent
            finally:
                self._offload_vae_if_needed()

        if any(latent is None for latent in cpu_latents):
            raise AssertionError("Reference latent cache/encode bookkeeping failed.")
        # Preserve physical order. Slot semantics are supplied separately as [0,1].
        return [
            [latent.to(device=self.device, dtype=self.dtype) for latent in cpu_latents]
        ]  # type: ignore[union-attr]

    def _encode_text_uncached(self, text: str) -> torch.Tensor:
        # Deliberately do not use the historical process_escape implementation:
        # it corrupts ordinary non-ASCII prompts. Tokenizers receive Unicode as-is.
        qwen_encoding = self._qwen_tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=DEFAULT_TEXT_MAX_LENGTH,
        )
        t5_encoding = self._t5_tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=DEFAULT_TEXT_MAX_LENGTH,
        )
        encoder_device = _module_device(self._text_encoder)
        qwen_input_ids = qwen_encoding["input_ids"].to(encoder_device)
        qwen_attention_mask = qwen_encoding["attention_mask"].to(encoder_device)
        source_hidden_states = self._text_encoder(
            input_ids=qwen_input_ids,
            attention_mask=qwen_attention_mask,
        ).last_hidden_state
        source_hidden_states[~qwen_attention_mask.bool()] = 0

        target_input_ids = t5_encoding["input_ids"].to(self.device)
        target_attention_mask = t5_encoding["attention_mask"].to(self.device)
        crossattn = self._anima._preprocess_text_embeds(
            source_hidden_states=source_hidden_states.to(self.device),
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            source_attention_mask=qwen_attention_mask.to(self.device),
        )
        crossattn[~target_attention_mask.bool()] = 0
        return crossattn.to(device="cpu", dtype=self.dtype).contiguous()

    def _get_text_embeddings(
        self, prompt: str, negative_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise InputValidationError("prompt and negative_prompt must be strings.")

        values: Dict[str, torch.Tensor] = {}
        missing: list[str] = []
        seen: set[str] = set()
        for text in (prompt, negative_prompt):
            if text in seen:
                continue
            seen.add(text)
            cached = self._prompt_cache.get(text)
            if cached is _MISSING:
                self._prompt_misses += 1
                missing.append(text)
            else:
                self._prompt_hits += 1
                values[text] = cached

        if missing:
            target_device: torch.device | str = (
                "cpu" if self.offload_mode == "text_encoder_cpu" else self.device
            )
            self._text_encoder.to(target_device)
            try:
                for text in missing:
                    encoded = self._encode_text_uncached(text)
                    self._prompt_cache.put(text, encoded)
                    values[text] = encoded
            finally:
                self._offload_text_encoder_if_needed()

        return values[prompt], values[negative_prompt]

    def _denoise(
        self,
        *,
        context: torch.Tensor,
        negative_context: torch.Tensor,
        reference_latents: list[list[torch.Tensor]],
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
        shape = (
            1,
            int(self._modules.anima_models.Anima.LATENT_CHANNELS),
            1,
            height // 8,
            width // 8,
        )
        # Match diffusers.randn_tensor in the formal CLI: a CPU generator makes
        # the random tensor on CPU in the requested BF16 dtype before transfer.
        latents = torch.randn(
            shape, generator=generator, device="cpu", dtype=self.dtype
        ).to(self.device)
        padding_mask = torch.zeros(
            1, 1, height // 8, width // 8, dtype=self.dtype, device=self.device
        )
        embed = context.to(self.device, dtype=self.dtype)
        negative_embed = negative_context.to(self.device, dtype=self.dtype)
        timesteps, sigmas = self._modules.hunyuan_image_utils.get_timesteps_sigmas(
            int(steps), float(flow_shift), self.device
        )
        timesteps = (timesteps / 1000).to(self.device, dtype=self.dtype)
        # Match anima_minimal_inference.generate_body exactly: every scale other
        # than one evaluates the negative branch (including values below one).
        do_cfg = float(guidance_scale) != 1.0
        total = int(len(timesteps))
        for index, timestep in enumerate(timesteps):
            if interrupt_callback is not None:
                interrupt_callback()
            timestep_batch = timestep.expand(latents.shape[0])
            noise_pred = self._anima(
                latents,
                timestep_batch,
                embed,
                padding_mask=padding_mask,
                reference_latents=reference_latents,
                reference_slot_ids=[list(ORDERED_SLOT_IDS[0])],
                reference_t_offset_scale=10,
                ip_adapter_latents=None,
                ip_adapter_embeds=None,
                use_ip_adapter=False,
                use_reference_sequence=True,
            )
            if do_cfg:
                uncond = self._anima(
                    latents,
                    timestep_batch,
                    negative_embed,
                    padding_mask=padding_mask,
                    reference_latents=reference_latents,
                    reference_slot_ids=[list(ORDERED_SLOT_IDS[0])],
                    reference_t_offset_scale=10,
                    ip_adapter_latents=None,
                    ip_adapter_embeds=None,
                    use_ip_adapter=False,
                    use_reference_sequence=True,
                )
                noise_pred = uncond + float(guidance_scale) * (noise_pred - uncond)
            latents = self._modules.hunyuan_image_utils.step(
                latents, noise_pred, sigmas, index
            ).to(latents.dtype)
            if progress_callback is not None:
                progress_callback(index + 1, total)
        return latents

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        self._move_vae_to(self.device)
        try:
            pixels = self._vae.decode_to_pixels(
                latent.to(
                    device=self.device, dtype=_module_dtype(self._vae, self.dtype)
                )
            )
            if pixels.ndim == 5:
                pixels = pixels.squeeze(2)
            if pixels.ndim != 4 or int(pixels.shape[0]) != 1:
                raise AnimaRuntimeError(
                    f"Unexpected VAE output shape: {tuple(pixels.shape)}"
                )
            pixels = pixels.to(device="cpu", dtype=torch.float32)
        finally:
            self._offload_vae_if_needed()

        if not bool(torch.isfinite(pixels).all()):
            raise AnimaRuntimeError("Decoded image contains NaN or Inf.")
        # Raw generated image only: [1,C,H,W] [-1,1] -> Comfy [1,H,W,C] [0,1].
        return ((pixels.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(0, 2, 3, 1).contiguous()

    def generate(
        self,
        reference_image_1: torch.Tensor,
        reference_image_2: torch.Tensor,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: int = 0,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        flow_shift: float = DEFAULT_FLOW_SHIFT,
        native_reference_scale: float = DEFAULT_NATIVE_REFERENCE_SCALE,
        reference_max_area: int = DEFAULT_REFERENCE_MAX_AREA,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        interrupt_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Generate one raw Comfy IMAGE tensor from two ordered references."""

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
            self._anima.set_native_reference_fixed2ref_vectorized(False)
            # Critical for a cached loader: scale is mutable per generation.
            self._anima.set_native_reference_scale(float(native_reference_scale))

            reference_latents: Optional[list[list[torch.Tensor]]] = None
            latent: Optional[torch.Tensor] = None
            try:
                reference_latents = self._encode_reference_pair(
                    reference_image_1,
                    reference_image_2,
                    reference_max_area=int(reference_max_area),
                )
                context, negative_context = self._get_text_embeddings(
                    prompt, negative_prompt
                )
                latent = self._denoise(
                    context=context,
                    negative_context=negative_context,
                    reference_latents=reference_latents,
                    width=int(width),
                    height=int(height),
                    seed=int(seed),
                    steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    flow_shift=float(flow_shift),
                    progress_callback=progress_callback,
                    interrupt_callback=interrupt_callback,
                )
                # Drop GPU reference streams before VAE decode.
                reference_latents = None
                return self._decode_latent(latent)
            finally:
                reference_latents = None
                latent = None

    def cache_statistics(self) -> CacheStatistics:
        with self.lock:
            return CacheStatistics(
                model_loads=self._model_loads,
                prompt_hits=self._prompt_hits,
                prompt_misses=self._prompt_misses,
                reference_hits=self._reference_hits,
                reference_misses=self._reference_misses,
                prompt_entries=len(self._prompt_cache),
                reference_entries=len(self._reference_cache),
            )

    def verify_release_sha256(self) -> None:
        """Verify all three published weight files once for this runtime."""

        with self.lock:
            self._ensure_open()
            if self._release_sha256_verified:
                return
            self.checkpoint_info = validate_e180_checkpoint(
                self.checkpoint_path, verify_sha256=True
            )
            validate_qwen_text_encoder(self.text_encoder_path, verify_sha256=True)
            validate_qwen_image_vae(self.vae_path, verify_sha256=True)
            self._release_sha256_verified = True

    @property
    def metadata(self) -> Mapping[str, str]:
        """Read-only checkpoint metadata validated at loader construction."""

        return self.checkpoint_info.metadata

    @property
    def fingerprint(self) -> FileFingerprint:
        """Validated E180 checkpoint fingerprint used by the model cache."""

        return self.checkpoint_info.fingerprint

    @property
    def model_fingerprints(self) -> Mapping[str, FileFingerprint]:
        return {
            "checkpoint": self.checkpoint_info.fingerprint,
            "text_encoder": self.text_encoder_fingerprint,
            "vae": self.vae_fingerprint,
        }

    def clear_condition_caches(self) -> None:
        with self.lock:
            self._prompt_cache.clear()
            self._reference_cache.clear()

    def set_condition_cache_limits(
        self, *, prompt_cache_entries: int, reference_cache_entries: int
    ) -> None:
        """Resize policy-only caches without reconstructing heavyweight models."""

        with self.lock:
            self._prompt_cache.resize(prompt_cache_entries)
            self._reference_cache.resize(reference_cache_entries)

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.clear_condition_caches()
            for name in ("_text_encoder", "_vae", "_anima"):
                module = getattr(self, name, None)
                if module is not None:
                    try:
                        module.to("cpu")
                    except Exception:
                        pass
                setattr(self, name, None)
            self._qwen_tokenizer = None
            self._t5_tokenizer = None
            if hasattr(self, "_tokenize_strategy"):
                self._tokenize_strategy = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(
        self,
    ) -> None:  # pragma: no cover - best-effort interpreter-shutdown path
        try:
            self.close()
        except Exception:
            pass


class AnimaNativeReferenceV4Runtime(AnimaNativeReferenceV2Runtime):
    """Final 49k competitive-router runtime for one or two logical ref slots."""

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
        vae_chunk_size: int = DEFAULT_VAE_CHUNK_SIZE,
        vae_disable_cache: bool = True,
        prompt_cache_entries: int = 64,
        reference_cache_entries: int = 16,
        verify_checkpoint_sha256: bool = False,
    ) -> None:
        if offload_mode not in self.SUPPORTED_OFFLOAD_MODES:
            raise ValueError(
                f"Unsupported offload_mode {offload_mode!r}; "
                f"expected one of {sorted(self.SUPPORTED_OFFLOAD_MODES)}."
            )
        if attn_mode != "torch":
            raise ValueError(
                "The final V4 fixed-one/two carrier supports attn_mode='torch' only."
            )
        if int(vae_chunk_size) <= 0:
            raise ValueError("vae_chunk_size must be positive")

        self.lock = threading.RLock()
        self.checkpoint_info = validate_v4_checkpoint(
            checkpoint_path, verify_sha256=verify_checkpoint_sha256
        )
        self.checkpoint_path = self.checkpoint_info.fingerprint.path
        self.text_encoder_fingerprint = validate_qwen_text_encoder(
            text_encoder_path, verify_sha256=verify_checkpoint_sha256
        )
        self.vae_fingerprint = validate_qwen_image_vae(
            vae_path, verify_sha256=verify_checkpoint_sha256
        )
        self._release_sha256_verified = bool(verify_checkpoint_sha256)
        self.text_encoder_path = self.text_encoder_fingerprint.path
        self.vae_path = self.vae_fingerprint.path
        self.runtime_root = str(
            Path(runtime_root or default_runtime_root())
            .expanduser()
            .resolve(strict=True)
        )
        self.device = _resolve_device(device)
        if self.device.type != "cuda":
            raise AnimaRuntimeError(
                "The final V4 publication runtime requires a CUDA BF16 device; "
                f"got {self.device}."
            )
        if not torch.cuda.is_available():
            raise AnimaRuntimeError(
                "CUDA device was selected but torch.cuda.is_available() is false."
            )
        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            raise AnimaRuntimeError(
                f"CUDA device {self.device} does not report BF16 support."
            )

        self.offload_mode = offload_mode
        self.attention_mode = attn_mode
        self.vae_chunk_size = int(vae_chunk_size)
        self.vae_disable_cache = bool(vae_disable_cache)
        self.dtype = torch.bfloat16
        self._prompt_cache: _BoundedLRU[str, RawPromptEncoding] = _BoundedLRU(
            prompt_cache_entries
        )
        self._reference_cache: _BoundedLRU[str, torch.Tensor] = _BoundedLRU(
            reference_cache_entries
        )
        self._prompt_hits = 0
        self._prompt_misses = 0
        self._reference_hits = 0
        self._reference_misses = 0
        self._model_loads = 0
        self._closed = False

        self._modules = load_runtime_modules(self.runtime_root)
        self._anima: Any = None
        self._vae: Any = None
        self._text_encoder: Any = None
        self._qwen_tokenizer: Any = None
        self._t5_tokenizer: Any = None
        self._tokenize_strategy: Any = None
        self._load_models_once()

    def _load_models_once(self) -> None:
        with self.lock:
            if self._closed:
                raise AnimaRuntimeError("Runtime is closed.")
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
                native_reference_scale=DEFAULT_NATIVE_REFERENCE_SCALE,
            )
            self._anima.to(self.device, dtype=self.dtype)
            self._anima.eval().requires_grad_(False)
            if not bool(
                getattr(self._anima, "native_reference_conditioning_enabled", False)
            ):
                raise AnimaRuntimeError(
                    "Loaded final V4 model did not enable integrated reference conditioning."
                )
            if getattr(self._anima, "native_reference_architecture", None) != "v2":
                raise AnimaRuntimeError(
                    "Loaded final V4 model is not native-reference architecture V2."
                )
            if int(getattr(self._anima, "native_reference_max_images", -1)) != 2:
                raise AnimaRuntimeError(
                    "Loaded final V4 model does not expose exactly two logical slots."
                )
            if (
                getattr(self._anima, "native_reference_routing_mode", None)
                != "competitive_text_slot_v1"
            ):
                raise AnimaRuntimeError(
                    "Loaded final V4 model did not construct competitive_text_slot_v1."
                )
            if float(
                getattr(self._anima, "native_reference_routing_alpha", float("nan"))
            ) != 1.0:
                raise AnimaRuntimeError(
                    "Loaded final V4 model did not retain routing_alpha=1.0."
                )
            if not bool(getattr(self._anima, "use_llm_adapter", False)):
                raise AnimaRuntimeError(
                    "Final V4 requires the native Qwen-to-neutral-T5 LLM adapter."
                )
            fixed12_setter = getattr(
                self._anima, "set_native_reference_fixed12ref_vectorized", None
            )
            if fixed12_setter is None:
                raise AnimaRuntimeError(
                    "Vendored model lacks set_native_reference_fixed12ref_vectorized()."
                )
            fixed12_setter(True)

            text_device: torch.device | str = (
                self.device if self.offload_mode == "high_vram" else "cpu"
            )
            self._text_encoder, self._qwen_tokenizer = (
                self._modules.anima_utils.load_qwen3_text_encoder(
                    self.text_encoder_path,
                    dtype=self.dtype,
                    device=text_device,
                    lora_weights=None,
                    lora_multipliers=None,
                )
            )
            self._text_encoder.eval().requires_grad_(False)
            self._t5_tokenizer = self._modules.anima_utils.load_t5_tokenizer(None)
            self._tokenize_strategy = (
                self._modules.strategy_anima.AnimaTokenizeStrategy(
                    qwen3_tokenizer=self._qwen_tokenizer,
                    t5_tokenizer=self._t5_tokenizer,
                    qwen3_max_length=DEFAULT_TEXT_MAX_LENGTH,
                    t5_max_length=DEFAULT_TEXT_MAX_LENGTH,
                )
            )

            vae_device: torch.device | str = (
                self.device if self.offload_mode == "high_vram" else "cpu"
            )
            self._vae = self._modules.qwen_image_autoencoder_kl.load_vae(
                self.vae_path,
                device=vae_device,
                disable_mmap=True,
                spatial_chunk_size=self.vae_chunk_size,
                disable_cache=self.vae_disable_cache,
            )
            self._vae.to(dtype=self.dtype)
            self._vae.eval().requires_grad_(False)
            self._model_loads += 1

    @staticmethod
    def _normalise_reference_request(
        reference_images: Sequence[torch.Tensor],
        reference_slot_ids: Sequence[int],
        *,
        preprocess_mode: str,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        images = tuple(reference_images)
        slots = tuple(int(slot_id) for slot_id in reference_slot_ids)
        if len(images) not in (1, 2):
            raise InputValidationError(
                f"Final V4 requires one or two physical reference images; got {len(images)}."
            )
        if len(slots) != len(images):
            raise InputValidationError(
                "reference_slot_ids must contain one logical slot per physical image: "
                f"{len(slots)} != {len(images)}."
            )
        if len(set(slots)) != len(slots):
            raise InputValidationError(
                f"reference_slot_ids must be unique; got {slots!r}."
            )
        invalid = [slot_id for slot_id in slots if slot_id not in (0, 1)]
        if invalid:
            raise InputValidationError(
                f"reference_slot_ids must be 0 or 1; got invalid values {invalid!r}."
            )
        if preprocess_mode not in V4_PREPROCESS_MODES:
            raise InputValidationError(
                f"Unknown preprocess_mode {preprocess_mode!r}; "
                f"expected one of {sorted(V4_PREPROCESS_MODES)}."
            )
        if len(images) == 2 and preprocess_mode != "independent_reference":
            raise InputValidationError(
                "Two-reference V4 generation must use independent_reference "
                "preprocessing; match_output_edit is a one-reference edit path."
            )
        return images, slots

    def _preprocess_reference_v4(
        self,
        image: torch.Tensor,
        *,
        name: str,
        preprocess_mode: str,
        reference_max_area: int,
        output_width: int,
        output_height: int,
    ) -> tuple[Image.Image, str]:
        pil = self._comfy_image_to_pil(image, name=name).convert("RGB")
        multiple_of = int(self._modules.qwen_image_autoencoder_kl.SCALE_FACTOR) * int(
            self._anima.patch_spatial
        )
        target_size_hw = (
            (int(output_height), int(output_width))
            if preprocess_mode == "match_output_edit"
            else None
        )
        try:
            prepared = self._modules.strategy_anima.preprocess_anima_reference_image(
                pil,
                max_area=int(reference_max_area),
                multiple_of=multiple_of,
                target_size_hw=target_size_hw,
                flipped=False,
            )
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                f"{name} preprocessing failed for mode {preprocess_mode!r}: {exc}"
            ) from exc

        pixels = prepared.tobytes()
        digest = hashlib.sha256()
        digest.update(b"anima-ref-v4-final49k-reference-latent-v1\0")
        digest.update(
            (
                f"mode={preprocess_mode}|prepared={prepared.width}x{prepared.height}|"
                f"output={output_width}x{output_height}|multiple={multiple_of}|"
                f"max_area={reference_max_area}|"
            ).encode("ascii")
        )
        digest.update(self.vae_fingerprint.path.encode("utf-8"))
        digest.update(str(self.vae_fingerprint.size).encode("ascii"))
        digest.update(str(self.vae_fingerprint.mtime_ns).encode("ascii"))
        digest.update(pixels)
        return prepared, digest.hexdigest()

    def _encode_references(
        self,
        reference_images: Sequence[torch.Tensor],
        *,
        preprocess_mode: str,
        reference_max_area: int,
        output_width: int,
        output_height: int,
    ) -> list[list[torch.Tensor]]:
        prepared = [
            self._preprocess_reference_v4(
                image,
                name=f"reference_image_{index + 1}",
                preprocess_mode=preprocess_mode,
                reference_max_area=reference_max_area,
                output_width=output_width,
                output_height=output_height,
            )
            for index, image in enumerate(reference_images)
        ]
        cpu_latents: list[Optional[torch.Tensor]] = [None] * len(prepared)
        misses: list[int] = []
        for index, (_pil, key) in enumerate(prepared):
            cached = self._reference_cache.get(key)
            if cached is _MISSING:
                self._reference_misses += 1
                misses.append(index)
            else:
                self._reference_hits += 1
                cpu_latents[index] = cached

        if misses:
            self._move_vae_to(self.device)
            try:
                vae_dtype = _module_dtype(self._vae, self.dtype)
                for index in misses:
                    pil, key = prepared[index]
                    array = np.asarray(pil, dtype=np.uint8).copy()
                    pixels = (
                        torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
                    )
                    pixels = (
                        pixels.mul_(2.0)
                        .sub_(1.0)
                        .unsqueeze(0)
                        .to(device=self.device, dtype=vae_dtype)
                    )
                    latent = self._vae.encode_pixels_to_latents(pixels).to(
                        device="cpu", dtype=self.dtype
                    )
                    if latent.ndim == 5:
                        latent = latent.squeeze(2)
                    latent = latent.contiguous()
                    self._reference_cache.put(key, latent)
                    cpu_latents[index] = latent
            finally:
                self._offload_vae_if_needed()

        if any(latent is None for latent in cpu_latents):
            raise AssertionError("Reference latent cache/encode bookkeeping failed.")
        # Never fabricate or duplicate a missing second reference.  One physical
        # latent is the complete one-reference carrier.
        return [
            [
                latent.to(device=self.device, dtype=self.dtype)
                for latent in cpu_latents
                if latent is not None
            ]
        ]

    def _encode_text_uncached_v4(self, text: str) -> RawPromptEncoding:
        # No process_escape: Unicode reaches both tokenizers byte-for-byte.
        tokens = self._tokenize_strategy.tokenize(text)
        if len(tokens) != 4:
            raise AnimaRuntimeError(
                "AnimaTokenizeStrategy did not return "
                "[Qwen IDs, Qwen mask, T5 IDs, T5 mask]."
            )
        qwen_input_ids, qwen_attention_mask, target_input_ids, target_attention_mask = (
            tokens
        )
        encoder_device = _module_device(self._text_encoder)
        qwen_input_ids = qwen_input_ids.to(encoder_device)
        qwen_attention_mask = qwen_attention_mask.to(encoder_device)
        source_hidden_states = self._text_encoder(
            input_ids=qwen_input_ids,
            attention_mask=qwen_attention_mask,
        ).last_hidden_state
        source_hidden_states[~qwen_attention_mask.bool()] = 0
        return RawPromptEncoding(
            raw_qwen_context=source_hidden_states.to(
                device="cpu", dtype=self.dtype
            ).contiguous(),
            source_attention_mask=qwen_attention_mask.to(device="cpu").contiguous(),
            target_input_ids=target_input_ids.to(
                device="cpu", dtype=torch.long
            ).contiguous(),
            target_attention_mask=target_attention_mask.to(
                device="cpu"
            ).contiguous(),
        )

    def _get_raw_text_encodings(
        self, prompt: str, negative_prompt: str
    ) -> tuple[RawPromptEncoding, RawPromptEncoding]:
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise InputValidationError("prompt and negative_prompt must be strings.")

        values: Dict[str, RawPromptEncoding] = {}
        missing: list[str] = []
        seen: set[str] = set()
        for text in (prompt, negative_prompt):
            if text in seen:
                continue
            seen.add(text)
            cached = self._prompt_cache.get(text)
            if cached is _MISSING:
                self._prompt_misses += 1
                missing.append(text)
            else:
                self._prompt_hits += 1
                values[text] = cached

        if missing:
            target_device: torch.device | str = (
                "cpu" if self.offload_mode == "text_encoder_cpu" else self.device
            )
            self._text_encoder.to(target_device)
            try:
                for text in missing:
                    encoded = self._encode_text_uncached_v4(text)
                    self._prompt_cache.put(text, encoded)
                    values[text] = encoded
            finally:
                self._offload_text_encoder_if_needed()

        return values[prompt], values[negative_prompt]

    def _prepare_text_conditionings(
        self,
        prompt: str,
        negative_prompt: str,
        *,
        reference_slot_ids: tuple[int, ...],
    ) -> tuple[Any, Any]:
        positive_raw, negative_raw = self._get_raw_text_encodings(
            prompt, negative_prompt
        )
        slot_rows = [list(reference_slot_ids)]
        requires_binding = (
            self._modules.anima_text_conditioning.reference_router_requires_clause_masks(
                self._anima,
                slot_rows,
                use_reference_sequence=True,
            )
        )
        if not requires_binding:
            raise AnimaRuntimeError(
                "The final V4 checkpoint unexpectedly did not require clause masks."
            )
        prepare = (
            self._modules.anima_text_conditioning.prepare_anima_prompt_conditioning
        )
        try:
            positive = prepare(
                prompt,
                positive_raw.as_sequence(),
                tokenize_strategy=self._tokenize_strategy,
                device=self.device,
                dtype=self.dtype,
                reference_slot_ids=slot_rows,
                require_reference_binding=True,
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
            raise InputValidationError(
                "Invalid V4 reference prompt. Use explicit canonical Image 1/"
                f"Image 2 clauses matching the selected slots: {exc}"
            ) from exc
        masks = positive.reference_clause_masks
        expected_shape = (1, 2, DEFAULT_TEXT_MAX_LENGTH)
        if (
            masks is None
            or masks.dtype is not torch.bool
            or tuple(masks.shape) != expected_shape
        ):
            raise AnimaRuntimeError(
                "V4 reference_clause_masks contract violation: "
                f"expected bool {expected_shape}, got "
                f"{None if masks is None else (masks.dtype, tuple(masks.shape))}."
            )
        occupied = tuple(bool(masks[0, slot].any()) for slot in range(2))
        expected_occupied = tuple(slot in reference_slot_ids for slot in range(2))
        if occupied != expected_occupied:
            raise AnimaRuntimeError(
                "V4 clause-mask occupancy disagrees with logical slots: "
                f"{occupied!r} != {expected_occupied!r}."
            )
        return positive, negative

    def _denoise_v4(
        self,
        *,
        positive_conditioning: Any,
        negative_conditioning: Any,
        reference_latents: list[list[torch.Tensor]],
        reference_slot_ids: tuple[int, ...],
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
        shape = (
            1,
            int(self._modules.anima_models.Anima.LATENT_CHANNELS),
            1,
            height // 8,
            width // 8,
        )
        latents = torch.randn(
            shape, generator=generator, device="cpu", dtype=self.dtype
        ).to(self.device)
        padding_mask = torch.zeros(
            1, 1, height // 8, width // 8, dtype=self.dtype, device=self.device
        )
        timesteps, sigmas = self._modules.hunyuan_image_utils.get_timesteps_sigmas(
            int(steps), float(flow_shift), self.device
        )
        timesteps = (timesteps / 1000).to(self.device, dtype=self.dtype)
        use_llm_adapter = bool(getattr(self._anima, "use_llm_adapter", False))
        positive_text_kwargs = positive_conditioning.model_text_kwargs(
            self.device,
            self.dtype,
            use_llm_adapter=use_llm_adapter,
        )
        # Standard negative text stays negative.  Only raw router context/mask
        # and positive reference clauses are shared from the positive branch.
        negative_text_kwargs = negative_conditioning.model_text_kwargs(
            self.device,
            self.dtype,
            router_conditioning=positive_conditioning,
            use_llm_adapter=use_llm_adapter,
        )
        slot_rows = [list(reference_slot_ids)]
        do_cfg = float(guidance_scale) != 1.0
        total = int(len(timesteps))
        common_kwargs = {
            "padding_mask": padding_mask,
            "reference_latents": reference_latents,
            "reference_slot_ids": slot_rows,
            "reference_t_offset_scale": 10,
            "ip_adapter_latents": None,
            "ip_adapter_embeds": None,
            "use_ip_adapter": False,
            "use_reference_sequence": True,
        }
        for index, timestep in enumerate(timesteps):
            if interrupt_callback is not None:
                interrupt_callback()
            timestep_batch = timestep.expand(latents.shape[0])
            noise_pred = self._anima(
                latents,
                timestep_batch,
                **common_kwargs,
                **positive_text_kwargs,
            )
            if do_cfg:
                uncond = self._anima(
                    latents,
                    timestep_batch,
                    **common_kwargs,
                    **negative_text_kwargs,
                )
                noise_pred = uncond + float(guidance_scale) * (noise_pred - uncond)
            latents = self._modules.hunyuan_image_utils.step(
                latents, noise_pred, sigmas, index
            ).to(latents.dtype)
            if progress_callback is not None:
                progress_callback(index + 1, total)
        return latents

    def generate(
        self,
        reference_images: Sequence[torch.Tensor],
        reference_slot_ids: Sequence[int],
        prompt: str,
        *,
        negative_prompt: str = "",
        preprocess_mode: str = "independent_reference",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: int = 0,
        steps: int = V4_DEFAULT_STEPS,
        guidance_scale: float = V4_DEFAULT_GUIDANCE_SCALE,
        flow_shift: float = DEFAULT_FLOW_SHIFT,
        native_reference_scale: float = DEFAULT_NATIVE_REFERENCE_SCALE,
        reference_max_area: int = DEFAULT_REFERENCE_MAX_AREA,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        interrupt_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Generate from exactly one or two real references and explicit slots."""

        images, slots = self._normalise_reference_request(
            reference_images,
            reference_slot_ids,
            preprocess_mode=preprocess_mode,
        )
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
            self._anima.set_native_reference_scale(float(native_reference_scale))

            reference_latents: Optional[list[list[torch.Tensor]]] = None
            latent: Optional[torch.Tensor] = None
            try:
                # Fail closed on prompt/slot ambiguity before entering the VAE or
                # denoising path.  This also prevents a cached stale mask from a
                # different logical-slot request.
                positive, negative = self._prepare_text_conditionings(
                    prompt,
                    negative_prompt,
                    reference_slot_ids=slots,
                )
                reference_latents = self._encode_references(
                    images,
                    preprocess_mode=preprocess_mode,
                    reference_max_area=int(reference_max_area),
                    output_width=int(width),
                    output_height=int(height),
                )
                latent = self._denoise_v4(
                    positive_conditioning=positive,
                    negative_conditioning=negative,
                    reference_latents=reference_latents,
                    reference_slot_ids=slots,
                    width=int(width),
                    height=int(height),
                    seed=int(seed),
                    steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    flow_shift=float(flow_shift),
                    progress_callback=progress_callback,
                    interrupt_callback=interrupt_callback,
                )
                reference_latents = None
                return self._decode_latent(latent)
            finally:
                reference_latents = None
                latent = None

    def verify_release_sha256(self) -> None:
        with self.lock:
            self._ensure_open()
            if self._release_sha256_verified:
                return
            self.checkpoint_info = validate_v4_checkpoint(
                self.checkpoint_path, verify_sha256=True
            )
            validate_qwen_text_encoder(self.text_encoder_path, verify_sha256=True)
            validate_qwen_image_vae(self.vae_path, verify_sha256=True)
            self._release_sha256_verified = True


_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: "OrderedDict[RuntimeCacheKey, AnimaNativeReferenceV2Runtime]" = OrderedDict()


def _make_runtime_cache_key(
    checkpoint_path: os.PathLike[str] | str,
    text_encoder_path: os.PathLike[str] | str,
    vae_path: os.PathLike[str] | str,
    *,
    runtime_root: os.PathLike[str] | str | None,
    device: str | torch.device | None,
    offload_mode: str,
    attn_mode: str,
    vae_chunk_size: int,
    vae_disable_cache: bool,
    runtime_kind: str = "v2",
) -> RuntimeCacheKey:
    root = (
        Path(runtime_root or default_runtime_root()).expanduser().resolve(strict=True)
    )
    return RuntimeCacheKey(
        runtime_kind=str(runtime_kind),
        checkpoint=FileFingerprint.from_path(checkpoint_path),
        text_encoder=FileFingerprint.from_path(text_encoder_path),
        vae=FileFingerprint.from_path(vae_path),
        runtime_root=str(root),
        device=str(_resolve_device(device)),
        offload_mode=offload_mode,
        attention_mode=attn_mode,
        vae_chunk_size=int(vae_chunk_size),
        vae_disable_cache=bool(vae_disable_cache),
    )


def get_or_create_runtime(
    checkpoint_path: os.PathLike[str] | str,
    text_encoder_path: os.PathLike[str] | str,
    vae_path: os.PathLike[str] | str,
    *,
    runtime_root: os.PathLike[str] | str | None = None,
    device: str | torch.device | None = None,
    offload_mode: str = "balanced",
    attn_mode: str = "torch",
    vae_chunk_size: int = DEFAULT_VAE_CHUNK_SIZE,
    vae_disable_cache: bool = True,
    prompt_cache_entries: int = 64,
    reference_cache_entries: int = 16,
    verify_checkpoint_sha256: bool = False,
) -> AnimaNativeReferenceV2Runtime:
    """Return the one process runtime for an identical model/config fingerprint."""

    key = _make_runtime_cache_key(
        checkpoint_path,
        text_encoder_path,
        vae_path,
        runtime_root=runtime_root,
        device=device,
        offload_mode=offload_mode,
        attn_mode=attn_mode,
        vae_chunk_size=vae_chunk_size,
        vae_disable_cache=vae_disable_cache,
        runtime_kind="v2",
    )
    with _MODEL_CACHE_LOCK:
        runtime = _MODEL_CACHE.get(key)
        if runtime is not None and not runtime._closed:
            runtime.set_condition_cache_limits(
                prompt_cache_entries=prompt_cache_entries,
                reference_cache_entries=reference_cache_entries,
            )
            if verify_checkpoint_sha256:
                runtime.verify_release_sha256()
            _MODEL_CACHE.move_to_end(key)
            return runtime
        if runtime is not None:
            # A caller may close a cached runtime explicitly.  Never return the
            # closed object and do not let it consume an LRU slot.
            del _MODEL_CACHE[key]
        runtime = AnimaNativeReferenceV2Runtime(
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
        _MODEL_CACHE[key] = runtime
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > PROCESS_MODEL_CACHE_CAPACITY:
            # Do not explicitly close an evicted object: a live Comfy loader
            # output may still own and use it.  If no outside owner exists,
            # normal reference counting / __del__ releases it immediately.
            _MODEL_CACHE.popitem(last=False)
        return runtime


def get_or_create_v4_runtime(
    checkpoint_path: os.PathLike[str] | str,
    text_encoder_path: os.PathLike[str] | str,
    vae_path: os.PathLike[str] | str,
    *,
    runtime_root: os.PathLike[str] | str | None = None,
    device: str | torch.device | None = None,
    offload_mode: str = "balanced",
    attn_mode: str = "torch",
    vae_chunk_size: int = DEFAULT_VAE_CHUNK_SIZE,
    vae_disable_cache: bool = True,
    prompt_cache_entries: int = 64,
    reference_cache_entries: int = 16,
    verify_checkpoint_sha256: bool = False,
) -> AnimaNativeReferenceV4Runtime:
    """Return the process-cached exact final-V4 runtime."""

    key = _make_runtime_cache_key(
        checkpoint_path,
        text_encoder_path,
        vae_path,
        runtime_root=runtime_root,
        device=device,
        offload_mode=offload_mode,
        attn_mode=attn_mode,
        vae_chunk_size=vae_chunk_size,
        vae_disable_cache=vae_disable_cache,
        runtime_kind="v4-final49k",
    )
    with _MODEL_CACHE_LOCK:
        runtime = _MODEL_CACHE.get(key)
        if runtime is not None and not runtime._closed:
            if not isinstance(runtime, AnimaNativeReferenceV4Runtime):
                raise AnimaRuntimeError(
                    "Runtime cache key collision between V2 and final V4."
                )
            runtime.set_condition_cache_limits(
                prompt_cache_entries=prompt_cache_entries,
                reference_cache_entries=reference_cache_entries,
            )
            if verify_checkpoint_sha256:
                runtime.verify_release_sha256()
            _MODEL_CACHE.move_to_end(key)
            return runtime
        if runtime is not None:
            del _MODEL_CACHE[key]
        created = AnimaNativeReferenceV4Runtime(
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
        _MODEL_CACHE[key] = created
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > PROCESS_MODEL_CACHE_CAPACITY:
            _MODEL_CACHE.popitem(last=False)
        return created


def clear_model_cache(*, close: bool = True) -> None:
    """Clear process-level model-cache references; optionally close live runtimes."""

    with _MODEL_CACHE_LOCK:
        runtimes = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
    if close:
        for runtime in runtimes:
            runtime.close()


__all__ = [
    "AnimaNativeReferenceV2Runtime",
    "AnimaNativeReferenceV4Runtime",
    "AnimaRuntimeError",
    "CacheStatistics",
    "CheckpointValidationError",
    "E180CheckpointInfo",
    "FileFingerprint",
    "InputValidationError",
    "RawPromptEncoding",
    "RuntimeImportError",
    "V4CheckpointInfo",
    "clear_model_cache",
    "default_runtime_root",
    "get_or_create_runtime",
    "get_or_create_v4_runtime",
    "load_runtime_modules",
    "validate_e180_checkpoint",
    "validate_qwen_image_vae",
    "validate_qwen_text_encoder",
    "validate_v4_checkpoint",
    "validate_vendored_runtime",
]
