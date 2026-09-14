"""Minimal shared primitives for the Anima Native Context V7 runtime.

This file contains only V7 loader, validation, cache, tensor-conversion, and
conditioning utilities.  It deliberately contains no V2/V4 checkpoint loader,
legacy router, prompt parser, inference fallback, adapter, or LoRA path.
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
from typing import Any, Dict, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch


VENDORED_ANIMA_COMMIT = "2ae811d296ff4159c6024c4a86415d19961a388c"
VENDORED_ANIMA_OVERLAY = "v7-native-context-20260810"
QWEN_EXPECTED_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
QWEN_EXPECTED_FILE_SIZE = 1_192_135_096
QWEN_EXPECTED_TOTAL_TENSORS = 310
VAE_EXPECTED_SHA256 = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
VAE_EXPECTED_FILE_SIZE = 253_806_246
VAE_EXPECTED_TOTAL_TENSORS = 194

DEFAULT_REFERENCE_MAX_AREA = 262_144
DEFAULT_TEXT_MAX_LENGTH = 512
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FLOW_SHIFT = 5.0
REFERENCE_PREPROCESS_MODES = frozenset({"independent_reference", "match_output_edit"})
PROCESS_MODEL_CACHE_CAPACITY = 2


class AnimaRuntimeError(RuntimeError):
    """Base exception for the custom runtime."""


class CheckpointValidationError(AnimaRuntimeError):
    """Raised when a file violates the integrated V7 checkpoint contract."""


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
    ``_anima_native_ref_vendor``; a generic ``library`` package in the hosting
    ComfyUI process is therefore never read or overwritten.
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
        root / "library" / "anima_reference_router.py",
        root / "library" / "anima_text_conditioning.py",
        root / "library" / "anima_runtime_strategy.py",
        root / "library" / "anima_edit_geometry.py",
        root / "library" / "hunyuan_image_utils.py",
        root / "library" / "qwen_image_autoencoder_kl.py",
        root / "configs" / "qwen3_06b" / "config.json",
        root / "configs" / "t5_old" / "spiece.model",
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
                    f"{_VENDOR_PACKAGE_NAME}.library.anima_runtime_strategy",
                    cached.strategy_anima,
                ),
                (
                    f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning",
                    cached.anima_text_conditioning,
                ),
            ):
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
                f"{_VENDOR_PACKAGE_NAME}.library.anima_runtime_strategy"
            )
            anima_text_conditioning = importlib.import_module(
                f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning"
            )
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
            (
                f"{_VENDOR_PACKAGE_NAME}.library.anima_runtime_strategy",
                strategy_anima,
            ),
            (
                f"{_VENDOR_PACKAGE_NAME}.library.anima_text_conditioning",
                anima_text_conditioning,
            ),
        ):
            _validate_module_origin(module, root, name)

        modules = RuntimeModules(
            anima_utils,
            anima_models,
            hunyuan_image_utils,
            qwen_vae,
            strategy_anima,
            anima_text_conditioning,
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


class NativeContextRuntimePrimitives:
    """Lifecycle, cache, and encoder primitives shared by the V7 runtime."""

    SUPPORTED_OFFLOAD_MODES = frozenset({"balanced", "high_vram", "text_encoder_cpu"})

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

    def _move_vae_to(self, device: torch.device | str) -> None:
        self._vae.to(device)

    def _offload_vae_if_needed(self) -> None:
        if self.offload_mode != "high_vram":
            self._vae.to("cpu")

    def _offload_text_encoder_if_needed(self) -> None:
        if self.offload_mode != "high_vram":
            self._text_encoder.to("cpu")

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
                f"V7 requires one or two physical reference images; got {len(images)}."
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
        if preprocess_mode not in REFERENCE_PREPROCESS_MODES:
            raise InputValidationError(
                f"Unknown preprocess_mode {preprocess_mode!r}; "
                f"expected one of {sorted(REFERENCE_PREPROCESS_MODES)}."
            )
        if len(images) == 2 and preprocess_mode != "independent_reference":
            raise InputValidationError(
                "Two-reference V7 generation must use independent_reference "
                "preprocessing; match_output_edit is a one-reference edit path."
            )
        return images, slots

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
            self._preprocess_reference(
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

    def _encode_text_uncached(self, text: str) -> RawPromptEncoding:
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
                    encoded = self._encode_text_uncached(text)
                    self._prompt_cache.put(text, encoded)
                    values[text] = encoded
            finally:
                self._offload_text_encoder_if_needed()

        return values[prompt], values[negative_prompt]

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

    def metadata(self) -> Mapping[str, str]:
        """Read-only checkpoint metadata validated at loader construction."""

        return self.checkpoint_info.metadata

    def fingerprint(self) -> FileFingerprint:
        """Validated V7 checkpoint fingerprint used by the model cache."""

        return self.checkpoint_info.fingerprint

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


_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: "OrderedDict[RuntimeCacheKey, NativeContextRuntimePrimitives]" = OrderedDict()


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
    runtime_kind: str = "v7-native-context-v1",
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


def clear_model_cache(*, close: bool = True) -> None:
    """Drop strong cache references; optionally close the current V7 runtimes."""

    with _MODEL_CACHE_LOCK:
        runtimes = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
    if close:
        for runtime in runtimes:
            runtime.close()


__all__ = [
    "AnimaRuntimeError",
    "CheckpointValidationError",
    "InputValidationError",
    "NativeContextRuntimePrimitives",
    "RuntimeImportError",
    "clear_model_cache",
    "load_runtime_modules",
    "validate_qwen_image_vae",
    "validate_qwen_text_encoder",
]
