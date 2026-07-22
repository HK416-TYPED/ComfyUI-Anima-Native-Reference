# Anima model loading/saving utilities

import json
import os
from typing import Any, Dict, List, Optional, Union
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device  # kept for potential future use
from accelerate import init_empty_weights

from _anima_native_ref_vendor.library.fp8_optimization_utils import apply_fp8_monkey_patch
from _anima_native_ref_vendor.library.lora_utils import load_safetensors_with_lora_and_fp8
from _anima_native_ref_vendor.library import anima_models
from _anima_native_ref_vendor.library.safetensors_utils import WeightTransformHooks, get_split_weight_filenames
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Original Anima high-precision keys. Kept for reference, but not used currently.
# # Keys that should stay in high precision (float32/bfloat16, not quantized)
# KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
# ".embed." excludes Embedding in LLMAdapter
FP8_OPTIMIZATION_EXCLUDE_KEYS = [
    "_embedder",
    "norm",
    "adaln",
    "final_layer",
    ".embed.",
    "native_reference",
    "reference_slot",
    "reference_type",
]


# Native multi-reference checkpoints are self-describing.  Keep the metadata
# names generic: a reference slot is only an image index, never a hard-coded
# source/identity/style role.
ANIMA_NATIVE_REFERENCE_VERSION = 1
ANIMA_NATIVE_REFERENCE_METADATA_KEY = "anima_native_reference_conditioning"
ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY = "anima_native_reference_config"
ANIMA_NATIVE_REFERENCE_STATE_KEYS = (
    "reference_slot_embeddings.weight",
    "reference_type_embedding",
)
ANIMA_NATIVE_REFERENCE_STATE_MARKER = ".native_reference_attn."


def _normalise_anima_state_key(key: str) -> str:
    return key[len("net.") :] if key.startswith("net.") else key


def is_native_reference_state_key(key: str) -> bool:
    """Return whether *key* belongs to integrated native reference conditioning."""

    key = _normalise_anima_state_key(key)
    return key in ANIMA_NATIVE_REFERENCE_STATE_KEYS or ANIMA_NATIVE_REFERENCE_STATE_MARKER in key


def _as_checkpoint_file_list(model_files: Union[str, List[str]]) -> List[str]:
    if isinstance(model_files, str):
        model_files = [model_files]
    result: List[str] = []
    for model_file in model_files:
        split_files = get_split_weight_filenames(model_file)
        result.extend(split_files if split_files is not None else [model_file])
    return result


def _metadata_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def inspect_anima_checkpoint(model_files: Union[str, List[str]]) -> Dict[str, Any]:
    """Inspect Anima checkpoint structure without loading its large tensors.

    The state keys are authoritative; metadata is an explicit, human-readable
    description used to round-trip the architecture configuration.  Old base
    checkpoints have neither and therefore remain native-reference disabled.
    """

    metadata: Dict[str, str] = {}
    native_keys: List[str] = []
    slot_rows: Optional[int] = None
    state_architecture: Optional[str] = None
    state_rank: Optional[int] = None
    state_router_dim: Optional[int] = None
    state_head_count: Optional[int] = None
    state_spatial_rows: Optional[int] = None
    for checkpoint_file in _as_checkpoint_file_list(model_files):
        if not str(checkpoint_file).lower().endswith(".safetensors"):
            continue
        with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
            metadata.update(handle.metadata() or {})
            for raw_key in handle.keys():
                key = _normalise_anima_state_key(raw_key)
                if is_native_reference_state_key(key):
                    native_keys.append(key)
                if key == "reference_slot_embeddings.weight":
                    slot_rows = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.k_down.weight"):
                    state_architecture = "v2"
                    state_rank = int(handle.get_slice(raw_key).get_shape()[0])
                elif key.endswith(".native_reference_attn.k_proj.weight") and state_architecture is None:
                    state_architecture = "v1"
                if key.endswith(".native_reference_attn.prompt_key.weight"):
                    state_router_dim = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.head_gate.weight"):
                    state_head_count = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.target_spatial.weight"):
                    state_spatial_rows = int(handle.get_slice(raw_key).get_shape()[0])

    config: Dict[str, Any] = {}
    raw_config = metadata.get(ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY)
    if raw_config:
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                config.update(parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY} metadata: {raw_config!r}") from exc

    metadata_enabled = _metadata_bool(metadata.get(ANIMA_NATIVE_REFERENCE_METADATA_KEY, False))
    enabled = bool(native_keys) or metadata_enabled
    if slot_rows is not None:
        config["max_reference_images"] = slot_rows
    if state_architecture is not None:
        # Tensor layout is authoritative when metadata is absent or stale.
        config["architecture"] = state_architecture
    if state_rank is not None:
        config["rank"] = state_rank
    if state_router_dim is not None:
        config["router_dim"] = state_router_dim
    if state_head_count and state_spatial_rows and state_spatial_rows % state_head_count == 0:
        config["gate_dim"] = state_spatial_rows // state_head_count
    config.setdefault("architecture", "v1")
    config.setdefault("rank", 64)
    config.setdefault("max_reference_images", 8)
    config.setdefault("router_dim", 128)
    config.setdefault("gate_dim", 8)
    config.setdefault("initial_gate", 0.5 if config["architecture"] == "v2" else 0.01)
    detected_version = 2 if config["architecture"] == "v2" else ANIMA_NATIVE_REFERENCE_VERSION

    public_config = dict(config)
    # Preserve the exact public schema of historical V1 checkpoints. V2 adds
    # architecture/rank only when those fields are semantically required.
    if public_config["architecture"] == "v1":
        public_config.pop("architecture", None)
        public_config.pop("rank", None)

    return {
        "native_reference_conditioning": enabled,
        "native_reference_version": int(metadata.get("anima_native_reference_version", detected_version)),
        "native_reference_config": public_config,
        "native_reference_keys": tuple(native_keys),
        "metadata": metadata,
    }


def add_anima_architecture_metadata(
    metadata: Optional[Dict[str, Any]],
    state_dict: Dict[str, torch.Tensor],
    native_reference_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return safetensors metadata describing integrated native-ref weights."""

    result = {str(k): str(v) for k, v in (metadata or {}).items()}
    native_keys = [key for key in state_dict if is_native_reference_state_key(key)]
    if not native_keys:
        return result

    config: Dict[str, Any] = {
        "architecture": "v1",
        "rank": 64,
        "max_reference_images": 8,
        "router_dim": 128,
        "gate_dim": 8,
        "initial_gate": 0.01,
    }
    config.update(native_reference_config or {})
    for key, tensor in state_dict.items():
        normalised_key = _normalise_anima_state_key(key)
        if normalised_key == "reference_slot_embeddings.weight":
            config["max_reference_images"] = int(tensor.shape[0])
        if normalised_key.endswith(".native_reference_attn.k_down.weight"):
            config["architecture"] = "v2"
            config["rank"] = int(tensor.shape[0])
    if config["architecture"] == "v2" and "initial_gate" not in (native_reference_config or {}):
        config["initial_gate"] = 0.5
    if config["architecture"] == "v1":
        config.pop("architecture", None)
        config.pop("rank", None)

    result[ANIMA_NATIVE_REFERENCE_METADATA_KEY] = "true"
    version = 2 if config.get("architecture") == "v2" else ANIMA_NATIVE_REFERENCE_VERSION
    result["anima_native_reference_version"] = str(version)
    result[ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # Explicitly documents that numbered slots carry no fixed semantic role.
    result["anima_reference_slot_semantics"] = "ordered_image_index"
    result["anima_checkpoint_layout"] = "integrated_single_checkpoint"
    result["anima_external_adapter_required"] = "false"
    # NetworkTrainer normally emits an adapter ModelSpec because it owns a
    # network object.  Native-reference publication instead contains the full
    # frozen base plus the integrated route, so remove that misleading suffix.
    architecture = result.get("modelspec.architecture")
    if architecture and architecture.endswith("/lora"):
        result["modelspec.architecture"] = architecture[: -len("/lora")]
    return result


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    enable_ip_adapter: bool = False,
    ip_adapter_scale: float = 1.0,
    ip_adapter_feature_dim: Optional[int] = None,
    ip_adapter_num_tokens: int = 4,
    ip_adapter_linear_adapter: bool = False,
    ip_adapter_mlp_adapter: bool = False,
    ip_adapter_norm_linear_adapter: bool = False,
    ip_adapter_omni_adapter: bool = False,
    enable_native_reference_conditioning: Optional[bool] = None,
    native_reference_max_images: Optional[int] = None,
    native_reference_router_dim: Optional[int] = None,
    native_reference_gate_dim: Optional[int] = None,
    native_reference_initial_gate: Optional[float] = None,
    native_reference_architecture: Optional[str] = None,
    native_reference_rank: Optional[int] = None,
    native_reference_scale: float = 1.0,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (
        not fp8_scaled and dit_weight_dtype is not None
    ) or dit_weight_dtype is None, "dit_weight_dtype should be None when fp8_scaled is True"

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    checkpoint_info = inspect_anima_checkpoint(dit_path)
    checkpoint_has_native_reference = checkpoint_info["native_reference_conditioning"]
    if enable_native_reference_conditioning is False and checkpoint_has_native_reference:
        raise ValueError(
            "Checkpoint contains integrated native reference weights, but "
            "enable_native_reference_conditioning=False was requested."
        )
    use_native_reference = (
        checkpoint_has_native_reference
        if enable_native_reference_conditioning is None
        else enable_native_reference_conditioning
    )
    native_config = dict(checkpoint_info["native_reference_config"])
    native_config.setdefault("architecture", "v1")
    native_config.setdefault("rank", 64)
    explicit_native_config = {
        "max_reference_images": native_reference_max_images,
        "router_dim": native_reference_router_dim,
        "gate_dim": native_reference_gate_dim,
        "initial_gate": native_reference_initial_gate,
        "architecture": native_reference_architecture,
        "rank": native_reference_rank,
    }
    if checkpoint_has_native_reference:
        for config_key, explicit_value in explicit_native_config.items():
            if explicit_value is None:
                continue
            detected_value = native_config[config_key]
            if explicit_value != detected_value:
                raise ValueError(
                    f"Native reference checkpoint config {config_key}={detected_value!r}, "
                    f"but {explicit_value!r} was requested."
                )
    native_config.update({key: value for key, value in explicit_native_config.items() if value is not None})

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
        # For an integrated checkpoint, build these modules in the empty-weight
        # context so state_dict assignment can populate them without a second
        # 2B-model allocation.  For a base checkpoint used to start new native
        # training, enable them after the base weights have been assigned.
        "native_reference_conditioning": bool(checkpoint_has_native_reference),
        "native_reference_max_images": int(native_config["max_reference_images"]),
        "native_reference_router_dim": int(native_config["router_dim"]),
        "native_reference_gate_dim": int(native_config["gate_dim"]),
        "native_reference_initial_gate": float(native_config["initial_gate"]),
        "native_reference_architecture": str(native_config["architecture"]),
        "native_reference_rank": int(native_config["rank"]),
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    rename_hooks = WeightTransformHooks(rename_hook=lambda k: k[len("net.") :] if k.startswith("net.") else k)
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=rename_hooks,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if use_native_reference and not checkpoint_has_native_reference:
        # New training from a legacy base checkpoint: create initialized native
        # modules only after assigning the base state, otherwise their missing
        # tensors would remain on the meta device.
        model.enable_native_reference_conditioning(
            max_reference_images=int(native_config["max_reference_images"]),
            router_dim=int(native_config["router_dim"]),
            gate_dim=int(native_config["gate_dim"]),
            initial_gate=float(native_config["initial_gate"]),
            architecture=str(native_config["architecture"]),
            rank=int(native_config["rank"]),
        )
    if use_native_reference:
        model.set_native_reference_scale(native_reference_scale)
        logger.info(
            "Native reference conditioning enabled: architecture=%s, rank=%s, max_images=%s, router_dim=%s, gate_dim=%s, scale=%s",
            native_config["architecture"],
            native_config["rank"],
            native_config["max_reference_images"],
            native_config["router_dim"],
            native_config["gate_dim"],
            native_reference_scale,
        )

    if enable_ip_adapter:
        model.enable_ip_adapter(
            ip_adapter_scale,
            feature_dim=ip_adapter_feature_dim,
            num_feature_tokens=ip_adapter_num_tokens,
            linear_adapter=ip_adapter_linear_adapter,
            mlp_adapter=ip_adapter_mlp_adapter,
            norm_linear_adapter=ip_adapter_norm_linear_adapter,
            omni_adapter=ip_adapter_omni_adapter,
        )
        ip_adapter_dtype = dit_weight_dtype
        if model.visual_condition_adapter is not None:
            model.visual_condition_adapter.to(device=loading_device, dtype=ip_adapter_dtype)
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen3_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora_and_fp8(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    fp8_optimization=False,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, "LoRA weights merging is only supported for safetensors checkpoints"
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str,
    dit_state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, Any]],
    dtype: Optional[torch.dtype] = None,
    native_reference_config: Optional[Dict[str, Any]] = None,
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        metadata: Metadata dict to include in the safetensors file
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            # v = v.to(dtype)
            v = v.detach().clone().to("cpu").to(dtype)  # Reduce GPU memory usage during save
        prefixed_sd["net." + k] = v.contiguous()

    metadata = add_anima_architecture_metadata(metadata, dit_state_dict, native_reference_config)
    metadata["format"] = "pt"  # For compatibility with the official .safetensors file

    save_file(prefixed_sd, save_path, metadata=metadata)  # safetensors.save_file consumes a lot of memory, but Anima is small enough
    logger.info(f"Saved Anima model to {save_path}")
