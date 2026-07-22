# Anima Training Utilities

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

from _anima_native_ref_vendor.library.device_utils import init_ipex, clean_memory_on_device, synchronize_device
from _anima_native_ref_vendor.library import anima_models, anima_utils, train_util, qwen_image_autoencoder_kl

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaSampleIPFeatureExtractor:
    def __init__(self, backend: str, model_dir: str, device: torch.device, dtype: torch.dtype):
        self.backend = backend
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.feature_dim = None
        if backend == "ccip":
            from ccip_lib.ccip import ccip_batch_extract_features, _open_feat_model

            self.extract_fn = ccip_batch_extract_features
            shape = _open_feat_model(model_dir).get_outputs()[0].shape
            self.feature_dim = shape[-1] if len(shape) >= 2 and isinstance(shape[-1], int) else None
        elif backend == "ccip_tokens":
            self.model, self.transform, self.feature_dim = self._load_ccip_tokens(model_dir, device)
        elif backend == "siglip2_tokens":
            self.model, self.transform, self.feature_dim = self._load_siglip2_tokens(model_dir, device)
        elif backend == "lsnet":
            self.model, self.transform, self.feature_dim = self._load_lsnet(model_dir, device)

    @staticmethod
    def _load_siglip2_tokens(model_path: str, device: torch.device):
        from transformers import Siglip2VisionModel, Siglip2ImageProcessor

        logger.info(f"Loading SigLIP2 vision model (sample) from: {model_path}")
        processor = Siglip2ImageProcessor.from_pretrained(model_path)
        vision_model = Siglip2VisionModel.from_pretrained(model_path)
        vision_model.to(device)
        vision_model.eval()
        feature_dim = vision_model.config.hidden_size
        logger.info(f"Loaded SigLIP2 vision model (sample). feature_dim={feature_dim}")
        return vision_model, processor, feature_dim

    @staticmethod
    def _find_ccip_checkpoint(model_path):
        path = Path(model_path)
        if path.is_file():
            return str(path)
        candidates = sorted(path.glob("*.ckpt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No CCIP checkpoint found in: {model_path}")
        return str(candidates[0])

    @staticmethod
    def _clean_ccip_state_dict(state_dict):
        return {
            key.removeprefix("module._orig_mod.").removeprefix("module.").removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _load_ccip_tokens(model_path, device):
        from ccip_lib.models.caformer import get_caformer

        checkpoint = AnimaSampleIPFeatureExtractor._find_ccip_checkpoint(model_path)
        logger.info(f"Loading CCIP token backbone from: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = AnimaSampleIPFeatureExtractor._clean_ccip_state_dict(state_dict)
        first_conv = state_dict.get("feature.backbone.caformer.downsample_layers.0.conv.weight")
        arch = "caformer_b36_384_in21ft1k" if first_conv is not None and first_conv.shape[0] == 128 else "caformer_s36_384_in21ft1k"
        logger.info(f"Detected CCIP token backbone arch: {arch}")
        backbone, transform = get_caformer(arch=arch, pretrained=False)
        backbone_state = {
            key.removeprefix("feature.backbone."): value
            for key, value in state_dict.items()
            if key.startswith("feature.backbone.")
        }
        backbone.load_state_dict(backbone_state, strict=False)
        backbone.to(device).eval()
        logger.info(f"Loaded CCIP token backbone. feature_dim={backbone.caformer.output_dim}")
        return backbone, transform, backbone.caformer.output_dim

    @staticmethod
    def _find_lsnet_checkpoint(model_dir):
        candidates = sorted(Path(model_dir).glob("*.pth")) + sorted(Path(model_dir).glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No LSNet checkpoint found in: {model_dir}")
        return str(candidates[0])

    @staticmethod
    def _load_lsnet(model_dir, device):
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from lsnet_lib.inference_artist import load_checkpoint_state, load_model, normalize_state_dict_keys, resolve_feature_dim, resolve_num_classes
        from lsnet_lib.lsnet_model.lsnet_artist import default_cfgs_artist

        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            model_name = json.load(f)["model"]
        checkpoint = AnimaSampleIPFeatureExtractor._find_lsnet_checkpoint(model_dir)
        state_dict = normalize_state_dict_keys(load_checkpoint_state(checkpoint))
        feature_dim = resolve_feature_dim(None, state_dict)
        load_args = SimpleNamespace(
            model=model_name,
            checkpoint=checkpoint,
            num_classes=resolve_num_classes(None, None, state_dict),
            feature_dim=feature_dim,
            mode="cluster",
            allow_head_reinit=True,
            device=str(device),
        )
        model = load_model(load_args, state_dict).to(device).eval()
        input_size = default_cfgs_artist.get(model_name, {}).get("input_size", (3, 448 if model_name.endswith("_448") else 224, 224))[1]
        transform = create_transform(**resolve_data_config({"input_size": (3, input_size, input_size)}, model=model))
        return model, transform, feature_dim

    def extract(self, image_paths):
        if self.backend == "ccip":
            features = self.extract_fn(image_paths, model=self.model_dir)
            return torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device, dtype=self.dtype)
        if self.backend == "ccip_tokens":
            tensors = []
            for path in image_paths:
                image = Image.open(path).convert("RGB").resize((384, 384), resample=Image.BILINEAR)
                tensor = transforms.ToTensor()(image)
                for transform in self.transform:
                    tensor = transform(tensor)
                tensors.append(tensor)
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                fmap = self.model._get_cnn_result(batch)
                features = fmap.flatten(2).transpose(1, 2).contiguous()
            return features.reshape(1, -1, features.shape[-1]).to(dtype=self.dtype)
        if self.backend == "siglip2_tokens":
            images = [Image.open(path).convert("RGB") for path in image_paths]
            inputs = self.transform(images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            pixel_attention_mask = inputs["pixel_attention_mask"].to(self.device)
            spatial_shapes = inputs["spatial_shapes"].to(self.device)
            with torch.no_grad():
                vision_outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                )
                features = vision_outputs.last_hidden_state
            return features.reshape(1, -1, features.shape[-1]).to(dtype=self.dtype)
        if self.backend == "lsnet":
            tensors = [self.transform(Image.open(path).convert("RGB")) for path in image_paths]
            with torch.no_grad():
                features = self.model(torch.stack(tensors).to(self.device), return_features=True)
            return features.unsqueeze(0).to(dtype=self.dtype)
        return None


# Anima-specific training arguments


def add_anima_training_arguments(parser: argparse.ArgumentParser):
    """Add Anima-specific training arguments to the parser."""
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B model (safetensors file or directory)",
    )
    parser.add_argument(
        "--llm_adapter_path",
        type=str,
        default=None,
        help="Path to separate LLM adapter weights. If None, adapter is loaded from DiT file if present",
    )
    parser.add_argument(
        "--llm_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for LLM adapter. None=same as base LR, 0=freeze adapter",
    )
    parser.add_argument(
        "--self_attn_lr",
        type=float,
        default=None,
        help="Learning rate for self-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--cross_attn_lr",
        type=float,
        default=None,
        help="Learning rate for cross-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=None,
        help="Learning rate for MLP layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mod_lr",
        type=float,
        default=None,
        help="Learning rate for AdaLN modulation layers. None=same as base LR, 0=freeze. Note: mod layers are not included in LoRA by default.",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer directory. If None, uses default configs/t5_old/",
    )
    parser.add_argument(
        "--qwen3_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for Qwen3 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--t5_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for T5 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training (default: 1.0)",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        help="Timestep sampling method (default: sigmoid (logit normal))",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0)",
    )
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). xformers requires --split_attn. sageattn does not support training (inference only). This option overrides --xformers or --sdpa."
        " / 使用するAttentionの実装。デフォルトはNone（torch）です。xformersは--split_attnの指定が必要です。sageattnはトレーニングをサポートしていません（推論のみ）。このオプションは--xformersまたは--sdpaを上書きします。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention computation to reduce memory usage / メモリ使用量を減らすためにattention時にバッチを分割する",
    )
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + " / メモリ使用量を減らすためのVAEエンコード/デコードの空間チャンクサイズ。偶数である必要があります。未指定の場合、チャンク処理は無効になります（公式の動作）。",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )
    parser.add_argument(
        "--anima_multi_image_edit",
        action="store_true",
        help="Enable multi-image reference conditioning for Anima training.",
    )
    parser.add_argument(
        "--anima_native_reference_conditioning",
        action="store_true",
        help="Enable integrated generic native reference slots (Image 1, Image 2, ...).",
    )
    parser.add_argument(
        "--anima_train_native_reference",
        action="store_true",
        help="Train only integrated native reference modules and save one complete Anima checkpoint.",
    )
    parser.add_argument(
        "--anima_native_reference_lr",
        type=float,
        default=None,
        help="Learning rate for native reference modules; defaults to unet_lr or learning_rate.",
    )
    parser.add_argument(
        "--anima_native_reference_max_images",
        type=int,
        default=8,
        help="Maximum number of logical indexed reference slots embedded in the checkpoint.",
    )
    parser.add_argument("--anima_native_reference_router_dim", type=int, default=128)
    parser.add_argument("--anima_native_reference_gate_dim", type=int, default=8)
    parser.add_argument("--anima_native_reference_initial_gate", type=float, default=0.01)
    parser.add_argument(
        "--anima_native_reference_architecture",
        choices=("v1", "v2"),
        default="v1",
        help="Native reference injection layout. v2 reuses frozen base K/V with zero-up low-rank deltas.",
    )
    parser.add_argument(
        "--anima_native_reference_rank",
        type=int,
        default=64,
        help="Low-rank width used by the v2 K/V and output residual branches.",
    )
    parser.add_argument("--anima_native_reference_scale", type=float, default=1.0)
    parser.add_argument(
        "--anima_native_fixed2ref_vectorized",
        action="store_true",
        help=(
            "Opt into the torch-SDPA fixed-two-reference fast path. Every sample must contain exactly two "
            "non-missing ordered references; the default serial/native route is unchanged."
        ),
    )
    parser.add_argument(
        "--anima_sync_single_reference_geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize a single edit reference to target bucket geometry and mirror the target flip.",
    )
    parser.add_argument(
        "--anima_ip_adapter",
        action="store_true",
        help="Enable IP-Adapter conditioning for Anima using the same *_ref reference images.",
    )
    parser.add_argument(
        "--anima_ip_adapter_scale",
        type=float,
        default=1.0,
        help="Scale for the Anima IP-Adapter K/V attention branch.",
    )
    parser.add_argument(
        "--anima_train_ip_adapter",
        action="store_true",
        help="Train and save Anima IP-Adapter weights separately from LoRA/LyCORIS weights.",
    )
    parser.add_argument(
        "--anima_ip_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for Anima IP-Adapter parameters. Defaults to unet_lr or learning_rate.",
    )
    parser.add_argument(
        "--anima_ip_adapter_weights",
        type=str,
        default=None,
        help="Optional Anima IP-Adapter safetensors weights to load before training.",
    )
    parser.add_argument(
        "--anima_ip_adapter_feature_backend",
        type=str,
        default="vae",
        choices=["vae", "ccip", "ccip_tokens", "lsnet", "siglip2_tokens"],
        help="Feature extractor for Anima IP-Adapter references. 'vae' reuses reference latent tokens.",
    )
    parser.add_argument(
        "--anima_ip_adapter_feature_model",
        type=str,
        default=None,
        help="Model directory/checkpoint for CCIP, CCIP token, or LSNet IP-Adapter feature extraction.",
    )
    parser.add_argument(
        "--anima_ip_adapter_feature_dim",
        type=int,
        default=None,
        help="Override IP-Adapter feature dimension if it cannot be inferred automatically.",
    )
    parser.add_argument(
        "--anima_ip_adapter_precomputed_emb_dir",
        type=str,
        default=None,
        help="Directory with precomputed .pt embedding files. When set, features are loaded from disk "
        "instead of running the feature extractor at training time. Each .pt file should have the "
        "same stem as the corresponding image file.",
    )
    parser.add_argument(
        "--anima_ip_adapter_num_tokens",
        type=int,
        default=4,
        help="Number of Anima visual context tokens produced from CCIP/LSNet feature vectors.",
    )
    parser.add_argument(
        "--linear-adapter",
        "--anima_ip_adapter_linear_adapter",
        dest="linear_adapter",
        action="store_true",
        help="Use a lightweight Linear + LayerNorm visual adapter instead of the transformer resampler.",
    )
    parser.add_argument(
        "--mlp-adapter",
        "--mlp_adapter",
        "--anima_ip_adapter_mlp_adapter",
        dest="mlp_adapter",
        action="store_true",
        help="Use a FaceID-style MLP + LayerNorm visual adapter instead of the transformer resampler.",
    )
    parser.add_argument(
        "--norm-linear-adapter",
        "--norm_linear_adapter",
        "--anima_ip_adapter_norm_linear_adapter",
        dest="norm_linear_adapter",
        action="store_true",
        help="Use a pre-norm bias-free Linear visual adapter instead of the transformer resampler.",
    )
    parser.add_argument(
        "--omni-adapter",
        "--omni_adapter",
        "--anima_ip_adapter_omni_adapter",
        dest="omni_adapter",
        action="store_true",
        help="Use a token-preserving visual self-attention refiner adapter.",
    )
    parser.add_argument(
        "--anima_disable_network_training",
        action="store_true",
        help="Do not train LoRA/LyCORIS network parameters; useful for IP-Adapter-only training.",
    )
    parser.add_argument(
        "--anima_reference_t_offset_scale",
        type=int,
        default=10,
        help="Legacy concat-only T-coordinate spacing; native reference conditioning ignores it.",
    )
    parser.add_argument(
        "--anima_reference_max_area",
        type=int,
        default=1024 * 1024,
        help="Maximum reference image area before aspect-preserving downscale. Set 0 to disable.",
    )
    parser.add_argument(
        "--anima_self_reference_test",
        action="store_true",
        help="Use each target training image as its own reference image when metadata does not provide references.",
    )
    parser.add_argument(
        "--anima_class_reference_test",
        action="store_true",
        help="Use a random different image from the same DreamBooth subfolder/class as reference when metadata does not provide references.",
    )
    parser.add_argument(
        "--anima_sample_reference_dir",
        type=str,
        default=None,
        help="Directory containing image/txt pairs for sampling. Each image is used as reference and the matching txt as prompt.",
    )
    parser.add_argument(
        "--anima_sample_max_references",
        type=int,
        default=4,
        help="Maximum number of reference images to sample when using --anima_sample_reference_dir. Default: 4",
    )
    parser.add_argument(
        "--anima_preference_training",
        action="store_true",
        help="Enable DPO-style preference training using *_neg, *_neg1, ... images paired with each target.",
    )
    parser.add_argument(
        "--anima_preference_beta",
        type=float,
        default=1000.0,
        help="Beta scale for Anima preference loss: -logsigmoid(beta * (negative_mse - positive_mse)).",
    )
    parser.add_argument(
        "--anima_preference_weight",
        type=float,
        default=1.0,
        help="Weight for the Anima preference loss added to the normal diffusion MSE loss.",
    )
    parser.add_argument(
        "--anima_preference_negative_mode",
        type=str,
        default="random",
        choices=["random", "all", "first"],
        help="How to use multiple *_neg images per target for preference training: random samples one per step, all expands every negative into a pair, first keeps the old behavior.",
    )


# Loss weighting


def compute_loss_weighting_for_anima(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute loss weighting for Anima training.

    Same schemes as SD3 but can add Anima-specific ones if needed in future.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    elif weighting_scheme == "none" or weighting_scheme is None:
        weighting = torch.ones_like(sigmas)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


# Parameter groups (6 groups with separate LRs)
def get_anima_param_groups(
    dit,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
):
    """Create parameter groups for Anima training with separate learning rates.

    Args:
        dit: Anima model
        base_lr: Base learning rate
        self_attn_lr: LR for self-attention layers (None = base_lr, 0 = freeze)
        cross_attn_lr: LR for cross-attention layers
        mlp_lr: LR for MLP layers
        mod_lr: LR for AdaLN modulation layers
        llm_adapter_lr: LR for LLM adapter

    Returns:
        List of parameter group dicts for optimizer
    """
    if self_attn_lr is None:
        self_attn_lr = base_lr
    if cross_attn_lr is None:
        cross_attn_lr = base_lr
    if mlp_lr is None:
        mlp_lr = base_lr
    if mod_lr is None:
        mod_lr = base_lr
    if llm_adapter_lr is None:
        llm_adapter_lr = base_lr

    base_params = []
    self_attn_params = []
    cross_attn_params = []
    mlp_params = []
    mod_params = []
    llm_adapter_params = []

    for name, p in dit.named_parameters():
        # Store original name for debugging
        p.original_name = name

        if "llm_adapter" in name:
            llm_adapter_params.append(p)
        elif ".self_attn" in name:
            self_attn_params.append(p)
        elif ".cross_attn" in name:
            cross_attn_params.append(p)
        elif ".mlp" in name:
            mlp_params.append(p)
        elif ".adaln_modulation" in name:
            mod_params.append(p)
        else:
            base_params.append(p)

    logger.info(f"Parameter groups:")
    logger.info(f"  base_params: {len(base_params)} (lr={base_lr})")
    logger.info(f"  self_attn_params: {len(self_attn_params)} (lr={self_attn_lr})")
    logger.info(f"  cross_attn_params: {len(cross_attn_params)} (lr={cross_attn_lr})")
    logger.info(f"  mlp_params: {len(mlp_params)} (lr={mlp_lr})")
    logger.info(f"  mod_params: {len(mod_params)} (lr={mod_lr})")
    logger.info(f"  llm_adapter_params: {len(llm_adapter_params)} (lr={llm_adapter_lr})")

    param_groups = []
    for lr, params, name in [
        (base_lr, base_params, "base"),
        (self_attn_lr, self_attn_params, "self_attn"),
        (cross_attn_lr, cross_attn_params, "cross_attn"),
        (mlp_lr, mlp_params, "mlp"),
        (mod_lr, mod_params, "mod"),
        (llm_adapter_lr, llm_adapter_params, "llm_adapter"),
    ]:
        if lr == 0:
            for p in params:
                p.requires_grad_(False)
            logger.info(f"  Frozen {name} params ({len(params)} parameters)")
        elif len(params) > 0:
            param_groups.append({"params": params, "lr": lr})

    total_trainable = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    return param_groups


# Save functions
def save_anima_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    dit: anima_models.Anima,
):
    """Save Anima model at the end of training."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        # Save with 'net.' prefix for ComfyUI compatibility
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


def save_anima_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    dit: anima_models.Anima,
):
    """Save Anima model at epoch end or specific steps."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Sampling (Euler discrete for rectified flow)
def do_sample(
    height: int,
    width: int,
    seed: Optional[int],
    dit: anima_models.Anima,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    reference_latents: Optional[list[list[torch.Tensor]]] = None,
    reference_slot_ids: Optional[list[list[int]]] = None,
    reference_t_offset_scale: int = 10,
    use_ip_adapter: bool = False,
    use_reference_sequence: bool = True,
    ip_adapter_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate a sample using Euler discrete sampling for rectified flow.

    Args:
        height, width: Output image dimensions
        seed: Random seed (None for random)
        dit: Anima model
        crossattn_emb: Cross-attention embeddings (B, N, D)
        steps: Number of sampling steps
        dtype: Compute dtype
        device: Compute device
        guidance_scale: CFG scale (1.0 = no guidance)
        flow_shift: Flow shift parameter for rectified flow
        neg_crossattn_emb: Negative cross-attention embeddings for CFG

    Returns:
        Denoised latents
    """
    # Latent shape: (1, 16, 1, H/8, W/8) for single image
    latent_h = height // 8
    latent_w = width // 8
    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Generate noise
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    noise = torch.randn(latent.size(), dtype=torch.float32, generator=generator, device="cpu").to(dtype).to(device)

    # Timestep schedule: linear from 1.0 to 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    flow_shift = float(flow_shift)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    # Start from pure noise
    x = noise.clone()

    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None
    ip_adapter_latents, ip_adapter_embeds = anima_models.resolve_ip_adapter_conditioning(
        use_ip_adapter=use_ip_adapter,
        reference_latents=reference_latents,
        ip_adapter_embeds=ip_adapter_embeds,
    )

    for i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)  # (1,)

        if use_cfg:
            # CFG: two separate passes to reduce memory usage
            pos_out = dit(
                x,
                t,
                crossattn_emb,
                padding_mask=padding_mask,
                reference_latents=reference_latents,
                reference_slot_ids=reference_slot_ids,
                reference_t_offset_scale=reference_t_offset_scale,
                ip_adapter_latents=ip_adapter_latents,
                ip_adapter_embeds=ip_adapter_embeds,
                use_ip_adapter=use_ip_adapter,
                use_reference_sequence=use_reference_sequence,
            )
            pos_out = pos_out.float()
            neg_out = dit(
                x,
                t,
                neg_crossattn_emb,
                padding_mask=padding_mask,
                reference_latents=reference_latents,
                reference_slot_ids=reference_slot_ids,
                reference_t_offset_scale=reference_t_offset_scale,
                ip_adapter_latents=ip_adapter_latents,
                ip_adapter_embeds=ip_adapter_embeds,
                use_ip_adapter=use_ip_adapter,
                use_reference_sequence=use_reference_sequence,
            )
            neg_out = neg_out.float()

            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(
                x,
                t,
                crossattn_emb,
                padding_mask=padding_mask,
                reference_latents=reference_latents,
                reference_slot_ids=reference_slot_ids,
                reference_t_offset_scale=reference_t_offset_scale,
                ip_adapter_latents=ip_adapter_latents,
                ip_adapter_embeds=ip_adapter_embeds,
                use_ip_adapter=use_ip_adapter,
                use_reference_sequence=use_reference_sequence,
            )
            model_output = model_output.float()

        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * model_output
        dt = sigmas[i + 1] - sigma
        x = x + model_output * dt
        x = x.to(dtype)

    return x


def _load_reference_sample_prompts(sample_reference_dir: str, max_references: int = 0) -> list[dict]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ref_suffix_re = re.compile(r"_ref\d*$", re.IGNORECASE)
    prompts = []
    seen_targets = set()
    for root, dirs, files in os.walk(sample_reference_dir):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
        if any(part.startswith(".") for part in Path(root).relative_to(sample_reference_dir).parts):
            continue
        image_by_stem = {}
        txt_by_stem = {}
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            path = os.path.join(root, filename)
            _, ext = os.path.splitext(path)
            stem_name = os.path.splitext(filename)[0]
            if ext.lower() not in image_exts:
                if ext.lower() == ".txt":
                    txt_by_stem[stem_name] = path
                continue

            # Use one image per stem if multiple extensions exist, and never treat
            # *_ref / *_ref1 / *_ref2 images as standalone sample prompts.
            image_by_stem.setdefault(stem_name, path)

        sample_stems = set(image_by_stem)
        sample_stems.update(txt_by_stem)
        for stem_name in sorted(sample_stems):
            if ref_suffix_re.search(stem_name):
                continue
            image_path = image_by_stem.get(stem_name)
            txt_path = txt_by_stem.get(stem_name)
            if txt_path is None and image_path is not None:
                txt_path = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.isfile(txt_path):
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if not prompt:
                continue
            prompt_dict = train_util.line_to_prompt_dict(prompt)

            ref_paths = []
            direct_ref = image_by_stem.get(f"{stem_name}_ref")
            if direct_ref is not None:
                ref_paths.append(direct_ref)
            ref_index = 1
            while True:
                ref_path = image_by_stem.get(f"{stem_name}_ref{ref_index}")
                if ref_path is None:
                    break
                ref_paths.append(ref_path)
                ref_index += 1
            if image_path is None and not ref_paths:
                continue

            ref_key = tuple(os.path.normcase(os.path.abspath(path)) for path in (ref_paths or [image_path]))
            prompt_key = (ref_key, prompt_dict.get("prompt", ""))
            if prompt_key in seen_targets:
                logger.warning(f"Skipping duplicate Anima sample prompt: {txt_path}")
                continue
            seen_targets.add(prompt_key)
            prompt_dict["image"] = ref_paths or image_path
            prompts.append(prompt_dict)
    if max_references > 0 and len(prompts) > max_references:
        import random as _random
        _random.seed(42)
        prompts = _random.sample(prompts, max_references)
        logger.info(f"Trimmed sample references to {max_references} (from {len(seen_targets)} total)")
    logger.info(f"Loaded {len(prompts)} Anima reference sample prompt(s) from: {sample_reference_dir}")
    return prompts


def _preprocess_reference_sample_image(image: Image.Image, args, dit: anima_models.Anima) -> Image.Image:
    image = image.convert("RGB")
    max_area = getattr(args, "anima_reference_max_area", 1024 * 1024)
    if max_area is not None and max_area > 0 and image.width * image.height > max_area:
        scale = (max_area / (image.width * image.height)) ** 0.5
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)

    multiple_of = qwen_image_autoencoder_kl.SCALE_FACTOR * dit.patch_spatial
    width = (image.width // multiple_of) * multiple_of
    height = (image.height // multiple_of) * multiple_of
    if width <= 0 or height <= 0:
        raise ValueError(f"Reference sample image is too small after alignment: {image.width}x{image.height}")
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _encode_sample_reference_images(prompt_dict, args, dit, vae, device, dtype):
    image_paths = prompt_dict.get("image") or prompt_dict.get("reference_image") or prompt_dict.get("reference_images")
    if image_paths is None:
        return None, []
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    refs = []
    preview_images = []
    org_vae_device = vae.device
    vae.to(device)
    try:
        for image_path in image_paths:
            image = _preprocess_reference_sample_image(Image.open(image_path), args, dit)
            preview_images.append(image.copy())
            image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=vae.dtype)
            latent = vae.encode_pixels_to_latents(image_tensor).to(device=device, dtype=dtype)
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            refs.append(latent)
    finally:
        vae.to(org_vae_device)
    return [refs], preview_images


def _get_sample_reference_paths(prompt_dict):
    image_paths = prompt_dict.get("image") or prompt_dict.get("reference_image") or prompt_dict.get("reference_images")
    if image_paths is None:
        return []
    if isinstance(image_paths, str):
        return [image_paths]
    return list(image_paths)


def _encode_sample_ip_adapter_embeds(prompt_dict, args, device, dtype):
    if not getattr(args, "anima_ip_adapter", False) or args.anima_ip_adapter_feature_backend == "vae":
        return None
    image_paths = _get_sample_reference_paths(prompt_dict)
    if not image_paths:
        return None
    extractor = getattr(args, "_anima_sample_ip_feature_extractor", None)
    if extractor is None:
        extractor = AnimaSampleIPFeatureExtractor(
            args.anima_ip_adapter_feature_backend,
            args.anima_ip_adapter_feature_model,
            device,
            dtype,
        )
        args._anima_sample_ip_feature_extractor = extractor
    return extractor.extract(image_paths)


def _make_reference_comparison(reference_images: list[Image.Image], output_image: Image.Image) -> Image.Image:
    if not reference_images:
        return output_image

    target_height = output_image.height
    resized_refs = []
    for ref in reference_images:
        ref = ref.convert("RGB")
        if ref.height != target_height:
            width = max(1, round(ref.width * target_height / ref.height))
            ref = ref.resize((width, target_height), Image.Resampling.LANCZOS)
        resized_refs.append(ref)

    ref_canvas = Image.new("RGB", (sum(ref.width for ref in resized_refs), target_height), (255, 255, 255))
    x_offset = 0
    for ref in resized_refs:
        ref_canvas.paste(ref, (x_offset, 0))
        x_offset += ref.width

    comparison = Image.new("RGB", (ref_canvas.width + output_image.width, target_height), (255, 255, 255))
    comparison.paste(ref_canvas, (0, 0))
    comparison.paste(output_image.convert("RGB"), (ref_canvas.width, 0))
    return comparison


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: anima_models.Anima,
    vae,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    on_prompt_start=None,
    on_prompt_end=None,
):
    """Generate sample images during training.

    This is a simplified sampler for Anima - it generates images using the current model state.

    on_prompt_start / on_prompt_end:
        Optional callbacks invoked around each prompt's `_sample_image_inference` call. Useful
        for injecting per-prompt state into wrapper modules (e.g. ControlNet-LLLite cond image).
        Signature: ``on_prompt_start(prompt_dict, accelerator)`` / ``on_prompt_end(prompt_dict)``.
    """
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info(f"Generating sample images at step {steps}")
    if args.anima_sample_reference_dir is None and not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    # Unwrap models
    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    dit_was_training = dit.training
    dit.eval()
    dit.switch_block_swap_for_inference()

    if args.anima_sample_reference_dir is not None:
        prompts = _load_reference_sample_prompts(args.anima_sample_reference_dir, getattr(args, "anima_sample_max_references", 0))
        if len(prompts) == 0:
            logger.error(f"No image/txt sample pairs found in: {args.anima_sample_reference_dir}")
            return
        for i, prompt_dict in enumerate(prompts):
            prompt_dict["enum"] = i
    else:
        prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in prompts:
            dit.prepare_block_swap_before_forward()
            if on_prompt_start is not None:
                on_prompt_start(prompt_dict, accelerator)
            try:
                _sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoder,
                    vae,
                    tokenize_strategy,
                    text_encoding_strategy,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
            finally:
                if on_prompt_end is not None:
                    on_prompt_end(prompt_dict)

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    if dit_was_training:
        dit.train()
    clean_memory_on_device(accelerator.device)


def _sample_image_inference(
    accelerator,
    args,
    dit,
    text_encoder,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    scale = prompt_dict.get("scale", 5.0)
    seed = prompt_dict.get("seed")
    flow_shift = prompt_dict.get("flow_shift", 3.0)
    # The integrated native-reference route intentionally keeps its trainable
    # parameters in FP32 while the frozen DiT base remains BF16.  dit.dtype
    # therefore is not a reliable sampling dtype for this heterogeneous model.
    model_dtype = dit.x_embedder.proj[1].weight.dtype

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    logger.info(
        f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, seed: {seed}"
    )
    sample_ref_paths = _get_sample_reference_paths(prompt_dict)
    if sample_ref_paths:
        logger.info(f"  reference image(s): {sample_ref_paths}")

    # Encode prompt
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            return encoded
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded

    # Convert to tensors if numpy
    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
        attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
        t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
        t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)

    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=model_dtype)
    attn_mask = attn_mask.to(accelerator.device)
    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # Process through LLM adapter if available
    if dit.use_llm_adapter:
        crossattn_emb = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0
    else:
        crossattn_emb = prompt_embeds

    # Encode negative prompt for CFG
    neg_crossattn_emb = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is not None:
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded
            if isinstance(neg_pe, np.ndarray):
                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)
                neg_am = torch.from_numpy(neg_am).unsqueeze(0)
                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)
                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)

            neg_pe = neg_pe.to(accelerator.device, dtype=model_dtype)
            neg_am = neg_am.to(accelerator.device)
            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)
            neg_t5_am = neg_t5_am.to(accelerator.device)

            if dit.use_llm_adapter:
                neg_crossattn_emb = dit.llm_adapter(
                    source_hidden_states=neg_pe,
                    target_input_ids=neg_t5_ids,
                    target_attention_mask=neg_t5_am,
                    source_attention_mask=neg_am,
                )
                neg_crossattn_emb[~neg_t5_am.bool()] = 0
            else:
                neg_crossattn_emb = neg_pe

    # Generate sample
    clean_memory_on_device(accelerator.device)
    ip_adapter_embeds = _encode_sample_ip_adapter_embeds(prompt_dict, args, accelerator.device, model_dtype)
    reference_latents, reference_preview_images = None, []
    needs_reference_latents = (
        getattr(args, "anima_use_reference_sequence", args.anima_multi_image_edit)
        or (getattr(args, "anima_ip_adapter", False) and args.anima_ip_adapter_feature_backend == "vae")
    )
    if needs_reference_latents:
        reference_latents, reference_preview_images = _encode_sample_reference_images(
            prompt_dict, args, dit, vae, accelerator.device, model_dtype
        )
    else:
        reference_preview_images = [Image.open(path).convert("RGB") for path in _get_sample_reference_paths(prompt_dict)]
    reference_slot_ids = None
    if reference_latents and reference_latents[0]:
        configured_slots = prompt_dict.get("reference_slot_ids")
        slots = list(range(len(reference_latents[0]))) if configured_slots is None else [int(x) for x in configured_slots]
        if len(slots) != len(reference_latents[0]) or any(slot < 0 for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError(
                f"Sample reference_slot_ids must be unique non-negative IDs matching the image count, got {slots}."
            )
        reference_slot_ids = [slots]
    latents = do_sample(
        height,
        width,
        seed,
        dit,
        crossattn_emb,
        sample_steps,
        model_dtype,
        accelerator.device,
        scale,
        flow_shift,
        neg_crossattn_emb,
        reference_latents=reference_latents,
        reference_slot_ids=reference_slot_ids,
        reference_t_offset_scale=args.anima_reference_t_offset_scale,
        use_ip_adapter=args.anima_ip_adapter,
        use_reference_sequence=getattr(args, "anima_use_reference_sequence", args.anima_multi_image_edit),
        ip_adapter_embeds=ip_adapter_embeds,
    )

    # Decode latents
    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    decoded = vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    # Convert to image
    image = decoded.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    # Remove temporal dim if present
    if image.ndim == 4:
        image = image[:, 0, :, :]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)

    image = Image.fromarray(decoded_np)
    save_image = _make_reference_comparison(reference_preview_images, image)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    save_image.save(os.path.join(save_dir, img_filename))

    # Log to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(save_image, caption=prompt)}, commit=False)
