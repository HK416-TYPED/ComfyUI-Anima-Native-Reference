import argparse
import datetime
import gc
import json
import math
from importlib.util import find_spec
import random
import os
import sys
import time
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Optional, List, Any, Dict, Union

import torch
import numpy as np
from torchvision import transforms
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from _anima_native_ref_vendor.library import anima_models, anima_utils, hunyuan_image_utils, qwen_image_autoencoder_kl, strategy_anima, strategy_base
from _anima_native_ref_vendor.library.device_utils import clean_memory_on_device, synchronize_device

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights

from _anima_native_ref_vendor.library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class IPAdapterFeatureExtractor:
    def __init__(self, backend: str, model_dir: Optional[str], device: torch.device, dtype: torch.dtype):
        self.backend = backend
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.model = None
        self.transform = None
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

        checkpoint = IPAdapterFeatureExtractor._find_ccip_checkpoint(model_path)
        logger.info(f"Loading CCIP token backbone from: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = IPAdapterFeatureExtractor._clean_ccip_state_dict(state_dict)
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
    def _load_siglip2_tokens(model_path, device):
        from transformers import Siglip2VisionModel, Siglip2ImageProcessor

        logger.info(f"Loading SigLIP2 vision model from: {model_path}")
        processor = Siglip2ImageProcessor.from_pretrained(model_path)
        vision_model = Siglip2VisionModel.from_pretrained(model_path)
        vision_model.to(device).eval()
        feature_dim = vision_model.config.hidden_size
        logger.info(f"Loaded SigLIP2 vision model. feature_dim={feature_dim}")
        return vision_model, processor, feature_dim

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
        from lsnet_lib.inference_artist import (
            load_checkpoint_state,
            load_model,
            normalize_state_dict_keys,
            resolve_feature_dim,
            resolve_num_classes,
        )
        from lsnet_lib.lsnet_model.lsnet_artist import default_cfgs_artist

        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            model_name = json.load(f)["model"]
        checkpoint = IPAdapterFeatureExtractor._find_lsnet_checkpoint(model_dir)
        state_dict = normalize_state_dict_keys(load_checkpoint_state(checkpoint))
        feature_dim = resolve_feature_dim(None, state_dict)
        args = SimpleNamespace(
            model=model_name,
            checkpoint=checkpoint,
            num_classes=resolve_num_classes(None, None, state_dict),
            feature_dim=feature_dim,
            mode="cluster",
            allow_head_reinit=True,
            device=str(device),
        )
        model = load_model(args, state_dict).to(device).eval()
        input_size = default_cfgs_artist.get(model_name, {}).get("input_size", (3, 448 if model_name.endswith("_448") else 224, 224))[1]
        transform = create_transform(**resolve_data_config({"input_size": (3, input_size, input_size)}, model=model))
        return model, transform, feature_dim

    def extract(self, paths):
        if not paths or self.backend == "vae":
            return None
        if self.backend == "ccip":
            features = self.extract_fn(paths, model=self.model_dir)
            return torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device, dtype=self.dtype)
        if self.backend == "ccip_tokens":
            tensors = []
            for path in paths:
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
            images = [Image.open(path).convert("RGB") for path in paths]
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
            tensors = [self.transform(Image.open(path).convert("RGB")) for path in paths]
            with torch.no_grad():
                features = self.model(torch.stack(tensors).to(self.device), return_features=True)
            return features.unsqueeze(0).to(dtype=self.dtype)
        raise ValueError(f"Unsupported IP-Adapter feature backend: {self.backend}")


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
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
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder 1 (Qwen2.5-VL) directory or path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")

    # inference
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier free guidance. Default is 3.5."
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt for generation, default is empty string")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument(
        "--reference_image",
        "--reference_images",
        type=str,
        nargs="*",
        default=None,
        help="Ordered reference image list: entries are logical Image 1, Image 2, ...; slots have no fixed semantic role.",
    )
    parser.add_argument(
        "--reference_slot_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based logical slot ID per --reference_image (defaults to 0..N-1).",
    )
    parser.add_argument("--reference_max_area", type=int, default=1024 * 1024, help="maximum reference image area before downscale")
    parser.add_argument(
        "--reference_t_offset_scale",
        type=int,
        default=10,
        help="Legacy concatenation-only T-coordinate spacing; ignored by native reference checkpoints.",
    )
    parser.add_argument(
        "--native_reference_scale",
        type=float,
        default=1.0,
        help="Global scale for integrated native reference conditioning.",
    )
    parser.add_argument("--no_reference_sequence", action="store_true", help="Do not concatenate reference tokens into the main image sequence")
    parser.add_argument("--ip_adapter", action="store_true", help="Use reference image(s) through Anima IP-Adapter instead of token concat")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0, help="Scale for Anima IP-Adapter conditioning")
    parser.add_argument("--ip_adapter_weight", type=str, default=None, help="Anima IP-Adapter safetensors weight path")
    parser.add_argument("--ip_adapter_feature_backend", type=str, default="vae", choices=["vae", "ccip", "ccip_tokens", "lsnet", "siglip2_tokens"])
    parser.add_argument("--ip_adapter_feature_model", type=str, default=None, help="CCIP, CCIP token, or LSNet model directory/checkpoint")
    parser.add_argument("--ip_adapter_feature_dim", type=int, default=None)
    parser.add_argument("--ip_adapter_num_tokens", type=int, default=4)
    parser.add_argument("--linear-adapter", "--ip_adapter_linear_adapter", dest="linear_adapter", action="store_true")
    parser.add_argument("--mlp-adapter", "--mlp_adapter", "--ip_adapter_mlp_adapter", dest="mlp_adapter", action="store_true")
    parser.add_argument(
        "--norm-linear-adapter",
        "--norm_linear_adapter",
        "--ip_adapter_norm_linear_adapter",
        dest="norm_linear_adapter",
        action="store_true",
    )
    parser.add_argument(
        "--omni-adapter",
        "--omni_adapter",
        "--ip_adapter_omni_adapter",
        dest="omni_adapter",
        action="store_true",
    )
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Shift factor for flow matching schedulers. Default is 5.0.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")

    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoders")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument(
        "--lycoris", action="store_true", help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}"
    )

    # arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "i" or option == "image" or option == "ref":
            overrides["reference_image"] = [v.strip() for v in value.split(",") if v.strip()]

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    # Anima's Qwen VAE downsamples by 8 and the DiT uses 2x2 latent
    # patches, so the actual pixel multiple is 16.  Requiring 32 here
    # rejected valid training buckets such as 208x304 and made exact-aspect
    # evaluation impossible.
    required_multiple = qwen_image_autoencoder_kl.SCALE_FACTOR * 2
    if height % required_multiple != 0 or width % required_multiple != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by {required_multiple} but are {height} and {width}."
        )

    return height, width


def preprocess_reference_image(image: Image.Image, args: argparse.Namespace, anima: anima_models.Anima) -> Image.Image:
    image = image.convert("RGB")
    max_area = getattr(args, "reference_max_area", 1024 * 1024)
    if max_area is not None and max_area > 0 and image.width * image.height > max_area:
        scale = math.sqrt(max_area / (image.width * image.height))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)

    multiple_of = qwen_image_autoencoder_kl.SCALE_FACTOR * anima.patch_spatial
    width = (image.width // multiple_of) * multiple_of
    height = (image.height // multiple_of) * multiple_of
    if width <= 0 or height <= 0:
        raise ValueError(f"Reference image is too small after alignment: {image.width}x{image.height}")

    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def encode_reference_latents(
    args: argparse.Namespace,
    anima: anima_models.Anima,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[list[list[torch.Tensor]]], list[Image.Image]]:
    if not args.reference_image:
        return None, []

    refs = []
    preview_images = []
    org_vae_device = vae.device
    vae.to(device)
    try:
        for image_path in args.reference_image:
            image = preprocess_reference_image(Image.open(image_path), args, anima)
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


# region Model


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> anima_models.Anima:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        anima_models.Anima: DiT model instance
    """
    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    loading_device = "cpu"
    if not args.lycoris:
        loading_device = device

    # load LoRA weights
    if not args.lycoris and args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            # lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}  # only keep unet lora weights
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    ip_feature_dim = args.ip_adapter_feature_dim
    if args.ip_adapter and args.ip_adapter_feature_backend != "vae":
        extractor = IPAdapterFeatureExtractor(args.ip_adapter_feature_backend, args.ip_adapter_feature_model, device, torch.bfloat16)
        args._ip_adapter_feature_extractor = extractor
        ip_feature_dim = ip_feature_dim or extractor.feature_dim
        if ip_feature_dim is None:
            raise ValueError("Could not infer IP-Adapter feature dim. Please pass --ip_adapter_feature_dim.")

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8

    if sum(bool(getattr(args, name, False)) for name in ("linear_adapter", "mlp_adapter", "norm_linear_adapter", "omni_adapter")) > 1:
        raise ValueError("Only one of --linear-adapter, --mlp-adapter, --norm-linear-adapter, and --omni-adapter can be enabled.")

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        args.attn_mode,
        True,  # enable split_attn to trim masked tokens
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
        enable_ip_adapter=args.ip_adapter,
        ip_adapter_scale=args.ip_adapter_scale,
        ip_adapter_feature_dim=ip_feature_dim,
        ip_adapter_num_tokens=args.ip_adapter_num_tokens,
        ip_adapter_linear_adapter=getattr(args, "linear_adapter", False),
        ip_adapter_mlp_adapter=getattr(args, "mlp_adapter", False),
        ip_adapter_norm_linear_adapter=getattr(args, "norm_linear_adapter", False),
        ip_adapter_omni_adapter=getattr(args, "omni_adapter", False),
        native_reference_scale=getattr(args, "native_reference_scale", 1.0),
    )
    if args.ip_adapter_weight is not None:
        if not args.ip_adapter:
            raise ValueError("--ip_adapter_weight requires --ip_adapter")
        logger.info(f"Loading Anima IP-Adapter weight from: {args.ip_adapter_weight}")
        ip_sd = load_file(args.ip_adapter_weight)
        model.load_state_dict(ip_sd, strict=False)
    if not args.fp8_scaled:
        # simple cast to dit_weight_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        logger.info(f"Move model to device: {device}")
        target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    # model.to(device)
    model.to(device, dtype=torch.bfloat16)  # ensure model is in bfloat16 for inference

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def load_text_encoder(
    args: argparse.Namespace, dtype: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    lora_weights_list = None
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            # lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_sd = {
                "model_" + k[len("lora_te_") :]: v for k, v in lora_sd.items() if k.startswith("lora_te_")
            }  # only keep Text Encoder lora weights, remove prefix "lora_te_" and add "model_" prefix
            lora_weights_list.append(lora_sd)

    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder, dtype=dtype, device=device, lora_weights=lora_weights_list, lora_multipliers=args.lora_multiplier
    )
    text_encoder.eval()
    return text_encoder


# endregion


def decode_latent(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, latent: torch.Tensor, device: torch.device
) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
        # pixels = vae.decode(latent.to(device, dtype=torch.bfloat16), scale=vae_scale)
    if pixels.ndim == 5:  # remove frame dimension if exists, [B, C, F, H, W] -> [B, C, H, W]
        pixels = pixels.squeeze(2)

    pixels = pixels.to("cpu", dtype=torch.float32)  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def process_escape(text: str) -> str:
    """Process escape sequences in text

    Args:
        text: Input text with escape sequences

    Returns:
        str: Processed text
    """
    return text.encode("utf-8").decode("unicode_escape")


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, anima: anima_models.Anima, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for T2I: LLM encoding. Anima model is also needed for preprocessing"""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        text_encoder = shared_models.get("text_encoder")

        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
        text_encoder = load_text_encoder(args, dtype=text_encoder_dtype, device=text_encoder_device)
        text_encoder.eval()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        # Store references so load_target_model can reuse them

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder_original_device = text_encoder.device if text_encoder else None

    # Ensure text_encoder is not None before proceeding
    if not text_encoder:
        raise ValueError("Text encoder is not loaded properly.")

    # Define a function to move models to device if needed
    # This is to avoid moving models if not needed, especially in interactive mode
    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving Text Encoder to appropriate device: {text_encoder_device}")
        text_encoder.to(text_encoder_device)  # If text_encoder_cpu is True, this will be CPU

    logger.info("Encoding prompt with Text Encoder")

    prompt = process_escape(args.prompt)
    cache_key = prompt
    if cache_key in conds_cache:
        embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            # embed = anima_text_encoder.get_text_embeds(anima, tokenizer, text_encoder, t5xxl_tokenizer, prompt)
            tokens = tokenize_strategy.tokenize(prompt)
            embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            crossattn_emb = anima._preprocess_text_embeds(
                source_hidden_states=embed[0].to(anima.device),
                target_input_ids=embed[2].to(anima.device),
                target_attention_mask=embed[3].to(anima.device),
                source_attention_mask=embed[1].to(anima.device),
            )
            crossattn_emb[~embed[3].bool()] = 0
            embed[0] = crossattn_emb
        embed[0] = embed[0].cpu()

        conds_cache[cache_key] = embed

    negative_prompt = process_escape(args.negative_prompt)
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            # negative_embed = anima_text_encoder.get_text_embeds(anima, tokenizer, text_encoder, t5xxl_tokenizer, negative_prompt)
            tokens = tokenize_strategy.tokenize(negative_prompt)
            negative_embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            crossattn_emb = anima._preprocess_text_embeds(
                source_hidden_states=negative_embed[0].to(anima.device),
                target_input_ids=negative_embed[2].to(anima.device),
                target_attention_mask=negative_embed[3].to(anima.device),
                source_attention_mask=negative_embed[1].to(anima.device),
            )
            crossattn_emb[~negative_embed[3].bool()] = 0
            negative_embed[0] = crossattn_emb
        negative_embed[0] = negative_embed[0].cpu()

        conds_cache[cache_key] = negative_embed

    if not (shared_models and "text_encoder" in shared_models):  # if loaded locally
        # There is a bug text_encoder is not freed from GPU memory when text encoder is fp8
        del text_encoder
        gc.collect()  # This may force Text Encoder to be freed from GPU memory
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_image_data: Optional dictionary with precomputed image data
        precomputed_text_data: Optional dictionary with precomputed text data

    Returns:
        tuple: (HunyuanVAE2D model (vae) or None, torch.Tensor generated latent)
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        anima = load_dit_model(args, device, dit_weight_dtype)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        # use shared model
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    reference_latents = None
    ip_adapter_embeds = None
    reference_preview_images = []
    if args.reference_image:
        if args.ip_adapter and args.ip_adapter_feature_backend != "vae":
            extractor = getattr(args, "_ip_adapter_feature_extractor", None)
            if extractor is None:
                extractor = IPAdapterFeatureExtractor(args.ip_adapter_feature_backend, args.ip_adapter_feature_model, device, torch.bfloat16)
            ip_adapter_embeds = extractor.extract(args.reference_image)
            reference_preview_images = [Image.open(path).convert("RGB") for path in args.reference_image]
            if not args.no_reference_sequence:
                if shared_models is not None and "vae" in shared_models:
                    vae = shared_models["vae"]
                else:
                    vae = qwen_image_autoencoder_kl.load_vae(
                        args.vae,
                        device="cpu",
                        disable_mmap=True,
                        spatial_chunk_size=args.vae_chunk_size,
                        disable_cache=args.vae_disable_cache,
                    )
                    vae.to(torch.bfloat16)
                    vae.eval()
                    if shared_models is not None:
                        shared_models["vae"] = vae
                reference_latents, _ = encode_reference_latents(args, anima, vae, device, torch.bfloat16)
        else:
            if shared_models is not None and "vae" in shared_models:
                vae = shared_models["vae"]
            else:
                vae = qwen_image_autoencoder_kl.load_vae(
                    args.vae,
                    device="cpu",
                    disable_mmap=True,
                    spatial_chunk_size=args.vae_chunk_size,
                    disable_cache=args.vae_disable_cache,
                )
                vae.to(torch.bfloat16)
                vae.eval()
                if shared_models is not None:
                    shared_models["vae"] = vae
            reference_latents, reference_preview_images = encode_reference_latents(args, anima, vae, device, torch.bfloat16)
    args._reference_preview_images = reference_preview_images

    return generate_body(args, anima, context, context_null, device, seed, reference_latents, ip_adapter_embeds)


def resolve_inference_reference_slot_ids(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    reference_latents: Optional[list[list[torch.Tensor]]],
) -> Optional[list[list[int]]]:
    """Resolve generic ordered image slots for the single inference sample."""

    if not reference_latents or not reference_latents[0]:
        return None
    reference_count = len(reference_latents[0])
    configured = getattr(args, "reference_slot_ids", None)
    slot_ids = list(range(reference_count)) if configured is None else list(configured)
    if len(slot_ids) != reference_count:
        raise ValueError(
            f"--reference_slot_ids supplied {len(slot_ids)} IDs for {reference_count} reference images."
        )
    if any(slot_id < 0 for slot_id in slot_ids):
        raise ValueError("Reference slot IDs must be zero-based non-negative integers.")
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("Reference slot IDs must be unique within one sample.")
    max_images = getattr(anima, "native_reference_max_images", None)
    if getattr(anima, "native_reference_conditioning_enabled", False) and max_images is not None:
        invalid = [slot_id for slot_id in slot_ids if slot_id >= max_images]
        if invalid:
            raise ValueError(
                f"Reference slot IDs {invalid} exceed checkpoint capacity (max_reference_images={max_images})."
            )
    return [slot_ids]


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
    reference_latents: Optional[list[list[torch.Tensor]]] = None,
    ip_adapter_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prepare latent variables
    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    shape = (
        1,
        num_channels_latents,
        1,  # Frame dimension
        height // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
        width // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
    )
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    # Create padding mask
    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device)

    logger.info(f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}")
    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    # Prepare timesteps
    timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    timesteps /= 1000  # scale to [0,1] range
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    # Denoising loop
    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8
    ip_adapter_latents, ip_adapter_embeds = anima_models.resolve_ip_adapter_conditioning(
        use_ip_adapter=args.ip_adapter,
        reference_latents=reference_latents,
        ip_adapter_embeds=ip_adapter_embeds,
    )
    reference_slot_ids = resolve_inference_reference_slot_ids(args, anima, reference_latents)

    with tqdm(total=len(timesteps), desc="Denoising steps") as pbar:
        for i, t in enumerate(timesteps):
            t_expand = t.expand(latents.shape[0])

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                noise_pred = anima(
                    latents,
                    t_expand,
                    embed,
                    padding_mask=padding_mask,
                    reference_latents=reference_latents,
                    reference_slot_ids=reference_slot_ids,
                    reference_t_offset_scale=args.reference_t_offset_scale,
                    ip_adapter_latents=ip_adapter_latents,
                    ip_adapter_embeds=ip_adapter_embeds,
                    use_ip_adapter=args.ip_adapter,
                    use_reference_sequence=not args.no_reference_sequence,
                )

            if do_cfg:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    uncond_noise_pred = anima(
                        latents,
                        t_expand,
                        negative_embed,
                        padding_mask=padding_mask,
                        reference_latents=reference_latents,
                        reference_slot_ids=reference_slot_ids,
                        reference_t_offset_scale=args.reference_t_offset_scale,
                        ip_adapter_latents=ip_adapter_latents,
                        ip_adapter_embeds=ip_adapter_embeds,
                        use_ip_adapter=args.ip_adapter,
                        use_reference_sequence=not args.no_reference_sequence,
                    )
                noise_pred = uncond_noise_pred + args.guidance_scale * (noise_pred - uncond_noise_pred)

            # ensure latents dtype is consistent
            latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)

            pbar.update()

    return latents


def get_time_flag():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            # "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"

    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C

    image = Image.fromarray(x)
    save_image = image
    reference_preview_images = getattr(args, "_reference_preview_images", [])
    if reference_preview_images:
        target_height = image.height
        resized_refs = []
        for ref in reference_preview_images:
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

        comparison = Image.new("RGB", (ref_canvas.width + image.width, target_height), (255, 255, 255))
        comparison.paste(ref_canvas, (0, 0))
        comparison.paste(image.convert("RGB"), (ref_canvas.width, 0))
        save_image = comparison

    save_image.save(os.path.join(save_path, f"{image_name}.png"))

    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
    original_base_name: Optional[str] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_name: original base name (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8  # qwen_image_autoencoder_kl.SCALE_FACTOR
    width *= 8  # qwen_image_autoencoder_kl.SCALE_FACTOR
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    if latent.ndim == 2:  # S,C. For packed latents from other inference scripts
        latent = latent.unsqueeze(0)
        height, width = check_inputs(args)  # Get height/width from args
        latent = latent.view(
            1,
            vae.latent_channels,
            1,  # Frame dimension
            height // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
            width // 8,  # qwen_image_autoencoder_kl.SCALE_FACTOR,
        )

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        # save images
        if original_base_name is None:
            original_name = ""
        else:
            original_name = f"_{original_base_name}"
        save_images(image, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models (text/image encoders)
    """
    shared_models = {}
    # Load text encoders to CPU
    text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
    text_encoder = load_text_encoder(args, dtype=text_encoder_dtype, device=torch.device("cpu"))
    shared_models["text_encoder"] = text_encoder
    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
    )
    vae_for_batch.to(torch.bfloat16)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)  # Validate each prompt's height/width

    # 2. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    anima = load_dit_model(first_prompt_args, device, dit_weight_dtype)  # Load directly to target device if possible

    shared_models_for_generate = {"model": anima}  # Pass DiT via shared_models

    # 3. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    # Text Encoder loaded to CPU by load_text_encoder
    text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
    text_encoder_batch = load_text_encoder(args, dtype=text_encoder_dtype, device=torch.device("cpu"))

    # Text Encoder to device for this phase
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_batch.to(text_encoder_device)  # Moved into prepare_text_inputs logic

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "text_encoder": text_encoder_batch,  # on GPU if not text_encoder_cpu
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")

        # prepare_text_inputs will move text_encoders to device temporarily
        context, context_null = prepare_text_inputs(prompt_args_item, device, anima, temp_shared_models_txt)
        text_data = {"context": context, "context_null": context_null}
        all_precomputed_text_data.append(text_data)

    # Models should be removed from device after prepare_text_inputs
    del text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_text_data = all_precomputed_text_data[i]
            height, width = check_inputs(prompt_args_item)  # Get height/width for each prompt

            logger.info(f"Generating latent for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                # generate is called with precomputed data, so it won't load Text Encoders.
                # It will use the DiT model from shared_models_for_generate.
                latent = generate(prompt_args_item, gen_settings, shared_models_for_generate, current_text_data)

                if latent is None:  # and prompt_args_item.save_merged_model:  # Should be caught earlier
                    continue

                # Save latent if needed (using data from precomputed_image_data for H/W)
                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    save_latent(latent, prompt_args_item, height, width)

                all_latents.append(latent)
            except Exception as e:
                logger.error(f"Error generating latent for prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                all_latents.append(None)  # Add placeholder for failed generations
                continue

    # Free DiT model
    logger.info("Releasing DiT model from memory...")

    del shared_models_for_generate["model"]
    del anima
    clean_memory_on_device(device)
    synchronize_device(device)  # Ensure memory is freed before loading VAE for decoding

    # 4. Decode latents and save outputs (using vae_for_batch)
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images using batched VAE...")
        vae_for_batch.to(device)  # Move VAE to device for decoding

        for i, latent in enumerate(all_latents):
            if latent is None:  # Skip failed generations
                logger.warning(f"Skipping decoding for prompt {i+1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i+1}/{len(all_latents)} for prompt: {current_args.prompt}")

            # if args.output_type is "latent_images", we already saved latent above.
            # so we skip saving latent here.
            if current_args.output_type == "latent_images":
                current_args.output_type = "images"

            # save_output expects latent to be [BCTHW] or [CTHW]. generate returns [BCTHW] (batch size 1).
            save_output(current_args, vae_for_batch, latent, device)  # Pass vae_for_batch

        vae_for_batch.to("cpu")  # Move VAE back to CPU

    del vae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}  # Initialize empty cache for interactive mode

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
    )
    vae.to(torch.bfloat16)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError  # Exit on Ctrl+D or Ctrl+Z

                # Parse prompt
                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Generate latent
                # For interactive, precomputed data is None. shared_models contains text encoders.
                latent = generate(prompt_args, gen_settings, shared_models)

                # # If not one_frame_inference, move DiT model to CPU after generation
                # model = shared_models.get("model")
                # model.to("cpu")  # Move DiT model to CPU after generation

                # Save latent and video
                # returned_vae from generate will be used for decoding here.
                save_output(prompt_args, vae, latent, device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        # assert len(args.latent_path) == 1, "Only one latent path is supported for now"

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device=device,
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        vae.to(torch.bfloat16)
        vae.eval()

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]
            save_output(args, vae, latent, device, original_base_names[i])

    else:
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder, t5_tokenizer_path=None, qwen3_max_length=512, t5_max_length=512
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

        if args.from_file:
            # Batch mode from file

            # Read prompts from file
            with open(args.from_file, "r", encoding="utf-8") as f:
                prompt_lines = f.readlines()

            # Process prompts
            prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
            process_batch_prompts(prompts_data, args)

        elif args.interactive:
            # Interactive mode
            process_interactive(args)

        else:
            # Single prompt mode (original behavior)

            # Generate latent
            gen_settings = get_generation_settings(args)

            # For single mode, precomputed data is None, shared_models is None.
            # generate will load all necessary models (Text Encoders, DiT).
            latent = generate(args, gen_settings)

            clean_memory_on_device(device)

            # Save latent and video
            vae = qwen_image_autoencoder_kl.load_vae(
                args.vae,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=args.vae_chunk_size,
                disable_cache=args.vae_disable_cache,
            )
            vae.to(torch.bfloat16)
            vae.eval()
            save_output(args, vae, latent, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
