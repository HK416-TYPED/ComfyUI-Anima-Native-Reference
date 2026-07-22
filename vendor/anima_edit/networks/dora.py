# DoRA (Weight-Decomposed Low-Rank Adaptation) network module.
# Reference: https://arxiv.org/abs/2402.09353
#
# This follows the local LoRA/LoKr module style: the original module forward is
# patched in-place, while the trainable network owns only adapter parameters.

import ast
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_base import AdditionalNetwork, detect_arch_config, _parse_kv_pairs
from _anima_native_ref_vendor.library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _weight_magnitude(weight: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(weight.detach().float().flatten(1), dim=1)


def _normalize_direction(direction: torch.Tensor, magnitude: torch.Tensor, eps: float, detach_norm: bool = False) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction.float().flatten(1), dim=1).clamp_min(eps)
    if detach_norm:
        norm = norm.detach()
    scale = magnitude.float() / norm
    return direction.float() * scale.view(-1, *([1] * (direction.dim() - 1)))


def _comfy_dora_scale_from_magnitude(
    org_weight: torch.Tensor,
    lora_weight: torch.Tensor,
    magnitude: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    org_norm = torch.linalg.vector_norm(org_weight.float().flatten(1), dim=1).clamp_min(eps)
    direction = org_weight.float() + lora_weight.float()
    direction_norm = torch.linalg.vector_norm(direction.flatten(1), dim=1).clamp_min(eps)
    dora_scale = magnitude.float().flatten() * org_norm / direction_norm
    return dora_scale.view(-1, *([1] * (org_weight.dim() - 1)))


def _magnitude_from_comfy_dora_scale(
    org_weight: torch.Tensor,
    lora_weight: torch.Tensor,
    dora_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    org_norm = torch.linalg.vector_norm(org_weight.float().flatten(1), dim=1).clamp_min(eps)
    direction = org_weight.float() + lora_weight.float()
    direction_norm = torch.linalg.vector_norm(direction.flatten(1), dim=1).clamp_min(eps)
    dora_scale = dora_scale.float().reshape(org_weight.shape[0], -1)[:, 0]
    return dora_scale * direction_norm / org_norm


def _scale_for_output(scale: torch.Tensor, output: torch.Tensor, is_conv: bool) -> torch.Tensor:
    if is_conv:
        return scale.view(1, -1, *([1] * (output.dim() - 2)))
    return scale.view(*([1] * (output.dim() - 1)), -1)


def _lora_scale(alpha, rank: int) -> float:
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().float().item()
    alpha = rank if alpha is None or alpha == 0 else float(alpha)
    return alpha / rank


def _build_lora_weight(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scale: float,
    is_conv: bool,
    conv_mode: Optional[str],
    out_dim: int,
    in_dim: int,
    kernel_size=None,
) -> torch.Tensor:
    if not is_conv:
        return (up_weight @ down_weight) * scale

    if conv_mode == "1x1":
        down = down_weight.squeeze(3).squeeze(2)
        up = up_weight.squeeze(3).squeeze(2)
        return (up @ down).unsqueeze(2).unsqueeze(3) * scale

    conved = F.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
    return conved.reshape(out_dim, in_dim, *kernel_size) * scale


class DoRAModule(nn.Module):
    """DoRA module for training. Replaces Linear/Conv2d forward methods."""

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        eps=1e-6,
        dora_simple=True,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = int(lora_dim)
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.eps = float(eps)
        self.dora_simple = _as_bool(dora_simple, default=True)
        self.enabled = True
        self.org_module_ref = [org_module]

        self.is_conv = org_module.__class__.__name__ == "Conv2d"
        if self.is_conv:
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
            self.conv_mode = "1x1" if self.kernel_size == (1, 1) else "flat"
            self.lora_down = nn.Conv2d(self.in_dim, self.lora_dim, self.kernel_size, self.stride, self.padding, bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, self.out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
            self.kernel_size = None
            self.stride = None
            self.padding = None
            self.dilation = None
            self.groups = None
            self.conv_mode = None
            self.lora_down = nn.Linear(self.in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, self.out_dim, bias=False)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / self.lora_dim
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        self.dora_magnitude = nn.Parameter(torch.empty(self.out_dim))
        self.reset_parameters(org_module.weight)

    def reset_parameters(self, org_weight: torch.Tensor):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        with torch.no_grad():
            self.dora_magnitude.copy_(_weight_magnitude(org_weight).to(self.dora_magnitude.device, self.dora_magnitude.dtype))

    def apply_to(self):
        self.org_forward = self.org_module_ref[0].forward
        self.org_module_ref[0].forward = self.forward

    def _drop_rank(self, up_weight: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.rank_dropout is None or not self.training:
            return up_weight, 1.0
        keep = (torch.rand(self.lora_dim, device=up_weight.device) > self.rank_dropout).to(up_weight.dtype)
        if self.is_conv:
            up_weight = up_weight * keep.view(1, -1, 1, 1)
        else:
            up_weight = up_weight * keep.view(1, -1)
        return up_weight, 1.0 / (1.0 - self.rank_dropout)

    def get_lora_weight(self, multiplier=None, use_rank_dropout=False) -> torch.Tensor:
        if multiplier is None:
            multiplier = self.multiplier
        down_weight = self.lora_down.weight
        up_weight = self.lora_up.weight
        rank_scale = 1.0
        if use_rank_dropout:
            up_weight, rank_scale = self._drop_rank(up_weight)
        return _build_lora_weight(
            down_weight,
            up_weight,
            self.scale * float(multiplier) * rank_scale,
            self.is_conv,
            self.conv_mode,
            self.out_dim,
            self.in_dim,
            self.kernel_size,
        )

    def get_effective_weight(self, multiplier=None, use_rank_dropout=False) -> torch.Tensor:
        org_weight = self.org_module_ref[0].weight
        lora_weight = self.get_lora_weight(multiplier=multiplier, use_rank_dropout=use_rank_dropout).to(
            device=org_weight.device, dtype=torch.float32
        )
        direction = org_weight.float() + lora_weight
        return _normalize_direction(
            direction,
            self.dora_magnitude.to(org_weight.device),
            self.eps,
            detach_norm=self.training and self.dora_simple,
        )

    def get_comfy_dora_scale(self, multiplier=1.0) -> torch.Tensor:
        org_weight = self.org_module_ref[0].weight
        lora_weight = self.get_lora_weight(multiplier=multiplier, use_rank_dropout=False).to(
            device=org_weight.device, dtype=torch.float32
        )
        return _comfy_dora_scale_from_magnitude(
            org_weight,
            lora_weight,
            self.dora_magnitude.to(org_weight.device),
            self.eps,
        )

    def magnitude_from_comfy_dora_scale(self, dora_scale: torch.Tensor, multiplier=1.0) -> torch.Tensor:
        org_weight = self.org_module_ref[0].weight
        lora_weight = self.get_lora_weight(multiplier=multiplier, use_rank_dropout=False).to(
            device=org_weight.device, dtype=torch.float32
        )
        return _magnitude_from_comfy_dora_scale(org_weight, lora_weight, dora_scale.to(org_weight.device), self.eps)

    def get_norm_scale(self, multiplier=None, use_rank_dropout=False) -> torch.Tensor:
        org_weight = self.org_module_ref[0].weight
        lora_weight = self.get_lora_weight(multiplier=multiplier, use_rank_dropout=use_rank_dropout).to(
            device=org_weight.device, dtype=torch.float32
        )
        direction = org_weight.float() + lora_weight
        norm = torch.linalg.vector_norm(direction.flatten(1), dim=1).clamp_min(self.eps)
        if self.training and self.dora_simple:
            norm = norm.detach()
        return self.dora_magnitude.to(org_weight.device).float() / norm

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        if self.module_dropout is not None and self.training and torch.rand(1, device=x.device) < self.module_dropout:
            return self.org_forward(x)

        org_module = self.org_module_ref[0]
        use_dropout = self.dropout is not None and self.training and float(self.dropout) > 0
        if use_dropout:
            org_forwarded = self.org_forward(x)
            dropout_x = F.dropout(x, p=float(self.dropout), training=True)

            up_weight = self.lora_up.weight
            rank_scale = 1.0
            if self.rank_dropout is not None:
                up_weight, rank_scale = self._drop_rank(up_weight)
            lora_weight = _build_lora_weight(
                self.lora_down.weight,
                up_weight,
                self.scale * float(self.multiplier) * rank_scale,
                self.is_conv,
                self.conv_mode,
                self.out_dim,
                self.in_dim,
                self.kernel_size,
            ).to(device=org_module.weight.device, dtype=torch.float32)
            direction = org_module.weight.float() + lora_weight
            direction_norm = torch.linalg.vector_norm(direction.flatten(1), dim=1).clamp_min(self.eps)
            if self.dora_simple:
                direction_norm = direction_norm.detach()
            norm_scale = (self.dora_magnitude.to(org_module.weight.device).float() / direction_norm).to(
                device=x.device, dtype=org_forwarded.dtype
            )
            out_scale = _scale_for_output(norm_scale, org_forwarded, self.is_conv)

            if self.is_conv:
                base_dropout = F.conv2d(
                    dropout_x,
                    org_module.weight.to(device=x.device, dtype=dropout_x.dtype),
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                lora_hidden = self.lora_down(dropout_x)
                delta = F.conv2d(lora_hidden, up_weight.to(device=x.device, dtype=lora_hidden.dtype), bias=None)
            else:
                base_dropout = F.linear(
                    dropout_x,
                    org_module.weight.to(device=x.device, dtype=dropout_x.dtype),
                    bias=None,
                )
                lora_hidden = self.lora_down(dropout_x)
                delta = F.linear(lora_hidden, up_weight.to(device=x.device, dtype=lora_hidden.dtype), bias=None)

            delta = delta * (self.scale * float(self.multiplier) * rank_scale)
            return org_forwarded + (out_scale - 1.0) * base_dropout + out_scale * delta

        weight = self.get_effective_weight(use_rank_dropout=True).to(device=x.device, dtype=x.dtype)
        bias = org_module.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        if self.is_conv:
            return F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return F.linear(x, weight, bias)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class DoRAInfModule(DoRAModule):
    """DoRA module for inference/merging."""

    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1, **kwargs):
        eps = kwargs.pop("eps", 1e-6)
        dora_simple = kwargs.pop("dora_simple", True)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, eps=eps, dora_simple=dora_simple)
        self.network: AdditionalNetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        org_sd = self.org_module_ref[0].state_dict()
        org_weight = org_sd["weight"]
        org_dtype = org_weight.dtype
        org_device = org_weight.device
        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        down_weight = sd["lora_down.weight"].to(device=device, dtype=torch.float32)
        up_weight = sd["lora_up.weight"].to(device=device, dtype=torch.float32)
        alpha = sd.get("alpha", self.alpha)
        scale = _lora_scale(alpha, down_weight.shape[0])
        base_lora_weight = _build_lora_weight(
            down_weight,
            up_weight,
            scale,
            self.is_conv,
            self.conv_mode,
            self.out_dim,
            self.in_dim,
            self.kernel_size,
        )
        org_weight_float = org_weight.to(device=device, dtype=torch.float32)
        if "dora_magnitude" in sd:
            direction = org_weight_float + base_lora_weight * float(self.multiplier)
            magnitude = sd["dora_magnitude"].to(device=device, dtype=torch.float32)
            merged = _normalize_direction(direction, magnitude, self.eps)
        else:
            dora_scale = sd["dora_scale"].to(device=device, dtype=torch.float32).reshape(self.out_dim, -1)[:, 0]
            direction = org_weight_float + base_lora_weight
            org_norm = torch.linalg.vector_norm(org_weight_float.flatten(1), dim=1).clamp_min(self.eps)
            full_merged = direction * (dora_scale / org_norm).view(-1, *([1] * (direction.dim() - 1)))
            merged = org_weight_float + float(self.multiplier) * (full_merged - org_weight_float)
        org_sd["weight"] = merged.to(dtype=dtype)
        self.org_module_ref[0].load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        org_weight = self.org_module_ref[0].weight
        effective = self.get_effective_weight(multiplier=multiplier).to(device=org_weight.device, dtype=torch.float32)
        return effective - org_weight.float()

    def default_forward(self, x):
        org_module = self.org_module_ref[0]
        weight = self.get_effective_weight().to(device=x.device, dtype=x.dtype)
        bias = org_module.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        if self.is_conv:
            return F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return F.linear(x, weight, bias)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class DoRANetwork(AdditionalNetwork):
    """AdditionalNetwork variant that saves DoRA in ComfyUI-compatible format."""

    def _convert_comfy_state_dict_for_load(self, weights_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted = dict(weights_sd)
        for lora in self.text_encoder_loras + self.unet_loras:
            prefix = lora.lora_name
            magnitude_key = f"{prefix}.dora_magnitude"
            dora_scale_key = f"{prefix}.dora_scale"
            if magnitude_key in converted:
                converted.pop(dora_scale_key, None)
                continue
            if dora_scale_key not in converted:
                continue

            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            if down_key not in converted or up_key not in converted:
                continue

            org_weight = lora.org_module_ref[0].weight
            down_weight = converted[down_key].to(device=org_weight.device, dtype=torch.float32)
            up_weight = converted[up_key].to(device=org_weight.device, dtype=torch.float32)
            alpha = converted.get(f"{prefix}.alpha", lora.alpha)
            lora_weight = _build_lora_weight(
                down_weight,
                up_weight,
                _lora_scale(alpha, down_weight.shape[0]),
                lora.is_conv,
                lora.conv_mode,
                lora.out_dim,
                lora.in_dim,
                lora.kernel_size,
            )
            converted[magnitude_key] = _magnitude_from_comfy_dora_scale(
                org_weight,
                lora_weight,
                converted[dora_scale_key].to(device=org_weight.device, dtype=torch.float32),
                lora.eps,
            ).to(dtype=lora.dora_magnitude.dtype, device="cpu")
            converted.pop(dora_scale_key, None)
        return converted

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        weights_sd = self._convert_comfy_state_dict_for_load(weights_sd)
        info = self.load_state_dict(weights_sd, False)
        return info

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            prefix = lora.lora_name
            state_dict[f"{prefix}.lora_down.weight"] = lora.lora_down.weight.detach().clone()
            state_dict[f"{prefix}.lora_up.weight"] = lora.lora_up.weight.detach().clone()
            state_dict[f"{prefix}.alpha"] = lora.alpha.detach().clone()
            state_dict[f"{prefix}.dora_scale"] = lora.get_comfy_dora_scale(multiplier=1.0).detach().clone()

        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to("cpu").to(dtype)
        else:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to("cpu")

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from _anima_native_ref_vendor.library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_dora_format"] = "comfyui"
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoder,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a DoRA network. Called by train_network.py via network_module.create_network()."""
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    train_llm_adapter = _as_bool(kwargs.get("train_llm_adapter", "false"))

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]
    exclude_patterns.extend(arch_config.default_excludes)
    if "JointTransformerBlock" in arch_config.unet_target_modules:
        train_text_refiner = _as_bool(kwargs.get("train_text_refiner", "true"), default=True)
        train_noise_refiner = _as_bool(kwargs.get("train_noise_refiner", "true"), default=True)
        if not train_text_refiner:
            exclude_patterns.append(r".*context_refiner.*")
        if not train_noise_refiner:
            exclude_patterns.append(r".*noise_refiner.*")

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    conv_lora_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_lora_dim is not None:
        conv_lora_dim = int(conv_lora_dim)
        conv_alpha = 1.0 if conv_alpha is None else float(conv_alpha)

    eps = float(kwargs.get("eps", kwargs.get("dora_eps", 1e-6)))
    dora_simple = _as_bool(kwargs.get("dora_simple", "true"), default=True)

    verbose = _as_bool(kwargs.get("verbose", "false"))
    reg_lrs = _parse_kv_pairs(kwargs.get("network_reg_lrs"), is_int=False) if kwargs.get("network_reg_lrs") is not None else None
    reg_dims = _parse_kv_pairs(kwargs.get("network_reg_dims"), is_int=True) if kwargs.get("network_reg_dims") is not None else None

    network = DoRANetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=DoRAModule,
        module_kwargs={"eps": eps, "dora_simple": dora_simple},
        conv_lora_dim=conv_lora_dim,
        conv_alpha=conv_alpha,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    """Create a DoRA network from saved weights. Called by train_network.py."""
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif key.endswith(".lora_down.weight"):
            modules_dim[lora_name] = value.shape[0]
        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)
    eps = float(kwargs.get("eps", kwargs.get("dora_eps", 1e-6)))
    dora_simple = _as_bool(kwargs.get("dora_simple", "true"), default=True)
    module_class = DoRAInfModule if for_inference else DoRAModule

    network = DoRANetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs={"eps": eps, "dora_simple": dora_simple},
        train_llm_adapter=train_llm_adapter,
    )
    return network, weights_sd
