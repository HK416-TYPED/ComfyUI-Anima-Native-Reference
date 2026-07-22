"""P0 multimodal conditioner for Anima.

This module deliberately does *not* use the community checkpoint's custom
``processor.py``.  That processor was authored against an older transformers
signature and expands one image to 169 tokens under transformers 4.57.6.  P0
uses a small canonical processor with an explicit, versioned role grammar and
exactly 64 image tokens per 512x512 image.

The text model is always the clean Qwen3-0.6B-Base used by Anima.  Text-only
requests bypass the vision model and connector completely, which preserves the
original Anima text-encoder path.  Multimodal requests use the community
SigLIP2 tower and native connector before entering that same clean Qwen model.
No new runtime adapter or bridge is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


ROLE_SCHEMA_VERSION = "anima-vlm-role-v1"
EXPECTED_VLM_REVISION = "047e009b91db8d59f6c0d1b54195f2fd3e7af340"
EXPECTED_VLM_SHA256 = "f3af92cd24a1d9523a70b9f980115a5dc2cf6a1b911cc5372fa9833a0f70e8df"

IMAGE_SIZE = 512
IMAGE_SEQ_LEN = 64
TEXT_ONLY_MAX_LENGTH = 512
MULTIMODAL_MAX_LENGTH = 768
HIDDEN_SIZE = 1024

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_TOKEN = "<|image_pad|>"
VISION_PAD_TOKEN = "<|vision_pad|>"

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655

_RESERVED_PROMPT_TOKENS = (VISION_START_TOKEN, VISION_END_TOKEN, IMAGE_TOKEN, VISION_PAD_TOKEN)


@dataclass(frozen=True)
class AnimaVLMRequest:
    """One explicitly routed edit request.

    ``identity_images`` always precede ``scene_image`` in both the role prompt
    and ``pixel_values``.  Callers must not infer roles from filenames such as
    ``_ref``/``_ref1``.
    """

    instruction: str
    identity_images: Sequence[Any]
    scene_image: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str):
            raise TypeError("instruction must be a string")
        if any(token in self.instruction for token in _RESERVED_PROMPT_TOKENS):
            raise ValueError("instruction contains a reserved vision token")
        if len(self.identity_images) == 0 and self.scene_image is None:
            raise ValueError("a multimodal request must contain at least one image")

    @property
    def ordered_images(self) -> Tuple[Any, ...]:
        images = tuple(self.identity_images)
        if self.scene_image is not None:
            images += (self.scene_image,)
        return images


@dataclass(frozen=True)
class AnimaVLMInputs:
    """Tensor inputs produced by :class:`AnimaCanonicalVLMProcessor`."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    pixel_attention_mask: torch.Tensor
    prompts: Tuple[str, ...]
    num_images: Tuple[int, ...]
    role_schema_version: str = ROLE_SCHEMA_VERSION

    def to(
        self,
        device: Union[str, torch.device],
        pixel_dtype: Optional[torch.dtype] = None,
    ) -> "AnimaVLMInputs":
        pixel_values = self.pixel_values.to(device=device)
        if pixel_dtype is not None:
            pixel_values = pixel_values.to(dtype=pixel_dtype)
        return replace(
            self,
            input_ids=self.input_ids.to(device=device),
            attention_mask=self.attention_mask.to(device=device),
            pixel_values=pixel_values,
            pixel_attention_mask=self.pixel_attention_mask.to(device=device),
        )

    def model_kwargs(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "pixel_attention_mask": self.pixel_attention_mask,
        }


@dataclass(frozen=True)
class AnimaVLMConditioningOutput:
    """Anima-compatible source context before the frozen native LLMAdapter."""

    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor
    is_multimodal: bool


def canonical_image_block(image_seq_len: int = IMAGE_SEQ_LEN) -> str:
    """Return the exact Qwen vision delimiter block for one image."""

    if image_seq_len != IMAGE_SEQ_LEN:
        raise ValueError(f"P0 requires exactly {IMAGE_SEQ_LEN} image tokens, got {image_seq_len}")
    return VISION_START_TOKEN + IMAGE_TOKEN * image_seq_len + VISION_END_TOKEN


def build_role_prompt(
    instruction: str,
    num_identity_images: int,
    has_scene_image: bool,
) -> str:
    """Build a role-explicit prompt with one placeholder per ordered image.

    Role text comes before its image because Qwen3 is causal.  The returned
    prompt contains one unexpanded ``<|image_pad|>`` placeholder per image;
    :func:`expand_canonical_image_placeholders` expands it deterministically.
    """

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    if num_identity_images < 0:
        raise ValueError("num_identity_images must be non-negative")
    if num_identity_images == 0 and not has_scene_image:
        raise ValueError("at least one identity or scene image is required")
    if any(token in instruction for token in _RESERVED_PROMPT_TOKENS):
        raise ValueError("instruction contains a reserved vision token")

    sections: List[str] = []
    for index in range(num_identity_images):
        if num_identity_images == 1:
            label = "Character identity and appearance reference"
        else:
            label = f"Character identity and appearance reference {index + 1} of {num_identity_images}"
        sections.append(f"{label}:\n{IMAGE_TOKEN}")

    if has_scene_image:
        sections.append(
            "Target scene, pose, composition, background and lighting reference:\n" + IMAGE_TOKEN
        )

    sections.append("Editing instruction:\n" + instruction)
    return "\n\n".join(sections)


def expand_canonical_image_placeholders(prompt: str, expected_images: int) -> str:
    """Expand role placeholders without invoking the broken remote processor."""

    actual = prompt.count(IMAGE_TOKEN)
    if actual != expected_images:
        raise ValueError(f"role prompt has {actual} image placeholders, expected {expected_images}")
    return canonical_image_block().join(prompt.split(IMAGE_TOKEN))


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        raise TypeError(f"tokenizer returned multiple IDs for special token {token!r}")
    return int(token_id)


def validate_anima_tokenizer(tokenizer: Any) -> None:
    """Assert the exact canonical Qwen vision token mapping."""

    expected = {
        VISION_START_TOKEN: VISION_START_TOKEN_ID,
        VISION_END_TOKEN: VISION_END_TOKEN_ID,
        IMAGE_TOKEN: IMAGE_TOKEN_ID,
    }
    actual = {token: _token_id(tokenizer, token) for token in expected}
    if actual != expected:
        raise RuntimeError(f"incompatible Qwen tokenizer vision token IDs: expected={expected}, actual={actual}")

    encoded = tokenizer(canonical_image_block(), add_special_tokens=False)["input_ids"]
    if len(encoded) != IMAGE_SEQ_LEN + 2:
        raise RuntimeError(
            f"canonical image block tokenized to {len(encoded)} tokens; expected {IMAGE_SEQ_LEN + 2}"
        )
    if encoded[0] != VISION_START_TOKEN_ID or encoded[-1] != VISION_END_TOKEN_ID:
        raise RuntimeError("canonical image block delimiters did not tokenize to the required IDs")
    if encoded[1:-1] != [IMAGE_TOKEN_ID] * IMAGE_SEQ_LEN:
        raise RuntimeError("canonical image block did not produce exactly 64 image-pad token IDs")


def validate_vlm_repository(vlm_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    """Validate the pinned community repository without importing its processor.

    The repository's ``processor.py`` is intentionally never imported.  In
    transformers 4.57.6 its third positional argument is ``video_processor``;
    the old code passes ``image_seq_len`` there and silently falls back to 169.
    """

    root = Path(vlm_path)
    config_path = root / "config.json"
    processor_config_path = root / "processor_config.json"
    if not config_path.is_file() or not processor_config_path.is_file():
        raise FileNotFoundError(f"missing VLM config files under {root}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    checks = {
        "model_type": config.get("model_type"),
        "image_token_id": config.get("image_token_id"),
        "scale_factor": config.get("scale_factor"),
        "text_hidden_size": config.get("text_config", {}).get("hidden_size"),
        "vision_image_size": config.get("vision_config", {}).get("image_size"),
        "image_seq_len": processor_config.get("image_seq_len"),
    }
    expected = {
        "model_type": "smolvlm",
        "image_token_id": IMAGE_TOKEN_ID,
        "scale_factor": 4,
        "text_hidden_size": HIDDEN_SIZE,
        "vision_image_size": IMAGE_SIZE,
        "image_seq_len": IMAGE_SEQ_LEN,
    }
    if checks != expected:
        raise RuntimeError(f"unexpected VLM repository configuration: expected={expected}, actual={checks}")
    return checks


def sha256_file(path: Union[str, os.PathLike[str]], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class AnimaCanonicalVLMProcessor:
    """Canonical 512px/64-token processor for explicit Anima edit roles."""

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any,
        *,
        text_only_max_length: int = TEXT_ONLY_MAX_LENGTH,
        multimodal_max_length: int = MULTIMODAL_MAX_LENGTH,
    ) -> None:
        validate_anima_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.text_only_max_length = text_only_max_length
        self.multimodal_max_length = multimodal_max_length

        # These assignments are intentional.  P0 must never inherit the
        # repository's do_image_splitting=true behavior.
        self.image_processor.do_image_splitting = False
        self.image_processor.do_resize = True
        self.image_processor.do_pad = True
        self.image_processor.size = {"longest_edge": IMAGE_SIZE}
        self.image_processor.max_image_size = {"longest_edge": IMAGE_SIZE}

    @classmethod
    def from_pretrained(
        cls,
        vlm_path: Union[str, os.PathLike[str]],
        tokenizer: Any,
        **kwargs: Any,
    ) -> "AnimaCanonicalVLMProcessor":
        from transformers import SmolVLMImageProcessor

        validate_vlm_repository(vlm_path)
        image_processor = SmolVLMImageProcessor.from_pretrained(
            vlm_path,
            local_files_only=True,
            do_image_splitting=False,
            do_resize=True,
            do_pad=True,
            size={"longest_edge": IMAGE_SIZE},
            max_image_size={"longest_edge": IMAGE_SIZE},
        )
        return cls(tokenizer=tokenizer, image_processor=image_processor, **kwargs)

    def prepare_text_only(self, text: Union[str, Sequence[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize exactly like the original Anima raw-caption path."""

        texts = [text] if isinstance(text, str) else list(text)
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.text_only_max_length,
        )
        return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}

    def prepare_multimodal(
        self,
        requests: Union[AnimaVLMRequest, Sequence[AnimaVLMRequest]],
        *,
        fixed_padding: bool = False,
    ) -> AnimaVLMInputs:
        if isinstance(requests, AnimaVLMRequest):
            requests = [requests]
        else:
            requests = list(requests)
        if not requests:
            raise ValueError("requests must not be empty")
        if not all(isinstance(request, AnimaVLMRequest) for request in requests):
            raise TypeError("all requests must be AnimaVLMRequest instances")

        expanded_prompts: List[str] = []
        nested_images: List[List[Any]] = []
        num_images: List[int] = []
        for request in requests:
            ordered_images = list(request.ordered_images)
            role_prompt = build_role_prompt(
                request.instruction,
                num_identity_images=len(request.identity_images),
                has_scene_image=request.scene_image is not None,
            )
            expanded = expand_canonical_image_placeholders(role_prompt, expected_images=len(ordered_images))
            expanded_prompts.append(expanded)
            nested_images.append(ordered_images)
            num_images.append(len(ordered_images))

        text_encoding = self.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            truncation=True,
            # Dynamic padding is the normal online path.  Fixed 768-token
            # padding is opt-in for cache formats that require a static shape.
            padding="max_length" if fixed_padding else True,
            max_length=self.multimodal_max_length,
        )
        input_ids = text_encoding["input_ids"]
        attention_mask = text_encoding["attention_mask"]

        observed_counts = (input_ids == IMAGE_TOKEN_ID).sum(dim=1).tolist()
        expected_counts = [count * IMAGE_SEQ_LEN for count in num_images]
        if observed_counts != expected_counts:
            raise RuntimeError(
                "image-token expansion/truncation mismatch: "
                f"expected={expected_counts}, observed={observed_counts}, max_length={self.multimodal_max_length}"
            )

        vision_encoding = self.image_processor(
            nested_images,
            return_tensors="pt",
            do_image_splitting=False,
            do_resize=True,
            do_pad=True,
            size={"longest_edge": IMAGE_SIZE},
            max_image_size={"longest_edge": IMAGE_SIZE},
        )
        pixel_values = vision_encoding["pixel_values"]
        pixel_attention_mask = vision_encoding["pixel_attention_mask"].bool()

        if pixel_values.ndim != 5 or tuple(pixel_values.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"unexpected pixel_values shape: {tuple(pixel_values.shape)}")
        if pixel_attention_mask.shape[:2] != pixel_values.shape[:2]:
            raise RuntimeError("pixel_attention_mask image axes do not match pixel_values")

        observed_images = pixel_attention_mask.flatten(2).any(dim=-1).sum(dim=-1).tolist()
        if observed_images != num_images:
            raise RuntimeError(f"image processor routing mismatch: expected={num_images}, observed={observed_images}")

        return AnimaVLMInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            prompts=tuple(expanded_prompts),
            num_images=tuple(num_images),
        )


class AnimaVLMConditioner(nn.Module):
    """Return 1024-d source context for Anima's frozen native LLMAdapter.

    No external adapter or bridge exists in this module.  Future visual-domain
    alignment is learned in the *native* SmolVLM connector (and optionally the
    last vision blocks).  The clean Qwen text body remains frozen.
    """

    def __init__(self, vlm_backbone: nn.Module) -> None:
        super().__init__()
        if not hasattr(vlm_backbone, "text_model"):
            raise TypeError("vlm_backbone must expose text_model")
        self.vlm_backbone = vlm_backbone
        self.freeze_qwen_text_body()

    @property
    def text_model(self) -> nn.Module:
        return self.vlm_backbone.text_model

    def freeze_qwen_text_body(self) -> None:
        self.text_model.requires_grad_(False)
        self.text_model.eval()

    def assert_qwen_text_body_frozen(self) -> None:
        trainable = [name for name, parameter in self.text_model.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"Anima Qwen text body must remain frozen; trainable keys: {trainable[:8]}")

    def configure_p0_frozen(self) -> None:
        """Freeze vision, connector and Qwen for non-training P0 probes."""

        self.vlm_backbone.requires_grad_(False)
        self.assert_qwen_text_body_frozen()
        self.vlm_backbone.eval()

    def configure_visual_alignment(self, *, train_connector: bool, vision_last_n: int = 0) -> None:
        """Expose only native visual modules for a later P1 alignment stage."""

        if vision_last_n < 0:
            raise ValueError("vision_last_n must be non-negative")
        self.vlm_backbone.requires_grad_(False)
        if train_connector:
            self.vlm_backbone.connector.requires_grad_(True)
        if vision_last_n:
            layers = self.vlm_backbone.vision_model.encoder.layers
            if vision_last_n > len(layers):
                raise ValueError(f"vision_last_n={vision_last_n} exceeds {len(layers)} vision layers")
            for layer in layers[-vision_last_n:]:
                layer.requires_grad_(True)
        self.freeze_qwen_text_body()
        self.assert_qwen_text_body_frozen()

    @staticmethod
    def _mask_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

    def encode_text_only(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> AnimaVLMConditioningOutput:
        """Bypass vision/connector and execute the original clean-Qwen path."""

        self.assert_qwen_text_body_frozen()
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = self._mask_hidden(outputs.last_hidden_state, attention_mask)
        return AnimaVLMConditioningOutput(hidden, attention_mask, is_multimodal=False)

    def encode_multimodal(self, inputs: AnimaVLMInputs) -> AnimaVLMConditioningOutput:
        self.assert_qwen_text_body_frozen()
        expected = [count * IMAGE_SEQ_LEN for count in inputs.num_images]
        observed = (inputs.input_ids == IMAGE_TOKEN_ID).sum(dim=1).tolist()
        if observed != expected:
            raise RuntimeError(f"invalid multimodal image-token counts: expected={expected}, observed={observed}")

        outputs = self.vlm_backbone(
            **inputs.model_kwargs(),
            use_cache=False,
            return_dict=True,
        )
        hidden = self._mask_hidden(outputs.last_hidden_state, inputs.attention_mask)
        if hidden.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(f"VLM hidden size must be {HIDDEN_SIZE}, got {hidden.shape[-1]}")
        return AnimaVLMConditioningOutput(hidden, inputs.attention_mask, is_multimodal=True)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
    ) -> AnimaVLMConditioningOutput:
        if pixel_values is None:
            if pixel_attention_mask is not None:
                raise ValueError("pixel_attention_mask was provided without pixel_values")
            return self.encode_text_only(input_ids, attention_mask)
        if pixel_attention_mask is None:
            raise ValueError("pixel_attention_mask is required with pixel_values")
        num_images = tuple(
            int(value)
            for value in pixel_attention_mask.bool().flatten(2).any(dim=-1).sum(dim=-1).tolist()
        )
        return self.encode_multimodal(
            AnimaVLMInputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                prompts=tuple("" for _ in range(input_ids.shape[0])),
                num_images=num_images,
            )
        )


def freeze_and_assert_anima_llm_adapter(anima_model: nn.Module) -> nn.Module:
    """Freeze the official Anima LLMAdapter without changing its structure."""

    if not hasattr(anima_model, "llm_adapter") or anima_model.llm_adapter is None:
        raise TypeError("Anima model does not expose a native llm_adapter")
    adapter = anima_model.llm_adapter
    adapter.requires_grad_(False)
    adapter.eval()
    trainable = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Anima native LLMAdapter must remain frozen: {trainable[:8]}")
    return adapter


def validate_clean_qwen_checkpoint_keys(
    text_model: nn.Module,
    qwen_path: Union[str, os.PathLike[str]],
) -> int:
    """Strictly validate every clean-Qwen key and shape without loading twice."""

    from safetensors import safe_open

    qwen_path = Path(qwen_path)
    if qwen_path.suffix != ".safetensors":
        raise ValueError("P0 strict Qwen validation requires a .safetensors checkpoint")

    expected_shapes = {key: tuple(value.shape) for key, value in text_model.state_dict().items()}
    with safe_open(qwen_path, framework="pt", device="cpu") as handle:
        raw_keys = list(handle.keys())
        normalized_to_raw = {
            key[len("model.") :] if key.startswith("model.") else key: key for key in raw_keys
        }
        actual_keys = set(normalized_to_raw)
        expected_keys = set(expected_shapes)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "clean Anima Qwen checkpoint key mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"expected_count={len(expected_keys)}, actual_count={len(actual_keys)}"
            )

        bad_shapes = []
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(handle.get_slice(normalized_to_raw[key]).get_shape())
            if actual_shape != expected_shape:
                bad_shapes.append((key, expected_shape, actual_shape))
        if bad_shapes:
            raise RuntimeError(f"clean Anima Qwen checkpoint shape mismatch: {bad_shapes[:8]}")

    return len(expected_keys)


def load_anima_vlm_conditioner(
    vlm_path: Union[str, os.PathLike[str]],
    anima_qwen_path: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    verify_weight_sha256: bool = False,
    preserve_package_connector_dtype: bool = False,
) -> Tuple[AnimaVLMConditioner, AnimaCanonicalVLMProcessor]:
    """Load community vision+connector and replace its text body with Anima Qwen.

    The full conditional-generation wrapper is discarded immediately after
    loading, so inference never constructs vocabulary-sized logits.
    """

    from transformers import AutoTokenizer, SmolVLMForConditionalGeneration, SmolVLMModel

    from _anima_native_ref_vendor.library import anima_utils

    vlm_path = Path(vlm_path)
    validate_vlm_repository(vlm_path)
    package_metadata_path = vlm_path / "anima_vlm_package.json"
    is_full_p1_package = package_metadata_path.is_file()
    if is_full_p1_package:
        package_metadata = json.loads(package_metadata_path.read_text(encoding="utf-8"))
        required_package_values = {
            "format": "anima-vlm-p1-full-v1",
            "model_class": "SmolVLMModel",
            "includes_vision_model": True,
            "includes_native_connector": True,
            "includes_clean_anima_qwen": True,
            "runtime_external_adapter_required": False,
            "clean_qwen_key_count": 310,
        }
        actual_package_values = {key: package_metadata.get(key) for key in required_package_values}
        if actual_package_values != required_package_values:
            raise RuntimeError(
                f"invalid full Anima VLM package metadata: expected={required_package_values}, "
                f"actual={actual_package_values}"
            )
        tokenizer_parity = package_metadata.get("tokenizer_parity")
        if not isinstance(tokenizer_parity, dict) or tokenizer_parity.get("verified") is not True:
            raise RuntimeError("full Anima VLM package is missing verified tokenizer-ID parity metadata")
        if tokenizer_parity.get("fix_mistral_regex") is not False:
            raise RuntimeError("full Anima VLM package must pin fix_mistral_regex=false")

        support_records = package_metadata.get("support_files")
        if not isinstance(support_records, list) or not support_records:
            raise RuntimeError("full Anima VLM package is missing tokenizer/config/processor file hashes")
        support_names = {record.get("name") for record in support_records if isinstance(record, dict)}
        required_support_names = {
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        }
        if not required_support_names.issubset(support_names):
            raise RuntimeError(
                "full Anima VLM package support-file metadata is incomplete: "
                f"missing={sorted(required_support_names - support_names)}"
            )

        if verify_weight_sha256:
            records = list(package_metadata.get("model_files", [])) + support_records
            if not package_metadata.get("model_files"):
                raise RuntimeError("full Anima VLM package has no model-file hash records")
            for record in records:
                if not isinstance(record, dict) or set(("name", "size", "sha256")) - set(record):
                    raise RuntimeError(f"invalid full-package file record: {record!r}")
                if Path(record["name"]).name != record["name"]:
                    raise RuntimeError(f"full-package file record must be a top-level filename: {record!r}")
                model_file = vlm_path / record["name"]
                if not model_file.is_file() or model_file.stat().st_size != record["size"]:
                    raise RuntimeError(f"full-package file size mismatch: {model_file}")
                actual_hash = sha256_file(model_file)
                if actual_hash != record["sha256"]:
                    raise RuntimeError(
                        f"full-package SHA256 mismatch for {model_file.name}: "
                        f"expected={record['sha256']}, actual={actual_hash}"
                    )

        # P1 packages already contain the exact clean Anima Qwen text body.
        # Loading SmolVLMModel directly avoids both an LM head and any external
        # runtime Qwen dependency.
        if preserve_package_connector_dtype:
            # Load via FP32 modules so a package's FP32 connector master
            # weights are not silently quantized to BF16 during resume. Frozen
            # vision/Qwen tensors are cast back to the requested runtime dtype.
            backbone = SmolVLMModel.from_pretrained(vlm_path, local_files_only=True)
            backbone.vision_model.to(dtype=dtype)
            backbone.text_model.to(dtype=dtype)
            backbone.connector.to(dtype=torch.float32)
        else:
            backbone = SmolVLMModel.from_pretrained(
                vlm_path,
                local_files_only=True,
                dtype=dtype,
            )
        # The outer config is SmolVLM while the embedded text body/tokenizer is
        # the original Anima Qwen3 tokenizer.  transformers 4.57.6 otherwise
        # guesses the Mistral regex migration from the outer config and warns
        # (or may change tokenization in later versions).  False is deliberate:
        # it preserves the exact tokenizer IDs used by the clean Anima model.
        tokenizer = AutoTokenizer.from_pretrained(
            vlm_path,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        backbone.config.text_config = backbone.text_model.config
        backbone.config.use_cache = False
        backbone.text_model.config.use_cache = False
        conditioner = AnimaVLMConditioner(backbone)
        conditioner.configure_p0_frozen()
        if preserve_package_connector_dtype:
            conditioner.to(device=device)
            conditioner.vlm_backbone.vision_model.to(dtype=dtype)
            conditioner.text_model.to(dtype=dtype)
            conditioner.vlm_backbone.connector.to(dtype=torch.float32)
        else:
            conditioner.to(device=device, dtype=dtype)
        processor = AnimaCanonicalVLMProcessor.from_pretrained(vlm_path, tokenizer)
        return conditioner, processor

    if vlm_path.parent.name == "snapshots" and vlm_path.name != EXPECTED_VLM_REVISION:
        raise RuntimeError(f"P0 is pinned to VLM revision {EXPECTED_VLM_REVISION}, got {vlm_path.name}")
    weight_path = vlm_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing VLM weights: {weight_path}")
    if verify_weight_sha256:
        actual_hash = sha256_file(weight_path)
        if actual_hash != EXPECTED_VLM_SHA256:
            raise RuntimeError(f"VLM weight SHA256 mismatch: expected={EXPECTED_VLM_SHA256}, actual={actual_hash}")

    # Never load trust_remote_code and never import the repository processor.py.
    causal_lm = SmolVLMForConditionalGeneration.from_pretrained(
        vlm_path,
        local_files_only=True,
        dtype=dtype,
    )
    backbone = causal_lm.model

    # Replace (do not fine-tune) the community-married Qwen with the exact clean
    # Anima Qwen3-0.6B-Base.  Keeping a single shared module also makes the
    # text-only route mathematically identical to the existing encoder path.
    if anima_qwen_path is None:
        raise ValueError("anima_qwen_path is required when loading the original community VLM")
    clean_qwen, tokenizer = anima_utils.load_qwen3_text_encoder(
        str(anima_qwen_path),
        dtype=dtype,
        device="cpu",
    )
    validated_qwen_keys = validate_clean_qwen_checkpoint_keys(clean_qwen, anima_qwen_path)
    if validated_qwen_keys != 310:
        raise RuntimeError(f"expected 310 clean Anima Qwen keys, validated {validated_qwen_keys}")
    backbone.text_model = clean_qwen
    # Publish/serialize the clean Anima Qwen config, not the community-married
    # text config (which has different EOS and max-position metadata).
    backbone.config.text_config = clean_qwen.config
    backbone.config.use_cache = False
    backbone.text_model.config.use_cache = False

    # Drop the language head and the old community text embedding retained by
    # it; only SmolVLMModel.last_hidden_state is used by Anima.
    del causal_lm
    gc.collect()

    conditioner = AnimaVLMConditioner(backbone)
    conditioner.configure_p0_frozen()
    conditioner.to(device=device, dtype=dtype)
    processor = AnimaCanonicalVLMProcessor.from_pretrained(vlm_path, tokenizer)
    return conditioner, processor
