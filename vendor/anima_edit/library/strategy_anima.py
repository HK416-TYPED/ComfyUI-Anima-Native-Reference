# Anima Strategy Classes

import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from _anima_native_ref_vendor.library import anima_utils, train_util
from _anima_native_ref_vendor.library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy
from _anima_native_ref_vendor.library import qwen_image_autoencoder_kl

from _anima_native_ref_vendor.library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def preprocess_anima_reference_image(
    image,
    *,
    max_area: Optional[int],
    multiple_of: int,
    target_size_hw: Optional[Tuple[int, int]] = None,
    flipped: bool = False,
) -> Image.Image:
    """Apply the one canonical preprocessing path used by online and cached refs.

    ``target_size_hw`` is used for the single-reference edit path, where the
    reference must use the target bucket geometry. Multi-reference inputs keep
    their own aspect ratio, are area-limited, and are centre-cropped to the VAE
    and DiT patch multiple. Keeping this helper shared is important: a cached
    latent must be bit-identical to what the previous online path encoded.
    """

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(image[:, :, :3]).convert("RGB")

    if target_size_hw is not None:
        target_h, target_w = (int(target_size_hw[0]), int(target_size_hw[1]))
        image = ImageOps.fit(
            image,
            (target_w, target_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        if flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    if max_area is not None and max_area > 0 and image.width * image.height > max_area:
        scale = math.sqrt(max_area / (image.width * image.height))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    width = (image.width // multiple_of) * multiple_of
    height = (image.height // multiple_of) * multiple_of
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Reference image is too small after alignment: {image.width}x{image.height}. "
            f"Both dimensions must be at least {multiple_of}px."
        )

    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen3_max_length
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.t5_max_length
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]
        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens and return embeddings + T5 token IDs.

        Args:
            models: [qwen3_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        """
        # Do not handle dropout here; handled dataset-side or in drop_cached_text_encoder_outputs()

        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior.
        """
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

        # Clone to avoid in-place modification of cached tensors
        prompt_embeds = prompt_embeds.clone()
        if attn_mask is not None:
            attn_mask = attn_mask.clone()
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.clone()
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.clone()

        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                # Use pre-cached unconditional embeddings
                prompt_embeds[i] = 0
                if attn_mask is not None:
                    attn_mask[i] = 0
                if t5_input_ids is not None:
                    t5_input_ids[i, 0] = 1  # Set to </s> token ID
                    t5_input_ids[i, 1:] = 0
                if t5_attn_mask is not None:
                    t5_attn_mask[i, 0] = 1
                    t5_attn_mask[i, 1:] = 0

        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (float), attn_mask (int), t5_input_ids (int), t5_attn_mask (int)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "prompt_embeds" not in npz:
                return False
            if "attn_mask" not in npz:
                return False
            if "t5_input_ids" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
            if "caption_dropout_rate" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        prompt_embeds = data["prompt_embeds"]
        attn_mask = data["attn_mask"]
        t5_input_ids = data["t5_input_ids"]
        t5_attn_mask = data["t5_attn_mask"]
        caption_dropout_rate = data["caption_dropout_rate"]
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = anima_text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks
            )

        # Convert to numpy for caching
        if prompt_embeds.dtype == torch.bfloat16:
            prompt_embeds = prompt_embeds.float()
        prompt_embeds = prompt_embeds.cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()
        t5_input_ids = t5_input_ids.cpu().numpy().astype(np.int32)
        t5_attn_mask = t5_attn_mask.cpu().numpy().astype(np.int32)

        for i, info in enumerate(infos):
            prompt_embeds_i = prompt_embeds[i]
            attn_mask_i = attn_mask[i]
            t5_input_ids_i = t5_input_ids[i]
            t5_attn_mask_i = t5_attn_mask[i]
            caption_dropout_rate = torch.tensor(info.caption_dropout_rate, dtype=torch.float32)

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    prompt_embeds=prompt_embeds_i,
                    attn_mask=attn_mask_i,
                    t5_input_ids=t5_input_ids_i,
                    t5_attn_mask=t5_attn_mask_i,
                    caption_dropout_rate=caption_dropout_rate,
                )
            else:
                info.text_encoder_outputs = (prompt_embeds_i, attn_mask_i, t5_input_ids_i, t5_attn_mask_i, caption_dropout_rate)


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using the Qwen Image VAE.

    Target latents keep the upstream cache behavior. Ordered reference latents
    may additionally be cached in host RAM; that mode deliberately rejects disk
    caching so one sample cannot silently mix differently quantized cache paths.
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        *,
        cache_reference_latents: bool = False,
        reference_max_area: Optional[int] = 1024 * 1024,
        reference_multiple_of: int = qwen_image_autoencoder_kl.SCALE_FACTOR * 2,
        sync_single_reference_geometry: bool = True,
        reference_preprocess_workers: int = 0,
    ) -> None:
        if cache_reference_latents and cache_to_disk:
            raise ValueError("Ordered reference latent caching is RAM-only; disable --cache_latents_to_disk.")
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.cache_reference_latents = bool(cache_reference_latents)
        self.reference_max_area = reference_max_area
        self.reference_multiple_of = int(reference_multiple_of)
        self.sync_single_reference_geometry = bool(sync_single_reference_geometry)
        self.reference_batch_size = max(1, int(batch_size or 1))
        self.reference_preprocess_workers = int(reference_preprocess_workers or 0)
        if self.reference_preprocess_workers < 0:
            raise ValueError("reference_preprocess_workers must be non-negative (0 keeps serial preprocessing).")

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.ANIMA_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def _cache_reference_latents(self, vae, image_infos: List, flip_aug: bool = False) -> int:
        """Cache explicit references in physical list order and return their count.

        Images are grouped by their post-preprocessing geometry so VAE calls can
        use ``vae_batch_size`` without padding or changing pixels. Assignment to
        ``ImageInfo`` happens only after every image has encoded successfully.
        """

        if not self.cache_reference_latents:
            return 0
        if flip_aug:
            raise ValueError("Reference latent caching requires flip augmentation to be disabled.")

        grouped: Dict[Tuple[int, int], List[Tuple[int, int, torch.Tensor, str]]] = defaultdict(list)
        cached_paths: List[List[str]] = []
        pending: List[List[Optional[torch.Tensor]]] = []
        preprocess_jobs: List[Tuple[int, int, str, Optional[Tuple[int, int]]]] = []

        for info_index, info in enumerate(image_infos):
            paths = list(getattr(info, "reference_image_paths", None) or [])
            if not paths or any(not path for path in paths):
                raise ValueError(
                    "--anima_cache_reference_latents requires non-empty explicit reference_image_paths "
                    f"for every sample; missing for {getattr(info, 'absolute_path', '<unknown>')}."
                )
            cached_paths.append(paths)
            pending.append([None] * len(paths))

            target_size_hw = None
            if self.sync_single_reference_geometry and len(paths) == 1:
                bucket_reso = getattr(info, "bucket_reso", None)
                if bucket_reso is None:
                    raise ValueError(f"Missing bucket_reso for synchronized reference: {info.absolute_path}")
                target_size_hw = (int(bucket_reso[1]), int(bucket_reso[0]))

            for slot_index, path in enumerate(paths):
                preprocess_jobs.append((info_index, slot_index, path, target_size_hw))

        def load_and_preprocess(job: Tuple[int, int, str, Optional[Tuple[int, int]]]):
            info_index, slot_index, path, target_size_hw = job
            image = preprocess_anima_reference_image(
                train_util.load_image(path),
                max_area=self.reference_max_area,
                multiple_of=self.reference_multiple_of,
                target_size_hw=target_size_hw,
                flipped=False,
            )
            tensor = train_util.IMAGE_TRANSFORMS(image)
            return info_index, slot_index, tensor, path

        # executor.map yields in input order. Geometry-group insertion and
        # per-group order therefore match the serial path even when individual
        # decodes finish out of order. Zero remains the conservative default.
        if self.reference_preprocess_workers:
            with ThreadPoolExecutor(
                max_workers=self.reference_preprocess_workers,
                thread_name_prefix="anima-ref-preprocess",
            ) as executor:
                preprocessed = executor.map(load_and_preprocess, preprocess_jobs)
                for info_index, slot_index, tensor, path in preprocessed:
                    grouped[(int(tensor.shape[1]), int(tensor.shape[2]))].append(
                        (info_index, slot_index, tensor, path)
                    )
        else:
            for job in preprocess_jobs:
                info_index, slot_index, tensor, path = load_and_preprocess(job)
                grouped[(int(tensor.shape[1]), int(tensor.shape[2]))].append(
                    (info_index, slot_index, tensor, path)
                )

        vae_device = vae.device
        vae_dtype = vae.dtype
        with torch.no_grad():
            for entries in grouped.values():
                for offset in range(0, len(entries), self.reference_batch_size):
                    chunk = entries[offset : offset + self.reference_batch_size]
                    pixels = torch.stack([entry[2] for entry in chunk]).to(vae_device, dtype=vae_dtype)
                    latents = vae.encode_pixels_to_latents(pixels)
                    if latents.ndim == 5:
                        if latents.shape[2] != 1:
                            raise ValueError(f"Reference VAE produced non-image temporal shape {tuple(latents.shape)}")
                        latents = latents.squeeze(2)
                    if latents.ndim != 4 or latents.shape[0] != len(chunk):
                        raise ValueError(
                            f"Reference VAE returned {tuple(latents.shape)} for a batch of {len(chunk)} images."
                        )
                    for latent, (info_index, slot_index, _tensor, path) in zip(latents, chunk):
                        latent = latent.detach().to("cpu").contiguous()
                        if not torch.isfinite(latent).all():
                            raise ValueError(f"Reference VAE produced non-finite latent for {path}")
                        pending[info_index][slot_index] = latent

        for info, paths, sample_latents in zip(image_infos, cached_paths, pending):
            if any(latent is None for latent in sample_latents):
                raise RuntimeError(f"Incomplete reference latent cache for {info.absolute_path}")
            info.reference_latent_paths = list(paths)
            info.reference_latents = list(sample_latents)
        return sum(len(paths) for paths in cached_paths)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        """Cache target latents and, when requested, ordered reference latents."""
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            latents = vae.encode_pixels_to_latents(img_tensor)
            return latents.to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )
        reference_count = self._cache_reference_latents(vae, image_infos, flip_aug=flip_aug)
        if reference_count:
            logger.info("cached %d ordered reference latents in host RAM", reference_count)

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
