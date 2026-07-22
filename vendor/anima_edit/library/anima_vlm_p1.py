"""Connector-only P1 alignment utilities for Anima VLM.

P1 aligns the native SmolVLM connector to the clean Anima Qwen hidden
manifold.  It intentionally introduces no learned projection, bridge, or
runtime adapter.  Qwen, SigLIP2, and the official Anima LLMAdapter remain
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from _anima_native_ref_vendor.library.anima_vlm import (
    AnimaVLMConditioner,
    AnimaVLMRequest,
    IMAGE_SEQ_LEN,
    ROLE_SCHEMA_VERSION,
    VISION_END_TOKEN,
    VISION_END_TOKEN_ID,
    build_role_prompt,
    canonical_image_block,
    expand_canonical_image_placeholders,
    freeze_and_assert_anima_llm_adapter,
)


P1_VISUAL_QUERY = (
    "Observe the anime image and summarize its character identity, appearance, pose, "
    "scene, background, lighting, and visual style.\nVisual semantic summary:"
)
P1_QUERY_BANK = (
    P1_VISUAL_QUERY,
    "Identify the anime character attributes, pose, setting, illumination, and visual style.\n"
    "Visual semantic summary:",
    "Read the anime image and condense the subject appearance and surrounding scene into semantics.\n"
    "Visual semantic summary:",
    "Represent the visible character, action, composition, background, lighting, and style.\n"
    "Visual semantic summary:",
)
P1_EVAL_QUERY_BANK = (
    "Give a compact semantic account of the anime subject and environment shown.\nImage meaning summary:",
    "Extract the visible identity traits, pose, scene layout, light, and artistic treatment.\nImage meaning summary:",
    "Encode what is visibly depicted in this anime illustration without inventing details.\nImage meaning summary:",
    "State the image-level semantics for the character and the scene around them.\nImage meaning summary:",
)
P1_NEUTRAL_LLM_ADAPTER_INSTRUCTION = "Describe the visual content neutrally."
P1_PACKAGE_FORMAT = "anima-vlm-p1-full-v1"
P1_PACKAGE_FILENAME = "anima_vlm_package.json"
P1_PROCESSOR_FILENAME = "processor_config.json"
EXPECTED_CONNECTOR_PARAMETERS = 12_582_912  # Linear(12288 -> 1024, bias=False)
P1_BLOCKED_TEACHER_TAGS = frozenset(
    {
        "artist_name",
        "signature",
        "watermark",
        "character_name",
        "copyright_name",
    }
)


class DeterministicEpochBatchSampler(Sampler[List[int]]):
    """Exact-resume batches from deterministic ``seed + epoch`` permutations."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int,
        start_global_batch: int = 0,
        num_batches: int,
    ) -> None:
        if dataset_size < batch_size or batch_size < 2:
            raise ValueError("dataset_size must be >= batch_size >= 2")
        if start_global_batch < 0 or num_batches <= 0:
            raise ValueError("invalid sampler range")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_global_batch = start_global_batch
        self.num_batches = num_batches
        self.batches_per_epoch = dataset_size // batch_size

    def batch_at(self, global_batch: int) -> List[int]:
        if global_batch < 0:
            raise ValueError("global_batch must be non-negative")
        epoch, batch_in_epoch = divmod(global_batch, self.batches_per_epoch)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        permutation = torch.randperm(self.dataset_size, generator=generator)
        start = batch_in_epoch * self.batch_size
        return permutation[start : start + self.batch_size].tolist()

    def __iter__(self):
        for global_batch in range(
            self.start_global_batch,
            self.start_global_batch + self.num_batches,
        ):
            yield self.batch_at(global_batch)

    def __len__(self) -> int:
        return self.num_batches


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(sample_ids).encode("utf-8")).hexdigest()


def normalize_caption_key(caption: str) -> str:
    """Match the manifest builder's comma-tag de-duplication protocol."""

    folded_tags = caption.strip().casefold().split(",")
    return ",".join(" ".join(tag.split()) for tag in folded_tags)


def _normalized_teacher_tag(tag: str) -> str:
    return " ".join(tag.strip().casefold().split())


def blocked_teacher_tag_count(caption: str) -> int:
    return sum(
        _normalized_teacher_tag(tag) in P1_BLOCKED_TEACHER_TAGS
        for tag in caption.split(",")
    )


def filter_teacher_caption_tags(caption: str) -> str:
    """Remove exact non-visual metadata tags from teacher supervision only."""

    kept = [
        tag.strip()
        for tag in caption.split(",")
        if tag.strip() and _normalized_teacher_tag(tag) not in P1_BLOCKED_TEACHER_TAGS
    ]
    # Keep a valid prefix even for a pathological all-meta custom caption.
    return ", ".join(kept) if kept else "anime illustration"


def select_p1_queries(
    sample_ids: Sequence[str],
    *,
    seed: int,
    query_bank: Sequence[str] = P1_QUERY_BANK,
) -> List[str]:
    if not query_bank or any(not query for query in query_bank):
        raise ValueError("query_bank must contain non-empty queries")
    queries = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
        queries.append(query_bank[int.from_bytes(digest[:8], "big") % len(query_bank)])
    return queries


def select_p1_batch_query(
    sample_ids: Sequence[str],
    *,
    seed: int,
    query_bank: Sequence[str] = P1_QUERY_BANK,
) -> str:
    """Choose one shared query so contrastive negatives differ only by image.

    Sorting makes the choice invariant to row order while still binding it to
    the exact batch membership and seed. Across deterministic batches the hash
    distributes coverage over the complete query bank.
    """

    if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("shared-query selection requires >=2 unique sample IDs")
    if not query_bank or any(not query for query in query_bank):
        raise ValueError("query_bank must contain non-empty queries")
    payload = f"{seed}\0" + "\0".join(sorted(sample_ids))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return query_bank[int.from_bytes(digest[:8], "big") % len(query_bank)]


def build_p1_student_requests(
    images: Sequence[Any],
    queries: Optional[Sequence[str]] = None,
) -> List[AnimaVLMRequest]:
    if not images:
        raise ValueError("images must not be empty")
    if queries is None:
        queries = [P1_VISUAL_QUERY] * len(images)
    if len(queries) != len(images):
        raise ValueError("queries length must match images")
    return [
        AnimaVLMRequest(
            instruction=query,
            identity_images=(image,),
            scene_image=None,
        )
        for image, query in zip(images, queries, strict=True)
    ]


def tokenize_teacher_captions(
    tokenizer: Any,
    captions: Sequence[str],
    *,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Tokenize raw captions with dynamic right-padding and no role/chat text."""

    if not captions:
        raise ValueError("captions must not be empty")
    encoding = tokenizer(
        list(captions),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}


def _query_tail_spec(tokenizer: Any, query: str) -> Tuple[List[int], List[int]]:
    """Return exact post-image tail IDs and query-only relative positions."""

    prefix = VISION_END_TOKEN + "\n\nEditing instruction:\n"
    full_tail = prefix + query
    encoded = tokenizer(
        full_tail,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not ids or ids[0] != VISION_END_TOKEN_ID:
        raise RuntimeError("P1 query tail does not begin with canonical vision_end")
    query_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if index > 0 and start >= len(prefix) and end > start
    ]
    if not query_positions:
        raise RuntimeError("P1 visual query tokenized to no query positions")
    if min(query_positions) <= 0:
        raise RuntimeError("P1 visual query mask includes the vision delimiter")
    return ids, query_positions


def locate_student_query_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    queries: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Locate only fixed-query tokens strictly after the final image block.

    The role header, image tokens, and ``Editing instruction:`` header are
    validated but excluded from the returned mask.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be matching rank-2 tensors")
    if queries is None:
        queries = [P1_VISUAL_QUERY] * input_ids.shape[0]
    if len(queries) != input_ids.shape[0]:
        raise ValueError("queries length must match the batch")
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)

    for batch_index in range(input_ids.shape[0]):
        tail_ids, query_relative_positions = _query_tail_spec(tokenizer, queries[batch_index])
        valid_positions = attention_mask[batch_index].bool().nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise RuntimeError(f"student row {batch_index} has no valid tokens")
        valid_end = int(valid_positions[-1].item()) + 1
        vision_positions = (
            (input_ids[batch_index, :valid_end] == VISION_END_TOKEN_ID)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if vision_positions.numel() == 0:
            raise RuntimeError(f"student row {batch_index} has no vision_end token")
        last_vision_end = int(vision_positions[-1].item())
        actual_tail = input_ids[batch_index, last_vision_end:valid_end].tolist()
        if actual_tail != tail_ids:
            raise RuntimeError(
                f"student row {batch_index} does not end in the fixed P1 query tail; "
                f"expected {len(tail_ids)} tokens, observed {len(actual_tail)}"
            )
        absolute_query = [last_vision_end + position for position in query_relative_positions]
        if min(absolute_query) <= last_vision_end:
            raise RuntimeError("student query mask is not strictly after final vision_end")
        mask[batch_index, absolute_query] = True

    if torch.any(mask & ~attention_mask.bool()):
        raise RuntimeError("student query mask selected padded tokens")
    return mask


def masked_summary(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool selected valid tokens and L2-normalize onto one manifold."""

    if hidden_states.ndim != 3 or mask.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_states must be [B,L,H] and mask [B,L]")
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each summary row must select at least one token")
    pooled = (hidden_states.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1).float()
    return F.normalize(pooled, dim=-1, eps=1e-6)


def teacher_caption_summary(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Summarize every valid raw-caption token from clean Qwen."""

    return masked_summary(hidden_states, attention_mask.bool())


def student_visual_summary(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    queries: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_mask = locate_student_query_mask(input_ids, attention_mask, tokenizer, queries)
    return masked_summary(hidden_states, query_mask), query_mask


@dataclass(frozen=True)
class MatchedTeacherInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    query_mask: torch.Tensor
    untruncated_lengths: Tuple[int, ...]
    caption_untruncated_lengths: Tuple[int, ...]

    def to(self, device: Union[str, torch.device]) -> "MatchedTeacherInputs":
        return MatchedTeacherInputs(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            query_mask=self.query_mask.to(device),
            untruncated_lengths=self.untruncated_lengths,
            caption_untruncated_lengths=self.caption_untruncated_lengths,
        )


def tokenize_matched_teacher_queries(
    tokenizer: Any,
    captions: Sequence[str],
    queries: Sequence[str],
    *,
    max_length: int = 256,
) -> MatchedTeacherInputs:
    """Build clean-Qwen teacher inputs with the same suffix as the student.

    Caption text is prefix-only supervision. Pooling selects only the matched
    query suffix. Left truncation, used for the <1% long captions, guarantees
    the query itself is never silently removed.
    """

    if not captions or len(captions) != len(queries):
        raise ValueError("captions and queries must be matching non-empty sequences")
    fixed_prefix = "Visual description:\n"
    texts = [
        f"{fixed_prefix}{caption}\n\n{query}"
        for caption, query in zip(captions, queries, strict=True)
    ]
    caption_untruncated = tokenizer(
        list(captions), add_special_tokens=True, truncation=False
    )["input_ids"]

    selected_id_rows: List[List[int]] = []
    selected_query_rows: List[List[bool]] = []
    untruncated_lengths: List[int] = []
    for row, (text, caption, query) in enumerate(
        zip(texts, captions, queries, strict=True)
    ):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids = [int(token_id) for token_id in encoding["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
        untruncated_lengths.append(len(ids))
        caption_start = len(fixed_prefix)
        caption_end = caption_start + len(caption)
        query_start = caption_end + len("\n\n")

        mandatory: List[int] = []
        caption_tokens: List[int] = []
        for index, (start, end) in enumerate(offsets):
            if end <= start:  # special tokens, if the tokenizer inserts any
                mandatory.append(index)
            elif end <= caption_start or start >= caption_end:
                mandatory.append(index)
            elif start >= caption_start and end <= caption_end:
                caption_tokens.append(index)
            else:
                # A rare BPE token crossing a text boundary is structural and
                # must not be silently dropped.
                mandatory.append(index)

        caption_budget = max_length - len(mandatory)
        if caption_budget < 0:
            raise RuntimeError(
                f"matched teacher row {row} fixed prefix/query exceed max_length={max_length}"
            )
        # Preserve the beginning of anime tag captions (identity, hair, eyes,
        # clothing) and discard only their tail. The complete query always
        # remains present.
        selected_indices = sorted(mandatory + caption_tokens[:caption_budget])
        selected_ids = [ids[index] for index in selected_indices]
        selected_query = [
            offsets[index][0] >= query_start and offsets[index][1] > offsets[index][0]
            for index in selected_indices
        ]
        if not any(selected_query):
            raise RuntimeError(f"matched teacher row {row} lost its query suffix")
        selected_id_rows.append(selected_ids)
        selected_query_rows.append(selected_query)

    padded_length = max(len(row) for row in selected_id_rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("matched teacher tokenizer must define pad_token_id")
    input_ids = torch.full(
        (len(selected_id_rows), padded_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (ids, selected_query) in enumerate(
        zip(selected_id_rows, selected_query_rows, strict=True)
    ):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        query_mask[row, :length] = torch.tensor(selected_query, dtype=torch.bool)

    return MatchedTeacherInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        query_mask=query_mask,
        untruncated_lengths=tuple(untruncated_lengths),
        caption_untruncated_lengths=tuple(len(ids) for ids in caption_untruncated),
    )


def teacher_matched_query_summary(
    hidden_states: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_summary(hidden_states, query_mask)


def build_caption_derangement(caption_keys: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
    """Build a deterministic perfect hard-negative matching.

    Every student image is paired with a different row and with a different
    normalized caption.  A bipartite matching is used instead of ``roll(1)`` so
    duplicate captions cannot become false hard negatives.
    """

    keys = [normalize_caption_key(key) for key in caption_keys]
    size = len(keys)
    if size < 2:
        raise ValueError("hard-negative derangement requires batch_size >= 2")

    match_destination_to_source = [-1] * size

    def augment(source: int, seen: List[bool]) -> bool:
        destinations = list(range(source + 1, size)) + list(range(0, source + 1))
        for destination in destinations:
            if destination == source or keys[destination] == keys[source] or seen[destination]:
                continue
            seen[destination] = True
            previous_source = match_destination_to_source[destination]
            if previous_source == -1 or augment(previous_source, seen):
                match_destination_to_source[destination] = source
                return True
        return False

    for source in range(size):
        if not augment(source, [False] * size):
            raise ValueError("batch does not admit a different-caption, no-fixed-point derangement")

    source_to_destination = [-1] * size
    for destination, source in enumerate(match_destination_to_source):
        source_to_destination[source] = destination
    if any(destination < 0 for destination in source_to_destination):
        raise RuntimeError("internal derangement matching failure")
    for source, destination in enumerate(source_to_destination):
        if source == destination or keys[source] == keys[destination]:
            raise RuntimeError("invalid caption derangement")
    return torch.tensor(source_to_destination, dtype=torch.long, device=device)


@dataclass(frozen=True)
class P1AlignmentLossOutput:
    total: torch.Tensor
    normalized_mse: torch.Tensor
    cosine: torch.Tensor
    info_nce: torch.Tensor
    raw_info_nce: torch.Tensor
    hard_negative_margin: torch.Tensor
    correct_cosine: torch.Tensor
    shuffled_cosine: torch.Tensor
    retrieval_top1: torch.Tensor
    raw_retrieval_top1: torch.Tensor
    centered_correct_cosine: torch.Tensor
    centered_shuffled_cosine: torch.Tensor
    correct_over_shuffle_rate: torch.Tensor
    derangement: torch.Tensor

    def detached_metrics(self) -> Dict[str, float]:
        return {
            "loss": float(self.total.detach().item()),
            "loss_normalized_mse": float(self.normalized_mse.detach().item()),
            "loss_cosine": float(self.cosine.detach().item()),
            "loss_info_nce": float(self.info_nce.detach().item()),
            "loss_info_nce_raw": float(self.raw_info_nce.detach().item()),
            "loss_hard_negative_margin": float(self.hard_negative_margin.detach().item()),
            "correct_cosine": float(self.correct_cosine.detach().item()),
            "shuffled_cosine": float(self.shuffled_cosine.detach().item()),
            "correct_vs_shuffle_cosine": float(
                (self.correct_cosine - self.shuffled_cosine).detach().item()
            ),
            "retrieval_top1": float(self.retrieval_top1.detach().item()),
            "retrieval_top1_raw": float(self.raw_retrieval_top1.detach().item()),
            "centered_correct_cosine": float(self.centered_correct_cosine.detach().item()),
            "centered_shuffled_cosine": float(self.centered_shuffled_cosine.detach().item()),
            "centered_correct_vs_shuffle_cosine": float(
                (self.centered_correct_cosine - self.centered_shuffled_cosine).detach().item()
            ),
            "correct_over_shuffle_rate": float(self.correct_over_shuffle_rate.detach().item()),
        }


def compute_p1_alignment_loss(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    caption_keys: Sequence[str],
    *,
    temperature: float = 0.07,
    hard_negative_margin: float = 0.10,
    normalized_mse_weight: float = 0.25,
    cosine_weight: float = 1.0,
    info_nce_weight: float = 1.0,
    hard_negative_weight: float = 1.0,
) -> P1AlignmentLossOutput:
    """Compute manifold, retrieval, and shuffled-image causal losses."""

    if student_summary.shape != teacher_summary.shape or student_summary.ndim != 2:
        raise ValueError("student and teacher summaries must be matching [B,H] tensors")
    batch_size, hidden_size = student_summary.shape
    if batch_size < 2:
        raise ValueError("P1 alignment loss requires batch_size >= 2")
    if len(caption_keys) != batch_size:
        raise ValueError("caption_keys length must match the batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student = F.normalize(student_summary.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher_summary.detach().float(), dim=-1, eps=1e-6)
    correct_per_row = (student * teacher).sum(dim=-1)

    # Scaling by H avoids a vanishing 1/H value for unit vectors.
    normalized_mse = F.mse_loss(student, teacher) * hidden_size
    cosine = (1.0 - correct_per_row).mean()

    raw_logits = student @ teacher.transpose(0, 1) / temperature
    centered_student = F.normalize(student - student.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
    centered_teacher = F.normalize(teacher - teacher.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
    logits = centered_student @ centered_teacher.transpose(0, 1) / temperature
    if raw_logits.dtype != torch.float32 or logits.dtype != torch.float32:
        raise RuntimeError(
            "P1 alignment similarities must be computed in FP32 outside autocast; "
            f"raw={raw_logits.dtype}, centered={logits.dtype}"
        )
    normalized_keys = [normalize_caption_key(key) for key in caption_keys]
    equivalent = torch.tensor(
        [[left == right for right in normalized_keys] for left in normalized_keys],
        dtype=torch.bool,
        device=logits.device,
    )
    eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    invalid_equivalent = equivalent & ~eye
    raw_logits = raw_logits.masked_fill(invalid_equivalent, torch.finfo(raw_logits.dtype).min)
    logits = logits.masked_fill(invalid_equivalent, torch.finfo(logits.dtype).min)
    labels = torch.arange(batch_size, device=logits.device)
    info_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
    raw_info_nce = 0.5 * (
        F.cross_entropy(raw_logits, labels)
        + F.cross_entropy(raw_logits.transpose(0, 1), labels)
    )
    retrieval_top1 = 0.5 * (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.argmax(dim=0) == labels).float().mean()
    )
    raw_retrieval_top1 = 0.5 * (
        (raw_logits.argmax(dim=1) == labels).float().mean()
        + (raw_logits.argmax(dim=0) == labels).float().mean()
    )

    derangement = build_caption_derangement(normalized_keys, device=student.device)
    shuffled_per_row = (student[derangement] * teacher).sum(dim=-1)
    centered_correct_per_row = (centered_student * centered_teacher).sum(dim=-1)
    centered_shuffled_per_row = (centered_student[derangement] * centered_teacher).sum(dim=-1)
    hard_negative = F.relu(
        hard_negative_margin - centered_correct_per_row + centered_shuffled_per_row
    ).mean()

    total = (
        normalized_mse_weight * normalized_mse
        + cosine_weight * cosine
        + info_nce_weight * info_nce
        + hard_negative_weight * hard_negative
    )
    return P1AlignmentLossOutput(
        total=total,
        normalized_mse=normalized_mse,
        cosine=cosine,
        info_nce=info_nce,
        raw_info_nce=raw_info_nce,
        hard_negative_margin=hard_negative,
        correct_cosine=correct_per_row.mean(),
        shuffled_cosine=shuffled_per_row.mean(),
        retrieval_top1=retrieval_top1,
        raw_retrieval_top1=raw_retrieval_top1,
        centered_correct_cosine=centered_correct_per_row.mean(),
        centered_shuffled_cosine=centered_shuffled_per_row.mean(),
        correct_over_shuffle_rate=(correct_per_row > shuffled_per_row).float().mean(),
        derangement=derangement,
    )


@torch.no_grad()
def compute_global_retrieval_metrics(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    caption_keys: Sequence[str],
) -> Dict[str, float]:
    """Compute retrieval and shuffle controls over one complete eval set.

    Unlike per-microbatch diagnostics, these ranks are taken from the full
    NxN similarity matrix.  Both raw and dataset-centered representations are
    reported, and duplicate captions (if a custom manifest has any) are masked
    as false negatives without masking their diagonal positives.
    """

    if student_summary.ndim != 2 or student_summary.shape != teacher_summary.shape:
        raise ValueError("global summaries must be matching [N,H] tensors")
    count = student_summary.shape[0]
    if count < 2 or len(caption_keys) != count:
        raise ValueError("global retrieval requires matching caption keys and N >= 2")

    student = F.normalize(student_summary.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher_summary.float(), dim=-1, eps=1e-6)
    centered_student = F.normalize(student - student.mean(0, keepdim=True), dim=-1, eps=1e-6)
    centered_teacher = F.normalize(teacher - teacher.mean(0, keepdim=True), dim=-1, eps=1e-6)
    raw_similarity = student @ teacher.transpose(0, 1)
    centered_similarity = centered_student @ centered_teacher.transpose(0, 1)

    keys = [normalize_caption_key(key) for key in caption_keys]
    equivalent = torch.tensor(
        [[left == right for right in keys] for left in keys],
        dtype=torch.bool,
        device=raw_similarity.device,
    )
    eye = torch.eye(count, dtype=torch.bool, device=raw_similarity.device)
    invalid = equivalent & ~eye
    floor = torch.finfo(raw_similarity.dtype).min
    raw_rank = raw_similarity.masked_fill(invalid, floor)
    centered_rank = centered_similarity.masked_fill(invalid, floor)
    labels = torch.arange(count, device=student.device)

    def symmetric_recall(similarity: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, count)
        row_hits = (similarity.topk(k, dim=1).indices == labels[:, None]).any(dim=1)
        column_hits = (similarity.topk(k, dim=0).indices == labels[None, :]).any(dim=0)
        return 0.5 * (row_hits.float().mean() + column_hits.float().mean())

    derangement = build_caption_derangement(keys, device=student.device)
    raw_correct = raw_similarity.diag()
    raw_shuffle = (student[derangement] * teacher).sum(-1)
    centered_correct = centered_similarity.diag()
    centered_shuffle = (centered_student[derangement] * centered_teacher).sum(-1)
    return {
        "global_retrieval_top1_raw": float(symmetric_recall(raw_rank, 1).item()),
        "global_retrieval_top5_raw": float(symmetric_recall(raw_rank, 5).item()),
        "global_retrieval_top1_centered": float(symmetric_recall(centered_rank, 1).item()),
        "global_retrieval_top5_centered": float(symmetric_recall(centered_rank, 5).item()),
        "global_correct_cosine_raw": float(raw_correct.mean().item()),
        "global_shuffle_cosine_raw": float(raw_shuffle.mean().item()),
        "global_correct_vs_shuffle_cosine_raw": float(
            (raw_correct.mean() - raw_shuffle.mean()).item()
        ),
        "global_correct_over_shuffle_rate_raw": float(
            (raw_correct > raw_shuffle).float().mean().item()
        ),
        "global_correct_cosine_centered": float(centered_correct.mean().item()),
        "global_shuffle_cosine_centered": float(centered_shuffle.mean().item()),
        "global_correct_vs_shuffle_cosine_centered": float(
            (centered_correct.mean() - centered_shuffle.mean()).item()
        ),
        "global_correct_over_shuffle_rate_centered": float(
            (centered_correct > centered_shuffle).float().mean().item()
        ),
    }


def compute_frozen_llm_adapter_context_loss(
    llm_adapter: nn.Module,
    *,
    student_source_hidden: torch.Tensor,
    student_source_mask: torch.Tensor,
    teacher_source_hidden: torch.Tensor,
    teacher_source_mask: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional native-context distillation with one neutral T5 target.

    The teacher and student receive the same target IDs.  The caption is never
    placed on the T5 side, preventing target leakage.
    """

    trainable = [name for name, parameter in llm_adapter.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"native LLMAdapter must be frozen: {trainable[:8]}")
    llm_adapter.eval()
    with torch.no_grad():
        teacher_context = llm_adapter(
            source_hidden_states=teacher_source_hidden,
            source_attention_mask=teacher_source_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
        )
    student_context = llm_adapter(
        source_hidden_states=student_source_hidden,
        source_attention_mask=student_source_mask,
        target_input_ids=target_input_ids,
        target_attention_mask=target_attention_mask,
    )
    valid = target_attention_mask.bool()
    teacher_valid = F.normalize(teacher_context.float(), dim=-1, eps=1e-6)
    student_valid = F.normalize(student_context.float(), dim=-1, eps=1e-6)
    token_loss = 1.0 - (student_valid * teacher_valid).sum(dim=-1)
    return token_loss.masked_select(valid).mean()


def enforce_connector_only_train_mode(
    conditioner: AnimaVLMConditioner,
    *,
    connector_dtype: torch.dtype = torch.float32,
) -> List[nn.Parameter]:
    """Set train/eval modes and assert the exact 12.58M connector contract."""

    conditioner.configure_visual_alignment(train_connector=True, vision_last_n=0)
    # Keep FP32 master weights and AdamW states. CUDA autocast performs the
    # connector matmul in BF16 without degrading optimizer precision.
    conditioner.vlm_backbone.connector.to(dtype=connector_dtype)
    conditioner.train()
    conditioner.vlm_backbone.connector.train()
    conditioner.vlm_backbone.vision_model.eval()
    conditioner.text_model.eval()
    conditioner.assert_qwen_text_body_frozen()

    trainable_named = [(name, parameter) for name, parameter in conditioner.named_parameters() if parameter.requires_grad]
    invalid = [name for name, _ in trainable_named if not name.startswith("vlm_backbone.connector.")]
    if invalid:
        raise RuntimeError(f"P1 has non-connector trainable parameters: {invalid[:8]}")
    count = sum(parameter.numel() for _, parameter in trainable_named)
    if count != EXPECTED_CONNECTOR_PARAMETERS:
        raise RuntimeError(
            f"P1 connector trainable parameter mismatch: expected={EXPECTED_CONNECTOR_PARAMETERS}, actual={count}, "
            f"keys={[name for name, _ in trainable_named]}"
        )
    invalid_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in trainable_named
        if parameter.dtype != connector_dtype
    ]
    if invalid_dtypes:
        raise RuntimeError(
            f"P1 connector parameters must use {connector_dtype} master weights: {invalid_dtypes[:8]}"
        )
    if conditioner.vlm_backbone.vision_model.training:
        raise RuntimeError("frozen vision tower must stay in eval mode")
    if conditioner.text_model.training:
        raise RuntimeError("frozen clean Qwen must stay in eval mode")
    return [parameter for _, parameter in trainable_named]


def assert_optimizer_connector_only(
    optimizer: torch.optim.Optimizer,
    conditioner: AnimaVLMConditioner,
) -> None:
    connector_ids = {
        id(parameter)
        for parameter in conditioner.vlm_backbone.connector.parameters()
        if parameter.requires_grad
    }
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != connector_ids:
        raise RuntimeError(
            f"optimizer parameter contract mismatch: connector={len(connector_ids)}, optimizer={len(optimizer_ids)}"
        )


def assert_adamw_fp32_state(optimizer: torch.optim.Optimizer) -> None:
    """Assert FP32 connector masters and FP32 floating AdamW moments."""

    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            if parameter.dtype != torch.float32:
                raise RuntimeError(
                    "AdamW master parameter is not FP32: "
                    f"group={group_index}, parameter={parameter_index}, dtype={parameter.dtype}"
                )
            for state_name, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                    raise RuntimeError(
                        "AdamW floating state is not FP32: "
                        f"group={group_index}, parameter={parameter_index}, "
                        f"state={state_name}, dtype={value.dtype}"
                    )


@torch.no_grad()
def frozen_module_tensor_checksum(module: nn.Module) -> str:
    """Fingerprint every frozen tensor via deterministic full reductions.

    This is intentionally stronger than checking ``requires_grad``: every
    tensor contributes its name, metadata, sum, squared sum, extrema, and edge
    values.  The checksum is recorded at startup, validation, and checkpoint.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach()
        flat = value.reshape(-1)
        stats = torch.stack(
            [
                flat.float().sum(),
                flat.float().square().sum(),
                flat.float().min(),
                flat.float().max(),
                flat.float()[0],
                flat.float()[-1],
            ]
        ).cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(stats.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_native_llm_adapter(
    dit_path: Union[str, os.PathLike[str]],
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> nn.Module:
    """Strict-load only the 118 official native LLMAdapter tensors."""

    from safetensors import safe_open
    from _anima_native_ref_vendor.library.anima_models import LLMAdapter

    adapter = LLMAdapter(
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        self_attn=True,
    )
    prefix = "net.llm_adapter."
    with safe_open(dit_path, framework="pt", device="cpu") as handle:
        state = {
            key[len(prefix) :]: handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(prefix)
        }
    if len(state) != 118:
        raise RuntimeError(f"expected 118 native LLMAdapter tensors, found {len(state)}")
    adapter.load_state_dict(state, strict=True)
    holder = type("AnimaAdapterHolder", (), {"llm_adapter": adapter})()
    freeze_and_assert_anima_llm_adapter(holder)
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    return adapter


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_ids(tokenizer: Any, texts: Sequence[str]) -> List[List[int]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if len(texts) == 1 and encoded and isinstance(encoded[0], int):
        encoded = [encoded]
    return [[int(token_id) for token_id in row] for row in encoded]


def verify_tokenizer_id_parity(
    source_tokenizer: Any,
    package_tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Prove that saving/reloading did not change a single tokenizer ID.

    The rolling digest is over length-delimited ID sequences rather than text
    bytes.  This makes the package metadata a compact, reproducible record of
    exactly what was compared without retaining any dataset caption content.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        raise ValueError("tokenizer parity corpus must not be empty")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("tokenizer parity corpus must contain only strings")

    digest = hashlib.sha256()
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        source_rows = _tokenizer_ids(source_tokenizer, batch)
        package_rows = _tokenizer_ids(package_tokenizer, batch)
        if len(source_rows) != len(batch) or len(package_rows) != len(batch):
            raise RuntimeError("tokenizer returned an unexpected batch dimension")
        for offset, (source_ids, package_ids) in enumerate(zip(source_rows, package_rows, strict=True)):
            if source_ids != package_ids:
                index = start + offset
                text_sha256 = hashlib.sha256(batch[offset].encode("utf-8")).hexdigest()
                mismatch = next(
                    (
                        position
                        for position, (left, right) in enumerate(
                            zip(source_ids, package_ids, strict=False)
                        )
                        if left != right
                    ),
                    min(len(source_ids), len(package_ids)),
                )
                raise RuntimeError(
                    "saved tokenizer ID parity failure: "
                    f"text_index={index}, text_sha256={text_sha256}, first_mismatch={mismatch}, "
                    f"source_length={len(source_ids)}, package_length={len(package_ids)}"
                )
            digest.update(len(source_ids).to_bytes(8, "big", signed=False))
            for token_id in source_ids:
                digest.update(token_id.to_bytes(4, "big", signed=False))
            total_tokens += len(source_ids)

    return {
        "verified": True,
        "text_count": len(texts),
        "token_count": total_tokens,
        "input_ids_sha256": digest.hexdigest(),
    }


def _canonical_tokenizer_parity_texts() -> List[str]:
    texts = [canonical_image_block()]
    for query in (*P1_QUERY_BANK, *P1_EVAL_QUERY_BANK):
        one_image = build_role_prompt(query, num_identity_images=1, has_scene_image=False)
        texts.append(expand_canonical_image_placeholders(one_image, expected_images=1))
        texts.append(f"Visual description:\n1girl, blue hair\n\n{query}")
    multi_role = build_role_prompt(
        P1_VISUAL_QUERY,
        num_identity_images=2,
        has_scene_image=True,
    )
    texts.append(expand_canonical_image_placeholders(multi_role, expected_images=3))
    return texts


def _file_record(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def package_metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Bind optimizer state to the exact intrinsic VLM/support file records."""

    payload = {
        "format": metadata.get("format"),
        "step": metadata.get("step"),
        "model_files": metadata.get("model_files"),
        "support_files": metadata.get("support_files"),
        "tokenizer_parity": metadata.get("tokenizer_parity"),
    }
    if payload["format"] != P1_PACKAGE_FORMAT or not payload["model_files"]:
        raise ValueError("cannot fingerprint invalid P1 package metadata")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def save_full_vlm_package(
    conditioner: AnimaVLMConditioner,
    processor: Any,
    output_dir: Union[str, os.PathLike[str]],
    *,
    step: int,
    source_vlm_revision: str,
    clean_qwen_sha256: str,
    manifest_captions: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save vision+native connector+clean Qwen as one self-contained package."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioner.assert_qwen_text_body_frozen()
    trainable = [name for name, parameter in conditioner.named_parameters() if parameter.requires_grad]
    if any(not name.startswith("vlm_backbone.connector.") for name in trainable):
        raise RuntimeError(f"cannot package unexpected trainable parameters: {trainable[:8]}")

    conditioner.vlm_backbone.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.tokenizer.save_pretrained(output_dir)
    processor.image_processor.save_pretrained(output_dir)
    (output_dir / P1_PROCESSOR_FILENAME).write_text(
        json.dumps(
            {
                "image_seq_len": IMAGE_SEQ_LEN,
                "processor_class": "AnimaCanonicalVLMProcessor",
                "role_schema_version": ROLE_SCHEMA_VERSION,
                "do_image_splitting": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Reload the package tokenizer through its actual public load path.  The
    # explicit False is part of the Anima tokenizer contract; see the matching
    # full-package loader comment in library/anima_vlm.py.
    from transformers import AutoTokenizer

    package_tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    manifest_captions = list(manifest_captions or ())
    canonical_parity_texts = _canonical_tokenizer_parity_texts()
    tokenizer_parity = verify_tokenizer_id_parity(
        processor.tokenizer,
        package_tokenizer,
        [*manifest_captions, *canonical_parity_texts],
    )
    tokenizer_parity.update(
        {
            "manifest_caption_count": len(manifest_captions),
            "canonical_text_count": len(canonical_parity_texts),
            "fix_mistral_regex": False,
        }
    )

    model_files = sorted(output_dir.glob("model*.safetensors"))
    if not model_files:
        raise RuntimeError("save_pretrained produced no model safetensors")
    support_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name != P1_PACKAGE_FILENAME
        and path not in model_files
    )
    support_names = {path.name for path in support_files}
    required_support_names = {
        "config.json",
        "preprocessor_config.json",
        P1_PROCESSOR_FILENAME,
        "tokenizer_config.json",
        "tokenizer.json",
    }
    if not required_support_names.issubset(support_names):
        raise RuntimeError(
            "self-contained VLM package is missing required tokenizer/config/processor files: "
            f"{sorted(required_support_names - support_names)}"
        )
    metadata: Dict[str, Any] = {
        "format": P1_PACKAGE_FORMAT,
        "step": int(step),
        "model_class": "SmolVLMModel",
        "includes_vision_model": True,
        "includes_native_connector": True,
        "includes_clean_anima_qwen": True,
        "runtime_external_adapter_required": False,
        "source_vlm_revision": source_vlm_revision,
        "clean_qwen_sha256": clean_qwen_sha256,
        "clean_qwen_key_count": 310,
        "connector_parameter_count": EXPECTED_CONNECTOR_PARAMETERS,
        "connector_dtype": str(next(conditioner.vlm_backbone.connector.parameters()).dtype),
        "role_schema_version": ROLE_SCHEMA_VERSION,
        "visual_query": P1_VISUAL_QUERY,
        "tokenizer_parity": tokenizer_parity,
        "model_files": [_file_record(path) for path in model_files],
        "support_files": [_file_record(path) for path in support_files],
    }
    if extra_metadata:
        metadata["extra"] = dict(extra_metadata)
    metadata["package_content_fingerprint"] = package_metadata_fingerprint(metadata)
    (output_dir / P1_PACKAGE_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def save_resume_state(
    path: Union[str, os.PathLike[str]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer_step: int,
    micro_step: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save optimizer/RNG state separately; it is not a runtime dependency."""

    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_resume_state(
    path: Union[str, os.PathLike[str]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        restore_rng_state(state)
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNGs after all resume-time module constructors have run."""

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def warmup_cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    # LambdaLR evaluates step=0 before the first optimizer update.  Dividing by
    # (total-warmup), rather than (total-warmup-1), leaves the final *used* LR
    # positive and reaches exactly zero only after the final scheduler.step().
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
