"""Shared primitives for Anima's unified, maskless visual conditioning.

Editing is represented as a property of one ordinary reference slot rather
than as a second visual branch.  The functions in this module deliberately do
not import :mod:`library.anima_models`: they validate the aligned source slot,
reduce *training-only* spatial labels, and combine global and same-position
source reads using a change probability predicted inside the checkpoint.

The spatial convention is zero = preserve source and one = change.  A spatial
label produced from source/target pairs must never be passed to the denoiser or
sampler; only the model's own predicted probabilities enter the blend helper.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence, TypeAlias

import torch


ReferenceSlotIds: TypeAlias = Sequence[Sequence[int]] | torch.Tensor
AlignedRows: TypeAlias = bool | torch.Tensor
PatchSize: TypeAlias = int | tuple[int, int]

__all__ = [
    "resolve_aligned_source_physical_indices",
    "raster_change_label_to_latent",
    "spatial_change_label_to_patch_tokens",
    "blend_predicted_source_scope",
]


def _require_integer_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if value.dtype is torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
        raise TypeError(f"{name} must use an integer dtype")


def _normalise_reference_slot_rows(reference_slot_ids: ReferenceSlotIds) -> list[list[int]]:
    if torch.is_tensor(reference_slot_ids):
        _require_integer_tensor("reference_slot_ids", reference_slot_ids, 2)
        if reference_slot_ids.shape[0] <= 0 or reference_slot_ids.shape[1] <= 0:
            raise ValueError("reference_slot_ids must have non-empty batch and slot axes")
        rows = reference_slot_ids.detach().cpu().tolist()
    else:
        if isinstance(reference_slot_ids, (str, bytes)) or not isinstance(
            reference_slot_ids, Sequence
        ):
            raise TypeError(
                "reference_slot_ids must be an integer [B,S] tensor or a sequence of rows"
            )
        if len(reference_slot_ids) == 0:
            raise ValueError("reference_slot_ids must contain at least one batch row")
        rows = []
        for batch_index, raw_row in enumerate(reference_slot_ids):
            if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
                raise TypeError(
                    f"reference_slot_ids[{batch_index}] must be a sequence of integers"
                )
            row: list[int] = []
            for physical_index, raw_slot in enumerate(raw_row):
                if isinstance(raw_slot, bool) or not isinstance(raw_slot, Integral):
                    raise TypeError(
                        "reference slot IDs must be integers; "
                        f"row {batch_index}, physical index {physical_index} is {raw_slot!r}"
                    )
                row.append(int(raw_slot))
            rows.append(row)

    for batch_index, row in enumerate(rows):
        if not row:
            raise ValueError(
                f"reference_slot_ids[{batch_index}] must contain at least one physical reference"
            )
        if any(slot_id < 0 for slot_id in row):
            raise ValueError(f"reference_slot_ids[{batch_index}] contains a negative slot ID")
        if len(set(row)) != len(row):
            raise ValueError(
                f"reference_slot_ids[{batch_index}] contains duplicate logical slot IDs"
            )
    return rows


def resolve_aligned_source_physical_indices(
    reference_slot_ids: ReferenceSlotIds,
    aligned_source_slot: torch.Tensor,
) -> torch.Tensor:
    """Resolve one logical aligned-source slot to its physical row index.

    ``reference_slot_ids[b][p]`` maps physical reference ``p`` to the logical
    ``Image N`` slot used by the prompt.  ``aligned_source_slot[b]`` is that
    logical ID.  Every row must contain it exactly once; missing and duplicate
    slot layouts fail rather than silently selecting the first image.

    The returned ``int64 [B]`` tensor stays on ``aligned_source_slot.device``.
    """

    rows = _normalise_reference_slot_rows(reference_slot_ids)
    _require_integer_tensor("aligned_source_slot", aligned_source_slot, 1)
    if aligned_source_slot.shape[0] != len(rows):
        raise ValueError(
            "aligned_source_slot batch must match reference_slot_ids: "
            f"{aligned_source_slot.shape[0]} != {len(rows)}"
        )
    if aligned_source_slot.shape[0] <= 0:
        raise ValueError("aligned_source_slot cannot be empty")

    logical_slots = aligned_source_slot.detach().cpu().tolist()
    physical_indices: list[int] = []
    for batch_index, (row, raw_slot) in enumerate(zip(rows, logical_slots, strict=True)):
        slot_id = int(raw_slot)
        if slot_id < 0:
            raise ValueError(
                f"aligned_source_slot[{batch_index}] must be non-negative, got {slot_id}"
            )
        matches = [index for index, candidate in enumerate(row) if candidate == slot_id]
        if len(matches) != 1:
            raise ValueError(
                f"aligned source logical slot {slot_id} must occur exactly once in "
                f"reference_slot_ids[{batch_index}], found {len(matches)}"
            )
        physical_indices.append(matches[0])

    return torch.tensor(
        physical_indices,
        dtype=torch.long,
        device=aligned_source_slot.device,
    )


def _normalise_patch_size(patch_size: PatchSize) -> tuple[int, int]:
    if isinstance(patch_size, bool):
        raise TypeError("patch_size must be an integer or a pair of integers")
    if isinstance(patch_size, Integral):
        height = width = int(patch_size)
    elif isinstance(patch_size, tuple) and len(patch_size) == 2:
        raw_height, raw_width = patch_size
        if (
            isinstance(raw_height, bool)
            or isinstance(raw_width, bool)
            or not isinstance(raw_height, Integral)
            or not isinstance(raw_width, Integral)
        ):
            raise TypeError("patch_size entries must be integers")
        height, width = int(raw_height), int(raw_width)
    else:
        raise TypeError("patch_size must be an integer or a (height, width) tuple")
    if height <= 0 or width <= 0:
        raise ValueError("patch_size entries must be positive")
    return height, width


def _validate_unit_interval_tensor(
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, ...] | None = None,
    name: str = "value",
) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype")
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(((value < 0) | (value > 1)).any().item()):
        raise ValueError(f"{name} values must lie in [0,1]")


def spatial_change_label_to_patch_tokens(
    change_label: torch.Tensor,
    patch_size: PatchSize,
) -> torch.Tensor:
    """Reduce a trainer-only spatial label to row-major patch tokens.

    Downsampling is an exact non-overlapping patch mean.  No interpolation,
    padding, threshold, dilation, or implicit spatial resize is performed.
    Consequently ``H`` and ``W`` must be divisible by the requested patch
    dimensions.  Only the image contract ``T=1`` is accepted.
    """

    if not torch.is_tensor(change_label):
        raise TypeError("change_label must be a torch.Tensor")
    if change_label.ndim != 5:
        raise ValueError("change_label must be [B,1,1,H,W]")
    batch, channels, temporal, height, width = change_label.shape
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("change_label batch and spatial dimensions must be positive")
    if channels != 1 or temporal != 1:
        raise ValueError("change_label must have exactly one channel and T=1")
    _validate_unit_interval_tensor(change_label, name="change_label")

    patch_height, patch_width = _normalise_patch_size(patch_size)
    if height % patch_height or width % patch_width:
        raise ValueError(
            "change_label spatial dimensions must be exactly divisible by patch_size: "
            f"label=({height},{width}), patch=({patch_height},{patch_width})"
        )

    patch_rows = height // patch_height
    patch_columns = width // patch_width
    spatial = change_label[:, 0, 0].contiguous().reshape(
        batch,
        patch_rows,
        patch_height,
        patch_columns,
        patch_width,
    )
    pooled = spatial.mean(dim=(2, 4))
    return pooled.reshape(batch, patch_rows * patch_columns, 1)


def raster_change_label_to_latent(
    change_label: torch.Tensor,
    latent_hw: tuple[int, int],
) -> torch.Tensor:
    """Reduce a trainer-only raster change label to the VAE grid.

    The input and output contracts are ``[B,1,1,H,W]``.  Reduction uses exact
    non-overlapping patch means, preserving soft mask coverage in ``[0,1]``.
    There is no interpolation, adaptive pooling, threshold, padding, or
    morphology.  Consequently the raster dimensions must be integer multiples
    of the requested latent dimensions.
    """

    if (
        not isinstance(latent_hw, tuple)
        or len(latent_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in latent_hw)
    ):
        raise TypeError("latent_hw must be a (height, width) tuple of integers")
    latent_height, latent_width = (int(value) for value in latent_hw)
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent_hw entries must be positive")
    if not torch.is_tensor(change_label) or change_label.ndim != 5:
        raise ValueError("change_label must be [B,1,1,H,W]")
    raster_height, raster_width = change_label.shape[-2:]
    if raster_height % latent_height or raster_width % latent_width:
        raise ValueError(
            "Raster change-label dimensions must be exact integer multiples of latent_hw: "
            f"label=({raster_height},{raster_width}), latent=({latent_height},{latent_width})"
        )

    tokens = spatial_change_label_to_patch_tokens(
        change_label,
        (raster_height // latent_height, raster_width // latent_width),
    )
    return tokens.reshape(change_label.shape[0], 1, 1, latent_height, latent_width)


def _normalise_aligned_rows(
    aligned: AlignedRows,
    *,
    batch: int,
    device: torch.device,
    require_any: bool = True,
) -> torch.Tensor | None:
    if isinstance(aligned, bool):
        if not aligned:
            return None
        return torch.ones(batch, dtype=torch.bool, device=device)
    if not torch.is_tensor(aligned):
        raise TypeError("aligned must be a bool or a bool [B] tensor")
    if aligned.dtype is not torch.bool or aligned.ndim != 1 or aligned.shape[0] != batch:
        raise ValueError(f"aligned must be bool [B]={batch}")
    if aligned.device != device:
        raise ValueError("aligned must already be on the attention tensor device")
    if require_any and not bool(aligned.any().item()):
        return None
    return aligned


def blend_predicted_source_scope(
    global_attention: torch.Tensor,
    same_position_value: torch.Tensor,
    predicted_change_tokens: torch.Tensor,
    aligned: AlignedRows,
    *,
    _trusted_internal: bool = False,
) -> torch.Tensor:
    """Blend a source read using model-predicted change probabilities.

    For an aligned row the deterministic rule is

    ``(1 - p_change) * same_position_value + p_change * global_attention``.

    A non-aligned row is selected directly from ``global_attention``.  If no
    row is aligned, this function returns the original tensor object before it
    inspects the diagonal value or probability, providing a true hard bypass for
    ordinary reference generation.
    """

    if not torch.is_tensor(global_attention):
        raise TypeError("global_attention must be a torch.Tensor")
    if global_attention.ndim < 3:
        raise ValueError("global_attention must be [B,N,...] with at least one feature axis")
    if not global_attention.dtype.is_floating_point:
        raise TypeError("global_attention must use a floating-point dtype")
    batch, sequence_length = global_attention.shape[:2]
    if batch <= 0 or sequence_length <= 0:
        raise ValueError("global_attention must have non-empty batch and token axes")

    aligned_rows = _normalise_aligned_rows(
        aligned,
        batch=batch,
        device=global_attention.device,
        require_any=not _trusted_internal,
    )
    if aligned_rows is None:
        return global_attention

    if not torch.is_tensor(same_position_value):
        raise TypeError("same_position_value must be a torch.Tensor")
    if tuple(same_position_value.shape) != tuple(global_attention.shape):
        raise ValueError("same_position_value must have the same shape as global_attention")
    if same_position_value.dtype != global_attention.dtype:
        raise TypeError("same_position_value must have the same dtype as global_attention")
    if same_position_value.device != global_attention.device:
        raise ValueError("same_position_value must be on the global_attention device")
    if _trusted_internal:
        # Production callers pass sigmoid/ramp probabilities produced inside
        # the model and have already established that an aligned source exists.
        # Keep structural fail-closed checks, but avoid three CUDA -> host
        # synchronizations per DiT block (finite/range/aligned-any). Public
        # callers retain the full numerical validation below.
        if not torch.is_tensor(predicted_change_tokens):
            raise TypeError("predicted_change_tokens must be a torch.Tensor")
        if not predicted_change_tokens.dtype.is_floating_point:
            raise TypeError("predicted_change_tokens must use a floating-point dtype")
        if tuple(predicted_change_tokens.shape) != (batch, sequence_length, 1):
            raise ValueError(
                "predicted_change_tokens must have shape "
                f"{(batch, sequence_length, 1)}, got {tuple(predicted_change_tokens.shape)}"
            )
    else:
        _validate_unit_interval_tensor(
            predicted_change_tokens,
            expected_shape=(batch, sequence_length, 1),
            name="predicted_change_tokens",
        )
    if predicted_change_tokens.device != global_attention.device:
        raise ValueError("predicted_change_tokens must be on the global_attention device")
    mask = predicted_change_tokens.to(dtype=global_attention.dtype).reshape(
        batch,
        sequence_length,
        *([1] * (global_attention.ndim - 2)),
    )
    blended = (1.0 - mask) * same_position_value + mask * global_attention
    # Arithmetic must not weaken the two semantic endpoints.
    blended = torch.where(
        mask == 0,
        same_position_value,
        torch.where(mask == 1, global_attention, blended),
    )
    aligned_view = aligned_rows.reshape(
        batch,
        *([1] * (global_attention.ndim - 1)),
    )
    return torch.where(aligned_view, blended, global_attention)
