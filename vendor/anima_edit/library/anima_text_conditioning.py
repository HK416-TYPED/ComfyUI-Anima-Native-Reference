"""Shared Anima sampling text-conditioning contract.

The standard Anima text path consumes four tensors:

* raw Qwen hidden states;
* the Qwen attention mask;
* neutral-T5 token IDs; and
* the neutral-T5 attention mask.

V4 competitive reference routing additionally consumes clause masks aligned to
the *raw* Qwen sequence.  In classifier-free guidance (CFG), the negative
branch must retain its own standard text tensors while reusing the positive
branch's router-only tensors.  Keeping that split in one small module prevents
the training sampler and the standalone inference script from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch


PromptBatch = Union[str, Sequence[str]]


def _normalise_prompts(prompt: PromptBatch) -> tuple[str, ...]:
    prompts = (prompt,) if isinstance(prompt, str) else tuple(str(item) for item in prompt)
    if not prompts:
        raise ValueError("At least one prompt is required.")
    return prompts


def _as_batched_tensor(
    value: Any,
    *,
    name: str,
    batched_ndim: int,
    batch_size: int,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required for the standard Anima text path.")
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == batched_ndim - 1 and batch_size == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != batched_ndim:
        raise ValueError(
            f"{name} must have {batched_ndim} dimensions (or be an unbatched "
            f"single-sample tensor), got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size {tensor.shape[0]} does not match prompt batch size {batch_size}."
        )
    return tensor


def normalise_reference_slot_ids(
    reference_slot_ids: Optional[Sequence[Sequence[int]]],
    *,
    batch_size: int,
    max_slots: int = 2,
) -> Optional[tuple[tuple[int, ...], ...]]:
    """Validate logical reference slots for a homogeneous sampling batch.

    Sampling in both current callers uses a batch size of one, but accepting a
    batch here makes the tensor contract explicit and independently testable.
    A batch may contain no references, or every row must contain references;
    mixing ref and no-ref rows would make the model's fixed-one/two carrier
    ambiguous and is rejected.
    """

    if reference_slot_ids is None:
        return None
    if torch.is_tensor(reference_slot_ids):
        reference_slot_ids = reference_slot_ids.detach().cpu().tolist()

    rows = list(reference_slot_ids)
    if batch_size == 1 and rows and all(not isinstance(item, (list, tuple)) for item in rows):
        rows = [rows]
    if len(rows) != batch_size:
        raise ValueError(
            "reference_slot_ids must contain one logical-slot row per prompt: "
            f"{len(rows)} != {batch_size}."
        )

    normalised: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        if torch.is_tensor(row):
            row = row.detach().cpu().tolist()
        slots = tuple(int(slot_id) for slot_id in (row or ()))
        if len(set(slots)) != len(slots):
            raise ValueError(f"reference_slot_ids row {row_index} contains duplicate slots: {slots}.")
        invalid = [slot_id for slot_id in slots if slot_id < 0 or slot_id >= max_slots]
        if invalid:
            raise ValueError(
                f"reference_slot_ids row {row_index} contains slots outside [0,{max_slots}): {invalid}."
            )
        normalised.append(slots)

    occupied = [bool(row) for row in normalised]
    if not any(occupied):
        return None
    if not all(occupied):
        raise ValueError("A sampling batch cannot mix reference and no-reference rows.")
    return tuple(normalised)


def reference_router_requires_clause_masks(
    model: Any,
    reference_slot_ids: Optional[Sequence[Sequence[int]]],
    *,
    use_reference_sequence: bool = True,
) -> bool:
    """Return whether an active reference sample must carry V4 clause masks."""

    if not use_reference_sequence or reference_slot_ids is None:
        return False
    routing_alpha = float(getattr(model, "native_reference_routing_alpha", 0.0))
    if routing_alpha <= 0.0:
        return False
    routing_mode = getattr(model, "native_reference_routing_mode", "legacy")
    if routing_mode != "competitive_text_slot_v1":
        raise RuntimeError(
            "A positive native-reference routing alpha requires "
            "competitive_text_slot_v1 mode."
        )
    return True


@dataclass(frozen=True)
class AnimaPromptConditioning:
    """Raw standard-text tensors plus optional positive reference clauses."""

    prompts: tuple[str, ...]
    raw_qwen_context: torch.Tensor
    source_attention_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    reference_clause_masks: Optional[torch.Tensor] = None

    def model_text_kwargs(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        router_conditioning: Optional["AnimaPromptConditioning"] = None,
        use_llm_adapter: bool = True,
    ) -> dict[str, Any]:
        """Build one :class:`Anima` forward call's text keyword arguments.

        ``self`` always supplies the standard text path.  When
        ``router_conditioning`` is supplied (the CFG negative branch), only the
        raw Qwen router context, its attention mask, and positive clause masks
        are taken from that object.
        """

        router = self if router_conditioning is None else router_conditioning
        kwargs: dict[str, Any] = {
            "context": self.raw_qwen_context.to(device=device, dtype=dtype),
            "source_attention_mask": self.source_attention_mask.to(device=device),
            "raw_qwen_context": router.raw_qwen_context.to(device=device, dtype=dtype),
            "router_source_attention_mask": router.source_attention_mask.to(device=device),
            "reference_clause_masks": (
                None
                if router.reference_clause_masks is None
                else router.reference_clause_masks.to(device=device, dtype=torch.bool)
            ),
        }
        if use_llm_adapter:
            kwargs["target_input_ids"] = self.target_input_ids.to(device=device, dtype=torch.long)
            kwargs["target_attention_mask"] = self.target_attention_mask.to(device=device)
        else:
            # Anima.forward treats a non-None target_input_ids value as a request
            # to run the native LLM adapter.  Legacy adapter-free models must
            # therefore receive explicit None values rather than stale IDs.
            kwargs["target_input_ids"] = None
            kwargs["target_attention_mask"] = None
        return kwargs


def prepare_anima_prompt_conditioning(
    prompt: PromptBatch,
    encoded: Sequence[Any],
    *,
    tokenize_strategy: Any,
    device: torch.device,
    dtype: torch.dtype,
    reference_slot_ids: Optional[Sequence[Sequence[int]]] = None,
    require_reference_binding: bool = False,
    max_slots: int = 2,
) -> AnimaPromptConditioning:
    """Validate and move raw Anima text outputs without applying the LLM adapter.

    Clause masks are built only when ``require_reference_binding`` is true.
    This preserves legacy alpha-zero and ordinary T2I behaviour while making
    the positive-alpha competitive route fail closed on ambiguous,
    non-canonical, or truncated reference instructions.
    """

    prompts = _normalise_prompts(prompt)
    if len(encoded) != 4:
        raise ValueError(
            "Anima text encoding must contain exactly "
            "[Qwen hidden, Qwen mask, T5 IDs, T5 mask]."
        )
    batch_size = len(prompts)
    raw_qwen = _as_batched_tensor(
        encoded[0], name="raw_qwen_context", batched_ndim=3, batch_size=batch_size
    )
    source_mask = _as_batched_tensor(
        encoded[1], name="source_attention_mask", batched_ndim=2, batch_size=batch_size
    )
    target_ids = _as_batched_tensor(
        encoded[2], name="target_input_ids", batched_ndim=2, batch_size=batch_size
    )
    target_mask = _as_batched_tensor(
        encoded[3], name="target_attention_mask", batched_ndim=2, batch_size=batch_size
    )

    if tuple(source_mask.shape) != tuple(raw_qwen.shape[:2]):
        raise ValueError(
            "source_attention_mask must match the raw Qwen batch/token shape: "
            f"{tuple(source_mask.shape)} != {tuple(raw_qwen.shape[:2])}."
        )
    if tuple(target_mask.shape) != tuple(target_ids.shape):
        raise ValueError(
            "target_attention_mask must match target_input_ids: "
            f"{tuple(target_mask.shape)} != {tuple(target_ids.shape)}."
        )

    slots = normalise_reference_slot_ids(
        reference_slot_ids,
        batch_size=batch_size,
        max_slots=max_slots,
    )
    clause_masks = None
    if require_reference_binding:
        if slots is None:
            raise ValueError("Competitive reference routing requires non-empty reference_slot_ids.")
        if not hasattr(tokenize_strategy, "tokenize_reference_clause_masks"):
            raise TypeError(
                "Competitive reference routing requires AnimaTokenizeStrategy "
                "with tokenize_reference_clause_masks()."
            )
        mask_batch = tokenize_strategy.tokenize_reference_clause_masks(
            list(prompts),
            slots,
            max_slots=max_slots,
            require_canonical=True,
        )
        validity = torch.as_tensor(mask_batch.binding_valid, dtype=torch.bool)
        if validity.shape != (batch_size,) or not bool(validity.all()):
            failures = []
            statuses = tuple(getattr(mask_batch, "statuses", ()))
            reasons = tuple(getattr(mask_batch, "reasons", ()))
            canonical = tuple(getattr(mask_batch, "canonical_instructions", ()))
            for index in range(batch_size):
                if index >= validity.numel() or not bool(validity[index]):
                    failures.append(
                        {
                            "row": index,
                            "status": statuses[index] if index < len(statuses) else "unknown",
                            "reason": reasons[index] if index < len(reasons) else "unknown",
                            "canonical": canonical[index] if index < len(canonical) else None,
                        }
                    )
            raise ValueError(
                "Competitive reference sampling requires byte-exact canonical "
                f"SAFE reference bindings; invalid rows: {failures}."
            )
        clause_masks = torch.as_tensor(mask_batch.clause_masks, dtype=torch.bool)
        expected_shape = (batch_size, max_slots, raw_qwen.shape[1])
        if tuple(clause_masks.shape) != expected_shape:
            raise ValueError(
                "reference_clause_masks must be bool [B,S,Lqwen] and align with "
                f"the encoded Qwen sequence: {tuple(clause_masks.shape)} != {expected_shape}."
            )
        active_tokens = source_mask.to(dtype=torch.bool, device=clause_masks.device)
        if bool((clause_masks & ~active_tokens.unsqueeze(1)).any()):
            raise ValueError("reference_clause_masks select padded Qwen tokens.")
        for row_index, row_slots in enumerate(slots):
            for slot_id in range(max_slots):
                occupied = bool(clause_masks[row_index, slot_id].any())
                if occupied != (slot_id in row_slots):
                    raise ValueError(
                        "reference clause occupancy does not match logical slots: "
                        f"row={row_index}, slot={slot_id}, occupied={occupied}, "
                        f"expected_slots={row_slots}."
                    )

    return AnimaPromptConditioning(
        prompts=prompts,
        raw_qwen_context=raw_qwen.to(device=device, dtype=dtype),
        source_attention_mask=source_mask.to(device=device),
        target_input_ids=target_ids.to(device=device, dtype=torch.long),
        target_attention_mask=target_mask.to(device=device),
        reference_clause_masks=(
            None if clause_masks is None else clause_masks.to(device=device, dtype=torch.bool)
        ),
    )
