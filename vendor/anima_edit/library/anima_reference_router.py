"""Slot-agnostic building blocks for competitive reference routing.

The classes in this file deliberately do *not* assign semantic roles to
physical reference positions.  A slot is only an axis in the input tensors;
all learned projections are shared over that axis.  Consequently, permuting
``candidates``, clause contexts, and ``slot_valid`` together permutes the slot
probabilities and leaves the merged result unchanged.

The module is intentionally independent from :mod:`library.anima_models` so it
can be exercised before it is wired into a particular attention backend.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _check_bool_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}.")


class SlotClausePooler(nn.Module):
    """Pool instruction tokens into one or more contexts per logical slot.

    Args:
        hidden_dim: Width ``D`` of the raw text hidden states.
        context_dim: Width ``R`` returned to the router.
        num_context_tokens: Number ``K`` of shared learned pooling queries.
        generic_fallback: Default policy for a slot whose clause mask is empty.
            ``False`` (the fail-closed default) returns a zero context.  ``True``
            pools from ``generic_token_mask`` (or ``token_valid`` when no
            separate generic mask is supplied).  The policy may be overridden
            explicitly on every :meth:`forward` call.

    Inputs:
        ``raw_hidden`` is ``[B, L, D]`` and ``clause_masks`` is a boolean
        ``[B, S, L]`` tensor.  The result is ``[B, S, K, R]`` plus a boolean
        ``[B, S]`` mask indicating that the *original supervised clause* was
        missing.  The missing flag remains true when generic fallback is used;
        this makes fallback auditable instead of silently manufacturing a
        supervised clause.

    There are no parameters indexed by ``S``.  Pooling queries distinguish
    context-token positions, not reference slots.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_context_tokens: int = 1,
        *,
        generic_fallback: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if num_context_tokens <= 0:
            raise ValueError("num_context_tokens must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.num_context_tokens = int(num_context_tokens)
        self.generic_fallback = bool(generic_fallback)

        # LayerNorm has no affine terms so an absent/all-zero path cannot gain a
        # learned constant.  All token projections and pooling queries are
        # shared by every physical slot.
        self.input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.key_proj = nn.Linear(hidden_dim, context_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, context_dim, bias=False)
        self.pool_queries = nn.Parameter(torch.empty(num_context_tokens, context_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.normal_(self.pool_queries, mean=0.0, std=self.context_dim**-0.5)

    def forward(
        self,
        raw_hidden: torch.Tensor,
        clause_masks: torch.Tensor,
        *,
        token_valid: Optional[torch.Tensor] = None,
        generic_token_mask: Optional[torch.Tensor] = None,
        generic_fallback: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(contexts, clause_missing)``.

        ``token_valid`` and ``generic_token_mask`` are boolean ``[B, L]``
        tensors.  ``token_valid`` always constrains both supervised and generic
        pooling, so padding can never be selected.  Generic fallback happens
        only when the effective supervised clause has no valid token.
        """

        if raw_hidden.ndim != 3:
            raise ValueError(
                f"raw_hidden must have shape [B,L,D], got {tuple(raw_hidden.shape)}."
            )
        batch, length, width = raw_hidden.shape
        if width != self.hidden_dim:
            raise ValueError(
                f"raw_hidden width must be {self.hidden_dim}, got {width}."
            )
        if clause_masks.ndim != 3:
            raise ValueError(
                f"clause_masks must have shape [B,S,L], got {tuple(clause_masks.shape)}."
            )
        slots = clause_masks.shape[1]
        _check_bool_mask(clause_masks, (batch, slots, length), "clause_masks")

        if token_valid is None:
            valid_tokens = torch.ones(
                (batch, length), dtype=torch.bool, device=raw_hidden.device
            )
        else:
            _check_bool_mask(token_valid, (batch, length), "token_valid")
            valid_tokens = token_valid.to(device=raw_hidden.device)

        # Moving masks between devices is cheap and gives a useful error for
        # malformed values before any large projection is performed.
        clause_masks = clause_masks.to(device=raw_hidden.device)
        clause_token_mask = clause_masks & valid_tokens[:, None, :]
        clause_missing = ~clause_token_mask.any(dim=-1)

        use_fallback = self.generic_fallback if generic_fallback is None else bool(generic_fallback)
        effective_mask = clause_token_mask
        if use_fallback:
            if generic_token_mask is None:
                fallback_mask = valid_tokens
            else:
                _check_bool_mask(
                    generic_token_mask, (batch, length), "generic_token_mask"
                )
                fallback_mask = generic_token_mask.to(device=raw_hidden.device) & valid_tokens
            effective_mask = torch.where(
                clause_missing[..., None],
                fallback_mask[:, None, :],
                clause_token_mask,
            )

        normed = self.input_norm(raw_hidden)
        keys = self.key_proj(normed)  # [B,L,R]
        values = self.value_proj(normed)  # [B,L,R]
        shared_logits = torch.einsum("kr,blr->bkl", self.pool_queries, keys)
        shared_logits = shared_logits / math.sqrt(self.context_dim)
        logits = shared_logits[:, None, :, :].expand(
            batch, slots, self.num_context_tokens, length
        )

        select = effective_mask[:, :, None, :]
        masked_logits = logits.masked_fill(~select, -torch.inf)

        # Softmax(all -inf) is NaN.  Replace only those rows temporarily, then
        # force their weights back to exact zero.  This keeps both forward and
        # backward finite while preserving -inf semantics for real masks.
        effective_missing = ~effective_mask.any(dim=-1)
        empty_rows = effective_missing[:, :, None, None]
        safe_logits = torch.where(empty_rows, torch.zeros_like(masked_logits), masked_logits)
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(empty_rows, torch.zeros_like(weights), weights)
        contexts = torch.einsum("bskl,blr->bskr", weights, values)
        return contexts, clause_missing


class CompetitiveReferenceRouter(nn.Module):
    """Competitively merge per-slot reference candidates plus a null route.

    The learned scoring function is shared across slots.  It combines the
    candidate at each target/head location with that slot's pooled clause
    context and compares the result to the target token.  Invalid slots receive
    an exact ``-inf`` logit.  A zero-valued null candidate with a fixed zero
    logit is appended, so even an all-missing row has a well-defined softmax.

    Args:
        candidate_dim: Candidate head width ``D``.
        context_dim: Clause-context width ``R``.
        target_dim: Target-token width ``Q``.
        router_dim: Shared latent scoring width.

    Forward inputs have shapes ``candidates=[B,S,N,H,D]``, clause contexts
    ``[B,S,K,R]`` (or the simplified ``[B,S,R]``), ``target=[B,N,Q]``, and
    ``slot_valid=[B,S]``.  It returns ``merged=[B,N,H,D]`` and probabilities
    ``[B,N,H,S+1]``.  The last probability is always the null route and the
    final axis sums to one.
    """

    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        target_dim: int,
        router_dim: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("candidate_dim", candidate_dim),
            ("context_dim", context_dim),
            ("target_dim", target_dim),
            ("router_dim", router_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.candidate_dim = int(candidate_dim)
        self.context_dim = int(context_dim)
        self.target_dim = int(target_dim)
        self.router_dim = int(router_dim)

        # No max-slots argument, slot embedding, per-slot bias, or parameter
        # with an S axis appears here.  These projections are applied pointwise.
        self.candidate_key = nn.Linear(candidate_dim, router_dim, bias=False)
        self.context_key = nn.Linear(context_dim, router_dim, bias=False)
        self.target_query = nn.Linear(target_dim, router_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.candidate_key.weight)
        nn.init.xavier_uniform_(self.context_key.weight)
        nn.init.xavier_uniform_(self.target_query.weight)

    def _validate_inputs(
        self,
        candidates: torch.Tensor,
        clause_contexts: torch.Tensor,
        target: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> tuple[int, int, int, int, int, torch.Tensor]:
        if candidates.ndim != 5:
            raise ValueError(
                f"candidates must have shape [B,S,N,H,D], got {tuple(candidates.shape)}."
            )
        batch, slots, target_len, heads, width = candidates.shape
        if width != self.candidate_dim:
            raise ValueError(
                f"candidate width must be {self.candidate_dim}, got {width}."
            )
        if target.ndim != 3 or tuple(target.shape[:2]) != (batch, target_len):
            raise ValueError(
                f"target must have shape [{batch},{target_len},Q], got {tuple(target.shape)}."
            )
        if target.shape[-1] != self.target_dim:
            raise ValueError(
                f"target width must be {self.target_dim}, got {target.shape[-1]}."
            )
        _check_bool_mask(slot_valid, (batch, slots), "slot_valid")

        if clause_contexts.ndim == 4:
            if clause_contexts.shape[2] <= 0:
                raise ValueError("clause_contexts must contain at least one context token.")
            context = clause_contexts.mean(dim=2)
        elif clause_contexts.ndim == 3:
            context = clause_contexts
        else:
            raise ValueError(
                "clause_contexts must have shape [B,S,K,R] or [B,S,R], "
                f"got {tuple(clause_contexts.shape)}."
            )
        if tuple(context.shape) != (batch, slots, self.context_dim):
            raise ValueError(
                "clause_contexts reduce to shape "
                f"[{batch},{slots},{self.context_dim}], got {tuple(context.shape)}."
            )
        return batch, slots, target_len, heads, width, context

    def compute_logits(
        self,
        candidates: torch.Tensor,
        clause_contexts: torch.Tensor,
        target: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return pre-softmax logits ``[B,N,H,S+1]``.

        This public helper makes the fail-closed ``-inf`` mask directly
        auditable.  The final element is the null logit and is exactly zero.
        """

        batch, slots, target_len, heads, _, context = self._validate_inputs(
            candidates, clause_contexts, target, slot_valid
        )
        # [B,S,N,H,A] + [B,S,1,1,A]
        keys = self.candidate_key(candidates)
        keys = keys + self.context_key(context)[:, :, None, None, :]
        queries = self.target_query(target)[:, None, :, None, :]
        slot_logits = torch.sum(keys * queries, dim=-1) / math.sqrt(self.router_dim)
        slot_logits = slot_logits.permute(0, 2, 3, 1)  # [B,N,H,S]

        valid = slot_valid.to(device=candidates.device)[:, None, None, :]
        slot_logits = slot_logits.masked_fill(~valid, -torch.inf)
        null_logits = torch.zeros(
            (batch, target_len, heads, 1),
            dtype=slot_logits.dtype,
            device=slot_logits.device,
        )
        return torch.cat((slot_logits, null_logits), dim=-1)

    def forward(
        self,
        candidates: torch.Tensor,
        clause_contexts: torch.Tensor,
        target: torch.Tensor,
        slot_valid: torch.Tensor,
        *,
        temperature: float = 1.0,
        null_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and strictly positive.")
        logits = self.compute_logits(candidates, clause_contexts, target, slot_valid)
        logits = logits / float(temperature)
        if not null_enabled:
            # No-reference inputs are hard-bypassed by Anima before reaching
            # this module.  With references present, disabling null is a safe
            # A1 curriculum operation and cannot create an all-masked row.
            if not bool(slot_valid.any(dim=-1).all()):
                raise ValueError("null may be disabled only when every sample has a valid reference.")
            logits = logits.clone()
            logits[..., -1] = -torch.inf
        probabilities = torch.softmax(logits, dim=-1)

        # Append a literal zero candidate.  Including it in this expression
        # (instead of merely multiplying slot probabilities) makes the route
        # axis and probability contract explicit.
        candidates_by_route = candidates.permute(0, 2, 3, 1, 4)
        null_candidate = torch.zeros_like(candidates_by_route[..., :1, :])
        candidates_by_route = torch.cat((candidates_by_route, null_candidate), dim=-2)
        merged = torch.sum(probabilities.unsqueeze(-1) * candidates_by_route, dim=-2)
        return merged, probabilities


def zero_init_linear(
    in_features: int, out_features: int, *, bias: bool = True
) -> nn.Linear:
    """Create a linear layer whose output and input-Jacobian start at zero."""

    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ZeroInitQueryCorrection(nn.Module):
    """Residual query correction with an exactly neutral initialization.

    ``query`` and ``condition`` may have arbitrary broadcastable leading
    dimensions and widths ``query_dim`` / ``condition_dim``.  A batch-only
    condition ``[B,C]`` is automatically interpreted as ``[B,1,C]`` for a
    token query ``[B,N,Q]``.  The final projection is zero initialized, hence
    :meth:`forward` returns the input query bit-for-bit at initialization while
    its output projection still receives a useful first-step gradient.
    """

    def __init__(
        self,
        query_dim: int,
        condition_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or condition_dim <= 0:
            raise ValueError("query_dim and condition_dim must be positive.")
        if hidden_dim is None:
            hidden_dim = max(query_dim, condition_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.query_dim = int(query_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(query_dim, elementwise_affine=False)
        self.condition_norm = nn.LayerNorm(condition_dim, elementwise_affine=False)
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim, bias=False)
        self.output_proj = zero_init_linear(hidden_dim, query_dim, bias=True)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.condition_proj.weight)

    def correction(self, query: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return the zero-initialized delta without adding the residual."""

        if query.shape[-1] != self.query_dim:
            raise ValueError(
                f"query width must be {self.query_dim}, got {query.shape[-1]}."
            )
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition width must be {self.condition_dim}, got {condition.shape[-1]}."
            )
        while condition.ndim < query.ndim:
            condition = condition.unsqueeze(-2)
        try:
            leading_shape = torch.broadcast_shapes(query.shape[:-1], condition.shape[:-1])
        except RuntimeError as error:
            raise ValueError(
                "query and condition leading dimensions are not broadcastable: "
                f"{tuple(query.shape[:-1])} vs {tuple(condition.shape[:-1])}."
            ) from error

        query_features = self.query_proj(self.query_norm(query)).expand(
            *leading_shape, self.hidden_dim
        )
        condition_features = self.condition_proj(self.condition_norm(condition)).expand(
            *leading_shape, self.hidden_dim
        )
        hidden = torch.nn.functional.silu(query_features + condition_features)
        return self.output_proj(hidden)

    def forward(self, query: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return query + self.correction(query, condition)


__all__ = [
    "SlotClausePooler",
    "CompetitiveReferenceRouter",
    "ZeroInitQueryCorrection",
    "zero_init_linear",
]

