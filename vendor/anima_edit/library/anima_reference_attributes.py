"""Versioned attribute labels for prompt-to-reference routing.

The runtime model never guesses that a physical image has a semantic role.
Instead, optional manifest supervision binds one or more *attributes* to a
logical image index.  This module is deliberately free of PyTorch so manifest
builders, audits, and tests can share exactly the same validation rules.

Example manifest fragment::

    {
      "reference_attribute_bindings": {
        "0": ["pose", "composition"],
        "1": ["identity", "clothing"]
      }
    }

Keys are zero-based logical image indices.  Physical list order is never used
as a proxy for identity/source/style/background semantics.
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Mapping, Sequence


REFERENCE_ATTRIBUTE_SCHEMA_VERSION = "anima-reference-attributes/v1"

# Keep the order frozen: it is the checkpoint classifier/output order.
REFERENCE_ATTRIBUTE_VOCAB: tuple[str, ...] = (
    "generic",
    "identity",
    "face",
    "hair",
    "clothing",
    "pose",
    "expression",
    "style",
    "background",
    "composition",
    "object",
    "color",
    "lighting",
    "text",
    "geometry",
    "preserve",
)
REFERENCE_ATTRIBUTE_TO_ID = {
    name: index for index, name in enumerate(REFERENCE_ATTRIBUTE_VOCAB)
}


class ReferenceAttributeError(ValueError):
    """Raised when a manifest attribute binding is ambiguous or malformed."""


def normalize_reference_attributes(attributes: Iterable[str]) -> tuple[str, ...]:
    """Validate, de-duplicate, and return attributes in checkpoint order."""

    if isinstance(attributes, (str, bytes)):
        raise ReferenceAttributeError("attribute labels must be a sequence, not a string")
    seen: set[str] = set()
    for attribute in attributes:
        if not isinstance(attribute, str):
            raise ReferenceAttributeError(
                f"attribute labels must be strings, got {type(attribute).__name__}"
            )
        canonical = attribute.strip().casefold()
        if canonical not in REFERENCE_ATTRIBUTE_TO_ID:
            raise ReferenceAttributeError(
                f"unsupported reference attribute {attribute!r}; expected one of "
                f"{REFERENCE_ATTRIBUTE_VOCAB}"
            )
        seen.add(canonical)
    if not seen:
        raise ReferenceAttributeError("an audited slot must contain at least one attribute")
    return tuple(name for name in REFERENCE_ATTRIBUTE_VOCAB if name in seen)


def _normalize_slot_id(value: object) -> int:
    if isinstance(value, bool):
        raise ReferenceAttributeError("logical slot IDs must be non-negative integers")
    if isinstance(value, Integral):
        slot_id = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        slot_id = int(value.strip())
    else:
        raise ReferenceAttributeError(
            f"logical slot ID {value!r} must be a non-negative integer"
        )
    if slot_id < 0:
        raise ReferenceAttributeError("logical slot IDs must be non-negative integers")
    return slot_id


def normalize_reference_attribute_bindings(
    bindings: Mapping[object, Iterable[str]] | None,
    *,
    expected_slot_ids: Sequence[int] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Validate JSON-style slot-to-attribute bindings.

    Unlabelled slots are intentionally allowed: they simply do not contribute
    to attribute BCE.  A binding for a slot absent from the sample fails closed.
    """

    if bindings is None:
        return {}
    if not isinstance(bindings, Mapping):
        raise ReferenceAttributeError("reference_attribute_bindings must be a mapping")

    expected: set[int] | None = None
    if expected_slot_ids is not None:
        expected = {_normalize_slot_id(slot_id) for slot_id in expected_slot_ids}

    normalized: dict[int, tuple[str, ...]] = {}
    for raw_slot_id, raw_attributes in bindings.items():
        slot_id = _normalize_slot_id(raw_slot_id)
        if expected is not None and slot_id not in expected:
            raise ReferenceAttributeError(
                f"attribute binding names logical slot {slot_id}, but sample contains "
                f"only {tuple(sorted(expected))}"
            )
        if slot_id in normalized:
            raise ReferenceAttributeError(f"duplicate logical slot binding {slot_id}")
        normalized[slot_id] = normalize_reference_attributes(raw_attributes)
    return dict(sorted(normalized.items()))


def remap_reference_attribute_bindings(
    bindings: Mapping[object, Iterable[str]] | None,
    slot_mapping: Mapping[int, int],
    *,
    expected_slot_ids: Sequence[int] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Apply a logical-slot permutation without changing attribute meaning."""

    normalized = normalize_reference_attribute_bindings(bindings)
    mapping = {
        _normalize_slot_id(old): _normalize_slot_id(new)
        for old, new in slot_mapping.items()
    }
    if len(set(mapping.values())) != len(mapping):
        raise ReferenceAttributeError("slot_mapping must be one-to-one")

    remapped: dict[int, tuple[str, ...]] = {}
    for old_slot, attributes in normalized.items():
        new_slot = mapping.get(old_slot, old_slot)
        if new_slot in remapped:
            raise ReferenceAttributeError(
                f"slot permutation aliases multiple bindings onto logical slot {new_slot}"
            )
        remapped[new_slot] = attributes
    return normalize_reference_attribute_bindings(
        remapped,
        expected_slot_ids=expected_slot_ids,
    )


def dense_reference_attribute_targets(
    bindings: Mapping[object, Iterable[str]] | None,
    *,
    max_slots: int,
    expected_slot_ids: Sequence[int] | None = None,
) -> tuple[list[list[float]], list[bool]]:
    """Return dense ``[S,A]`` multi-hot targets and an audited-slot mask."""

    if isinstance(max_slots, bool) or not isinstance(max_slots, Integral) or max_slots <= 0:
        raise ReferenceAttributeError("max_slots must be a positive integer")
    normalized = normalize_reference_attribute_bindings(
        bindings,
        expected_slot_ids=expected_slot_ids,
    )
    targets = [
        [0.0 for _ in REFERENCE_ATTRIBUTE_VOCAB] for _ in range(int(max_slots))
    ]
    valid = [False for _ in range(int(max_slots))]
    for slot_id, attributes in normalized.items():
        if slot_id >= max_slots:
            raise ReferenceAttributeError(
                f"logical slot {slot_id} exceeds dense max_slots={max_slots}"
            )
        valid[slot_id] = True
        for attribute in attributes:
            targets[slot_id][REFERENCE_ATTRIBUTE_TO_ID[attribute]] = 1.0
    return targets, valid


__all__ = [
    "REFERENCE_ATTRIBUTE_SCHEMA_VERSION",
    "REFERENCE_ATTRIBUTE_VOCAB",
    "REFERENCE_ATTRIBUTE_TO_ID",
    "ReferenceAttributeError",
    "normalize_reference_attributes",
    "normalize_reference_attribute_bindings",
    "remap_reference_attribute_bindings",
    "dense_reference_attribute_targets",
]
