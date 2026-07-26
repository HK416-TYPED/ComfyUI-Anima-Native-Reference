"""Deterministic text-to-logical-reference binding for Anima V4.

This module is deliberately independent of PyTorch and Transformers.  It is
used while building/auditing a manifest and by the tokenizer cache adapter.
The parser never infers semantic roles such as ``scene`` or ``character`` from
physical image order: only an explicit textual index can produce a binding.

All character spans returned by this module refer to
``BindingParseResult.canonical_instruction``.  Callers must tokenize that exact
string when converting the spans to token masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, Mapping, Sequence


PARSER_VERSION = "anima-reference-binding/v1"


class BindingStatus(str, Enum):
    """Manifest eligibility of one instruction/reference binding."""

    SAFE = "SAFE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


class PromptRewriteError(ValueError):
    """Raised when a prompt cannot be index-swapped without guessing roles."""


class SpanAlignmentError(ValueError):
    """Raised when verified character spans cannot be represented by tokens."""


@dataclass(frozen=True)
class ReferenceMention:
    """One explicit textual reference mention.

    ``slot_id`` is zero based even though the surface text is one based.
    """

    slot_id: int
    char_span: tuple[int, int]
    text: str
    form: str


@dataclass(frozen=True)
class ClauseBinding:
    """A minimal punctuation-delimited clause containing reference mentions."""

    char_span: tuple[int, int]
    slot_ids: tuple[int, ...]
    mention_spans: tuple[tuple[int, int], ...]
    multi_bind: bool


@dataclass(frozen=True)
class BindingParseResult:
    """Auditable result of parsing one canonical instruction."""

    original_instruction: str
    canonical_instruction: str
    parser_version: str
    status: BindingStatus
    reason: str
    expected_slot_ids: tuple[int, ...]
    mentions: tuple[ReferenceMention, ...]
    clauses: tuple[ClauseBinding, ...]

    @property
    def binding_valid(self) -> bool:
        return self.status is BindingStatus.SAFE

    @property
    def mentioned_slot_ids(self) -> tuple[int, ...]:
        return tuple(sorted({mention.slot_id for mention in self.mentions}))

    @property
    def mentioned_slot_mask(self) -> tuple[bool, ...]:
        """Dense mask over slots through the greatest expected/mentioned ID."""

        all_ids = (*self.expected_slot_ids, *self.mentioned_slot_ids)
        if not all_ids:
            return ()
        mentioned = set(self.mentioned_slot_ids)
        return tuple(slot_id in mentioned for slot_id in range(max(all_ids) + 1))

    @property
    def reference_text_spans(self) -> Mapping[int, tuple[tuple[int, int], ...]]:
        """Clause spans grouped by logical slot, suitable for manifest JSON."""

        grouped: dict[int, list[tuple[int, int]]] = {
            slot_id: [] for slot_id in self.expected_slot_ids
        }
        for clause in self.clauses:
            for slot_id in clause.slot_ids:
                grouped.setdefault(slot_id, []).append(clause.char_span)
        return {slot_id: tuple(spans) for slot_id, spans in grouped.items()}

    @property
    def reference_bindings(self) -> tuple[dict[str, object], ...]:
        """JSON-friendly mention-to-slot records for an audited manifest."""

        return tuple(
            {
                "mention": mention.text,
                "char_span": list(mention.char_span),
                "logical_slot_id": mention.slot_id,
                "form": mention.form,
            }
            for mention in self.mentions
        )


_WHITESPACE_RE = re.compile(r"\s+")

# Longest alternatives are intentionally first.  One ``finditer`` pass means
# ``reference image 1`` cannot also be emitted as a second ``image 1`` mention.
_REFERENCE_MENTION_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?P<reference_image>reference\s+image\s*#?\s*(?P<reference_image_n>[12]))"
    r"|(?P<ref>ref(?:erence)?\s*#?\s*(?P<ref_n>[12]))"
    r"|(?P<image>image\s*#?\s*(?P<image_n>[12]))"
    r"|(?P<ordinal_image>(?P<ordinal>first|second|1st|2nd)\s+(?:reference\s+)?image)"
    r"|(?P<ordinal_reference>(?P<ordinal_ref>first|second|1st|2nd)\s+reference)"
    r")(?!\w)",
    flags=re.IGNORECASE,
)

# Explicit but unsupported numeric references must not silently degrade to an
# AMBIGUOUS generic prompt.  They are an invalid v1 parser input.
_ANY_NUMERIC_REFERENCE_RE = re.compile(
    r"(?<!\w)(?:reference\s+image|ref(?:erence)?|image)\s*#?\s*(?P<number>\d+)(?!\w)",
    flags=re.IGNORECASE,
)

_REFERENCE_LIKE_RE = re.compile(
    r"\b(?:image|images|ref|refs|reference|references)\b", flags=re.IGNORECASE
)

# Clause delimiters are excluded from the returned span.  Coordinators are not
# split heuristically: a clause mentioning two slots remains explicitly marked
# ``multi_bind`` rather than pretending that a fragile role parse is certain.
_CLAUSE_DELIMITER_RE = re.compile(r"[,;.!?\uFF0C\uFF1B\u3002\uFF01\uFF1F\n]+")

_ORDINAL_SWAP = {
    "first": "second",
    "second": "first",
    "1st": "2nd",
    "2nd": "1st",
}


def canonicalize_instruction(instruction: str) -> str:
    """Return the stable text representation used for spans and swapping."""

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    return _WHITESPACE_RE.sub(" ", instruction.strip())


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source.istitle():
        return replacement.title()
    return replacement


def _slot_from_match(match: re.Match[str]) -> tuple[int, str]:
    for group_name, form in (
        ("reference_image_n", "reference_image"),
        ("ref_n", "ref"),
        ("image_n", "image"),
    ):
        number = match.group(group_name)
        if number is not None:
            return int(number) - 1, form

    ordinal = match.group("ordinal") or match.group("ordinal_ref")
    if ordinal is None:  # pragma: no cover - guarded by the compiled pattern
        raise AssertionError("reference mention regex produced no index")
    return (0 if ordinal.casefold() in {"first", "1st"} else 1), "ordinal"


def find_reference_mentions(instruction: str) -> tuple[ReferenceMention, ...]:
    """Find supported explicit mentions in canonical text, without role inference."""

    canonical = canonicalize_instruction(instruction)
    mentions: list[ReferenceMention] = []
    for match in _REFERENCE_MENTION_RE.finditer(canonical):
        slot_id, form = _slot_from_match(match)
        mentions.append(
            ReferenceMention(
                slot_id=slot_id,
                char_span=match.span(),
                text=match.group(0),
                form=form,
            )
        )
    return tuple(mentions)


def _nonempty_clause_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for delimiter in _CLAUSE_DELIMITER_RE.finditer(text):
        start, end = cursor, delimiter.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
        cursor = delimiter.end()
    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))
    return tuple(spans)


def parse_reference_bindings(
    instruction: str,
    *,
    expected_slot_ids: Sequence[int] = (0, 1),
) -> BindingParseResult:
    """Parse explicit reference mentions and assign auditable clause spans.

    ``expected_slot_ids`` describes the logical slots actually present in the
    sample, not physical list positions.  A dual-reference row mentioning only
    one of its two expected slots is AMBIGUOUS.  A single-reference row can be
    SAFE with e.g. ``expected_slot_ids=(1,)`` and an explicit ``Image 2``.
    """

    original = instruction if isinstance(instruction, str) else ""
    try:
        canonical = canonicalize_instruction(instruction)
    except (TypeError, AttributeError):
        canonical = ""

    try:
        expected = tuple(int(slot_id) for slot_id in expected_slot_ids)
    except (TypeError, ValueError):
        expected = ()

    def result(
        status: BindingStatus,
        reason: str,
        mentions: tuple[ReferenceMention, ...] = (),
        clauses: tuple[ClauseBinding, ...] = (),
    ) -> BindingParseResult:
        return BindingParseResult(
            original_instruction=original,
            canonical_instruction=canonical,
            parser_version=PARSER_VERSION,
            status=status,
            reason=reason,
            expected_slot_ids=expected,
            mentions=mentions,
            clauses=clauses,
        )

    if not canonical:
        return result(BindingStatus.INVALID, "empty_instruction")
    if not expected or any(slot_id < 0 for slot_id in expected) or len(set(expected)) != len(expected):
        return result(BindingStatus.INVALID, "invalid_expected_slot_ids")

    mentions = find_reference_mentions(canonical)
    supported_spans = {mention.char_span for mention in mentions}
    for numeric_match in _ANY_NUMERIC_REFERENCE_RE.finditer(canonical):
        if numeric_match.span() not in supported_spans:
            return result(BindingStatus.INVALID, "unsupported_reference_index", mentions)

    expected_set = set(expected)
    if any(mention.slot_id not in expected_set for mention in mentions):
        return result(BindingStatus.INVALID, "mentioned_slot_not_present", mentions)

    clause_bindings: list[ClauseBinding] = []
    for clause_span in _nonempty_clause_spans(canonical):
        clause_mentions = tuple(
            mention
            for mention in mentions
            if mention.char_span[0] >= clause_span[0]
            and mention.char_span[1] <= clause_span[1]
        )
        if not clause_mentions:
            continue
        slot_ids = tuple(sorted({mention.slot_id for mention in clause_mentions}))
        clause_bindings.append(
            ClauseBinding(
                char_span=clause_span,
                slot_ids=slot_ids,
                mention_spans=tuple(mention.char_span for mention in clause_mentions),
                multi_bind=len(slot_ids) > 1,
            )
        )

    mentioned = {mention.slot_id for mention in mentions}
    if not mentions:
        reason = "unindexed_reference_prompt" if _REFERENCE_LIKE_RE.search(canonical) else "no_reference_mentions"
        return result(BindingStatus.AMBIGUOUS, reason)
    missing = expected_set - mentioned
    if missing:
        return result(BindingStatus.AMBIGUOUS, "missing_expected_slot_mentions", mentions, tuple(clause_bindings))
    if not clause_bindings:
        return result(BindingStatus.INVALID, "mention_without_clause", mentions)
    return result(BindingStatus.SAFE, "explicit_verified_bindings", mentions, tuple(clause_bindings))


def swap_reference_indices(instruction: str) -> str:
    """Swap explicit logical slots 0/1 in one substitution pass.

    The function canonicalizes whitespace first.  Consequently the exact
    involution contract is ``swap(swap(p)) == canonicalize_instruction(p)``.
    """

    canonical = canonicalize_instruction(instruction)
    if not canonical:
        raise PromptRewriteError("prompt must be a non-empty string")
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        value = match.group(0)

        number_group = None
        for group_name in ("reference_image_n", "ref_n", "image_n"):
            if match.group(group_name) is not None:
                number_group = group_name
                break
        if number_group is not None:
            start, end = match.span(number_group)
            relative_start = start - match.start()
            relative_end = end - match.start()
            swapped_number = "2" if match.group(number_group) == "1" else "1"
            return value[:relative_start] + swapped_number + value[relative_end:]

        ordinal_group = "ordinal" if match.group("ordinal") is not None else "ordinal_ref"
        ordinal = match.group(ordinal_group)
        start, end = match.span(ordinal_group)
        relative_start = start - match.start()
        relative_end = end - match.start()
        swapped_ordinal = _match_case(ordinal, _ORDINAL_SWAP[ordinal.casefold()])
        return value[:relative_start] + swapped_ordinal + value[relative_end:]

    swapped = _REFERENCE_MENTION_RE.sub(replace, canonical)
    if replacements == 0:
        raise PromptRewriteError("prompt has no supported explicit reference index")
    return swapped


def _normalise_spans(spans: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for span in spans:
        if len(span) != 2:
            raise SpanAlignmentError(f"character span must contain two integers, got {span!r}")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise SpanAlignmentError(f"invalid character span {(start, end)!r}")
        result.append((start, end))
    return tuple(result)


def char_spans_to_token_mask(
    offset_mapping: Sequence[Sequence[int]],
    spans: Iterable[Sequence[int]],
    *,
    attention_mask: Sequence[int | bool] | None = None,
    text_length: int | None = None,
    require_full_coverage: bool = True,
) -> tuple[bool, ...]:
    """Convert verified character spans to a token-overlap mask.

    A token is selected iff ``token_start < span_end`` and
    ``token_end > span_start``.  ``(0, 0)`` special/padding offsets are ignored.
    In strict mode every span must be covered through its final character; this
    turns tokenizer truncation into a fail-closed ``SpanAlignmentError`` rather
    than a silently partial clause mask.
    """

    offsets: list[tuple[int, int]] = []
    for offset in offset_mapping:
        if len(offset) != 2:
            raise SpanAlignmentError(f"token offset must contain two integers, got {offset!r}")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start:
            raise SpanAlignmentError(f"invalid token offset {(start, end)!r}")
        offsets.append((start, end))

    if attention_mask is None:
        active = [True] * len(offsets)
    else:
        if len(attention_mask) != len(offsets):
            raise SpanAlignmentError("attention_mask length does not match offset_mapping")
        active = [bool(value) for value in attention_mask]

    normalised_spans = _normalise_spans(spans)
    if text_length is not None and any(end > int(text_length) for _, end in normalised_spans):
        raise SpanAlignmentError("character span exceeds canonical instruction length")

    mask = [False] * len(offsets)
    for span_start, span_end in normalised_spans:
        selected: list[tuple[int, int, int]] = []
        for token_index, ((token_start, token_end), is_active) in enumerate(zip(offsets, active, strict=True)):
            if not is_active or token_end <= token_start:
                continue
            if token_start < span_end and token_end > span_start:
                mask[token_index] = True
                selected.append((token_index, token_start, token_end))
        if not selected:
            raise SpanAlignmentError(f"character span {(span_start, span_end)} has no active token coverage")
        if require_full_coverage:
            covered_start = min(token_start for _, token_start, _ in selected)
            covered_end = max(token_end for _, _, token_end in selected)
            if covered_start > span_start or covered_end < span_end:
                raise SpanAlignmentError(
                    f"character span {(span_start, span_end)} is only partially tokenized; "
                    "the instruction was likely truncated"
                )
    return tuple(mask)


def build_slot_clause_token_masks(
    parsed: BindingParseResult,
    offset_mapping: Sequence[Sequence[int]],
    *,
    attention_mask: Sequence[int | bool] | None = None,
) -> Mapping[int, tuple[bool, ...]]:
    """Build one Qwen token mask per expected logical slot.

    Only SAFE results are eligible for supervised masks.  AMBIGUOUS/INVALID
    rows must remain fail-soft for generation but cannot enter pointer loss.
    """

    if not parsed.binding_valid:
        raise SpanAlignmentError(
            f"cannot build supervised clause masks for {parsed.status.value}: {parsed.reason}"
        )
    masks: dict[int, tuple[bool, ...]] = {}
    spans_by_slot = parsed.reference_text_spans
    for slot_id in parsed.expected_slot_ids:
        spans = spans_by_slot.get(slot_id, ())
        if not spans:
            raise SpanAlignmentError(f"SAFE binding has no clause span for logical slot {slot_id}")
        masks[slot_id] = char_spans_to_token_mask(
            offset_mapping,
            spans,
            attention_mask=attention_mask,
            text_length=len(parsed.canonical_instruction),
            require_full_coverage=True,
        )
    return masks


__all__ = [
    "PARSER_VERSION",
    "BindingStatus",
    "PromptRewriteError",
    "SpanAlignmentError",
    "ReferenceMention",
    "ClauseBinding",
    "BindingParseResult",
    "canonicalize_instruction",
    "find_reference_mentions",
    "parse_reference_bindings",
    "swap_reference_indices",
    "char_spans_to_token_mask",
    "build_slot_clause_token_masks",
]

