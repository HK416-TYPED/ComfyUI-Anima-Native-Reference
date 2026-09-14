# Anima model loading/saving utilities

import json
import hashlib
import math
import os
from typing import Any, Dict, List, Optional, Union
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device  # kept for potential future use
from accelerate import init_empty_weights

from _anima_native_ref_vendor.library import anima_models
from _anima_native_ref_vendor.library.anima_reference_attributes import (
    REFERENCE_ATTRIBUTE_SCHEMA_VERSION,
    REFERENCE_ATTRIBUTE_VOCAB,
)
from _anima_native_ref_vendor.library.safetensors_utils import get_split_weight_filenames
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Original Anima high-precision keys. Kept for reference, but not used currently.
# # Keys that should stay in high precision (float32/bfloat16, not quantized)
# KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
# ".embed." excludes Embedding in LLMAdapter
FP8_OPTIMIZATION_EXCLUDE_KEYS = [
    "_embedder",
    "norm",
    "adaln",
    "final_layer",
    ".embed.",
    "native_reference",
    "edit_scope",
    "reference_slot",
    "reference_type",
]


# Native multi-reference checkpoints are self-describing.  Keep the metadata
# names generic: a reference slot is only an image index, never a hard-coded
# source/identity/style role.
ANIMA_NATIVE_REFERENCE_VERSION = 1
ANIMA_NATIVE_REFERENCE_METADATA_KEY = "anima_native_reference_conditioning"
ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY = "anima_native_reference_config"
ANIMA_NATIVE_REFERENCE_STATE_KEYS = (
    "reference_slot_embeddings.weight",
    "reference_type_embedding",
)
ANIMA_NATIVE_REFERENCE_STATE_MARKER = ".native_reference_attn."
ANIMA_NATIVE_REFERENCE_INDEX_STATE_PREFIX = "native_reference_index_conditioner."
ANIMA_NATIVE_REFERENCE_ATTRIBUTE_STATE_PREFIX = "native_reference_attribute_router."
ANIMA_NATIVE_REFERENCE_CONTEXT_STATE_PREFIX = "native_reference_context_conditioner."
ANIMA_MASKLESS_EDIT_SCOPE_STATE_PREFIX = "edit_scope_predictor."
ANIMA_MASKLESS_EDIT_SCOPE_VERSION = "implicit_source_scope_v1"
ANIMA_MASKLESS_EDIT_SCOPE_CONFIG_KEYS = (
    "scope_version",
    "scope_hidden_dim",
    "scope_initial_change_probability",
    "scope_alpha",
)
ANIMA_MASKLESS_EDIT_SCOPE_REQUIRED_STATE_KEYS = (
    "edit_scope_predictor.source_proj.weight",
    "edit_scope_predictor.auxiliary_proj.weight",
    "edit_scope_predictor.prompt_proj.weight",
    "edit_scope_predictor.source_clause_proj.weight",
    "edit_scope_predictor.timestep_proj.weight",
    "edit_scope_predictor.output.weight",
    "edit_scope_predictor.output.bias",
)
ANIMA_NATIVE_REFERENCE_TEXT_SLOT_ROUTER_STATE_MARKER = ".native_reference_attn.clause_pooler."
ANIMA_NATIVE_REFERENCE_ROUTING_MODES = (
    "legacy",
    "competitive_text_slot_v1",
    "native_context_v1",
)
ANIMA_NATIVE_REFERENCE_ROUTING_MODE_METADATA_KEY = "anima_native_reference_routing_mode"
ANIMA_NATIVE_REFERENCE_ROUTING_ALPHA_METADATA_KEY = "anima_native_reference_routing_alpha"
ANIMA_NATIVE_REFERENCE_ROUTER_TEMPERATURE_METADATA_KEY = "anima_native_reference_router_temperature"
ANIMA_NATIVE_REFERENCE_ROUTER_NULL_ENABLED_METADATA_KEY = "anima_native_reference_router_null_enabled"
ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY = "anima_native_reference_router_config"
_ANIMA_NATIVE_REFERENCE_ROUTING_CONFIG_KEYS = (
    "routing_mode",
    "routing_alpha",
    "router_temperature",
    "router_null_enabled",
)
_ANIMA_NATIVE_REFERENCE_ROUTING_REQUIRED_CONFIG_KEYS = (
    "routing_mode",
    "routing_alpha",
    "router_temperature",
)
ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTING_VERSION = "native_ref_attribute_router_v1"
ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY = (
    "anima_native_reference_attribute_router_config"
)
ANIMA_NATIVE_REFERENCE_ATTRIBUTE_REQUIRED_STATE_KEYS = (
    "native_reference_index_conditioner.visual_proj.weight",
    "native_reference_index_conditioner.router_proj.weight",
    "native_reference_attribute_router.text_proj.weight",
    "native_reference_attribute_router.attribute_classifier.weight",
    "native_reference_attribute_router.attribute_classifier.bias",
    "native_reference_attribute_router.attribute_embeddings",
    "native_reference_attribute_router.route_proj.weight",
)
ANIMA_NATIVE_REFERENCE_ATTRIBUTE_CONFIG_KEYS = (
    "attribute_routing_version",
    "attribute_schema_version",
    "attribute_vocab",
    "index_dim",
    "attribute_hidden_dim",
    "index_alpha",
    "attribute_alpha",
)
ANIMA_NATIVE_REFERENCE_CONTEXT_ROUTING_VERSION = "native_reference_context_v1"
ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY = (
    "anima_native_reference_context_config"
)
ANIMA_NATIVE_REFERENCE_CONTEXT_REQUIRED_STATE_KEYS = (
    "native_reference_context_conditioner.instance_conditioner.visual_proj.weight",
    "native_reference_context_conditioner.instance_conditioner.router_proj.weight",
    "native_reference_context_conditioner.role_visual_proj.weight",
    "native_reference_context_conditioner.role_router_proj.weight",
)
ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_KEYS = (
    "context_routing_version",
    "context_index_dim",
    "context_alpha",
)


def _validate_native_reference_attribute_state_shapes(
    shapes: Dict[str, tuple[int, ...]],
    *,
    source: str,
    expected_router_dim: Optional[int] = None,
) -> tuple[int, int]:
    """Validate the complete V6 tensor geometry and return ``(index_dim, hidden_dim)``.

    Exact key presence alone is not enough: a corrupted classifier can retain
    all expected names while silently changing the frozen attribute order or
    producing a route width that does not fit the V4 competitive carrier.
    Keeping this validation shared by inspection and save makes such files
    impossible to publish or load as a valid integrated checkpoint.
    """

    expected_keys = set(ANIMA_NATIVE_REFERENCE_ATTRIBUTE_REQUIRED_STATE_KEYS)
    actual_keys = set(shapes)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Incomplete native reference attribute-router state from {source}: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}."
        )

    expected_ranks = {
        "native_reference_index_conditioner.visual_proj.weight": 2,
        "native_reference_index_conditioner.router_proj.weight": 2,
        "native_reference_attribute_router.text_proj.weight": 2,
        "native_reference_attribute_router.attribute_classifier.weight": 2,
        "native_reference_attribute_router.attribute_classifier.bias": 1,
        "native_reference_attribute_router.attribute_embeddings": 2,
        "native_reference_attribute_router.route_proj.weight": 2,
    }
    bad_ranks = {
        key: shapes[key]
        for key, rank in expected_ranks.items()
        if len(shapes[key]) != rank
    }
    if bad_ranks:
        raise ValueError(
            f"Invalid native reference attribute-router tensor ranks from {source}: "
            f"{bad_ranks}."
        )

    visual = shapes["native_reference_index_conditioner.visual_proj.weight"]
    index_route = shapes["native_reference_index_conditioner.router_proj.weight"]
    text = shapes["native_reference_attribute_router.text_proj.weight"]
    classifier = shapes[
        "native_reference_attribute_router.attribute_classifier.weight"
    ]
    classifier_bias = shapes[
        "native_reference_attribute_router.attribute_classifier.bias"
    ]
    embeddings = shapes["native_reference_attribute_router.attribute_embeddings"]
    attribute_route = shapes["native_reference_attribute_router.route_proj.weight"]

    index_dim = int(visual[1])
    hidden_dim = int(text[0])
    attribute_count = len(REFERENCE_ATTRIBUTE_VOCAB)
    problems: List[str] = []
    if index_dim <= 0 or int(index_route[1]) != index_dim:
        problems.append(
            f"index feature width mismatch: visual={visual}, router={index_route}"
        )
    # These are deliberately different carriers in the real Anima graph:
    # visual_proj emits model_channels (2048 in Cosmos2/Anima), whereas
    # text_proj consumes crossattn_emb_channels (1024).  They meet only after
    # both branches project into router_dim, so equating visual[0] with text[1]
    # would reject every real V6 checkpoint while letting tiny tests pass by
    # coincidence.
    if int(visual[0]) <= 0:
        problems.append(f"visual carrier output width must be positive: {visual}")
    if int(text[1]) <= 0:
        problems.append(f"text carrier input width must be positive: {text}")
    if int(classifier[1]) != hidden_dim:
        problems.append(
            f"classifier hidden width {classifier[1]} != {hidden_dim}"
        )
    if tuple(embeddings) != (attribute_count, hidden_dim):
        problems.append(
            f"attribute_embeddings={embeddings}, expected={(attribute_count, hidden_dim)}"
        )
    if tuple(classifier) != (attribute_count, hidden_dim):
        problems.append(
            f"attribute_classifier.weight={classifier}, expected={(attribute_count, hidden_dim)}"
        )
    if tuple(classifier_bias) != (attribute_count,):
        problems.append(
            f"attribute_classifier.bias={classifier_bias}, expected={(attribute_count,)}"
        )
    if int(attribute_route[1]) != hidden_dim:
        problems.append(
            f"attribute route hidden width {attribute_route[1]} != {hidden_dim}"
        )
    if int(index_route[0]) != int(attribute_route[0]):
        problems.append(
            "index/attribute route output widths disagree: "
            f"{index_route[0]} != {attribute_route[0]}"
        )
    if expected_router_dim is not None and int(attribute_route[0]) != int(
        expected_router_dim
    ):
        problems.append(
            f"route output width {attribute_route[0]} != router_dim {expected_router_dim}"
        )
    if problems:
        raise ValueError(
            f"Invalid native reference attribute-router tensor geometry from {source}: "
            + "; ".join(problems)
            + "."
        )
    return index_dim, hidden_dim


def _normalise_anima_state_key(key: str) -> str:
    return key[len("net.") :] if key.startswith("net.") else key


def is_native_reference_state_key(key: str) -> bool:
    """Return whether *key* belongs to integrated native reference conditioning."""

    key = _normalise_anima_state_key(key)
    return (
        key in ANIMA_NATIVE_REFERENCE_STATE_KEYS
        or ANIMA_NATIVE_REFERENCE_STATE_MARKER in key
        or key.startswith(ANIMA_NATIVE_REFERENCE_INDEX_STATE_PREFIX)
        or key.startswith(ANIMA_NATIVE_REFERENCE_ATTRIBUTE_STATE_PREFIX)
        or key.startswith(ANIMA_NATIVE_REFERENCE_CONTEXT_STATE_PREFIX)
        or key.startswith(ANIMA_MASKLESS_EDIT_SCOPE_STATE_PREFIX)
    )


def tensor_raw_sha256(tensor: torch.Tensor) -> str:
    """Hash shape, dtype, and exact storage bytes without NumPy dtype coercion."""

    value = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def assert_non_native_state_bitwise_equal(
    source_state: Dict[str, torch.Tensor],
    candidate_state: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """Fail unless every non-native checkpoint tensor is byte-identical.

    The narrow :func:`is_native_reference_state_key` predicate is the sole
    allowlist.  Shape and dtype are compared separately because
    ``torch.equal`` considers some equal-valued tensors with different dtypes
    equal, which is not a publish-safe frozen-base contract.
    """

    source_keys = set(source_state)
    candidate_keys = set(candidate_state)
    if source_keys != candidate_keys:
        raise ValueError(
            "Checkpoint key sets differ: "
            f"missing={sorted(source_keys - candidate_keys)[:8]}, "
            f"unexpected={sorted(candidate_keys - source_keys)[:8]}."
        )
    digests = {}
    for key in sorted(source_keys):
        if is_native_reference_state_key(key):
            continue
        source = source_state[key]
        candidate = candidate_state[key]
        if source.shape != candidate.shape or source.dtype != candidate.dtype:
            raise ValueError(
                f"Frozen non-native tensor metadata changed for {key}: "
                f"{tuple(source.shape)}/{source.dtype} != "
                f"{tuple(candidate.shape)}/{candidate.dtype}."
            )
        source_digest = tensor_raw_sha256(source)
        candidate_digest = tensor_raw_sha256(candidate)
        if source_digest != candidate_digest:
            raise ValueError(
                f"Frozen non-native tensor bytes changed for {key}: "
                f"{source_digest} != {candidate_digest}."
            )
        digests[key] = source_digest
    return digests


def is_native_reference_text_slot_router_state_key(key: str) -> bool:
    """Return whether *key* is a V4 text-slot router parameter/buffer.

    This deliberately uses the narrow clause-pooler marker from the published
    checkpoint contract instead of treating every native-reference key as V4.
    Published V1/V2 checkpoints contain ``native_reference_attn`` keys too and
    must continue to load through the exact legacy graph.
    """

    return ANIMA_NATIVE_REFERENCE_TEXT_SLOT_ROUTER_STATE_MARKER in _normalise_anima_state_key(key)


def _as_checkpoint_file_list(model_files: Union[str, List[str]]) -> List[str]:
    if isinstance(model_files, str):
        model_files = [model_files]
    result: List[str] = []
    for model_file in model_files:
        split_files = get_split_weight_filenames(model_file)
        result.extend(split_files if split_files is not None else [model_file])
    return result


def _metadata_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_metadata_dict(raw_value: Any, metadata_key: str) -> Dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {metadata_key} metadata: {raw_value!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {metadata_key} metadata: expected a JSON object, got {raw_value!r}")
    return parsed


def _validate_native_reference_routing_mode(value: Any, *, source: str) -> str:
    mode = str(value)
    if mode not in ANIMA_NATIVE_REFERENCE_ROUTING_MODES:
        raise ValueError(
            f"Invalid native reference routing mode from {source}: {mode!r}; "
            f"expected one of {ANIMA_NATIVE_REFERENCE_ROUTING_MODES}."
        )
    return mode


def _validate_native_reference_routing_float(
    value: Any,
    *,
    name: str,
    source: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid native reference {name} from {source}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid native reference {name} from {source}: expected a finite value, got {value!r}")
    if name in {
        "routing_alpha",
        "index_alpha",
        "attribute_alpha",
        "context_alpha",
    } and not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"Invalid native reference {name} from {source}: expected [0, 1], got {parsed!r}"
        )
    if name == "router_temperature" and parsed <= 0.0:
        raise ValueError(
            f"Invalid native reference router_temperature from {source}: expected a positive value, got {parsed!r}"
        )
    return parsed


def _validate_native_reference_routing_bool(value: Any, *, name: str, source: str) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid native reference {name} from {source}: expected a boolean, got {value!r}")


def _normalise_native_reference_routing_config(config: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    missing = [key for key in _ANIMA_NATIVE_REFERENCE_ROUTING_REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Incomplete native reference routing config from {source}: missing={missing}")
    return {
        "routing_mode": _validate_native_reference_routing_mode(config["routing_mode"], source=source),
        "routing_alpha": _validate_native_reference_routing_float(
            config["routing_alpha"], name="routing_alpha", source=source
        ),
        "router_temperature": _validate_native_reference_routing_float(
            config["router_temperature"], name="router_temperature", source=source
        ),
        # Early V4 checkpoints did not persist this runtime-only switch and
        # therefore evaluated with the model default (enabled). Keep those files
        # loadable while new saves duplicate the field in every metadata source.
        "router_null_enabled": _validate_native_reference_routing_bool(
            config.get("router_null_enabled", True),
            name="router_null_enabled",
            source=source,
        ),
    }


def _routing_configs_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return all(left[key] == right[key] for key in _ANIMA_NATIVE_REFERENCE_ROUTING_CONFIG_KEYS)


def _normalise_native_reference_attribute_config(
    config: Dict[str, Any],
    *,
    source: str,
    detected_index_dim: Optional[int] = None,
    detected_attribute_hidden_dim: Optional[int] = None,
) -> Dict[str, Any]:
    missing = [
        key for key in ANIMA_NATIVE_REFERENCE_ATTRIBUTE_CONFIG_KEYS if key not in config
    ]
    if missing:
        raise ValueError(
            f"Incomplete native reference attribute-router config from {source}: missing={missing}"
        )
    version = str(config["attribute_routing_version"])
    if version != ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTING_VERSION:
        raise ValueError(
            f"Unsupported attribute_routing_version from {source}: {version!r}."
        )
    schema_version = str(config["attribute_schema_version"])
    if schema_version != REFERENCE_ATTRIBUTE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported attribute_schema_version from {source}: {schema_version!r}."
        )
    raw_vocab = config["attribute_vocab"]
    if not isinstance(raw_vocab, (list, tuple)) or tuple(raw_vocab) != REFERENCE_ATTRIBUTE_VOCAB:
        raise ValueError(
            f"Attribute vocabulary from {source} does not match the frozen runtime vocabulary."
        )
    try:
        index_dim = int(config["index_dim"])
        hidden_dim = int(config["attribute_hidden_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid attribute-router dimensions from {source}.") from exc
    if index_dim <= 0 or index_dim % 2:
        raise ValueError(f"index_dim from {source} must be positive and even.")
    if hidden_dim <= 0:
        raise ValueError(f"attribute_hidden_dim from {source} must be positive.")
    if detected_index_dim is not None and index_dim != int(detected_index_dim):
        raise ValueError(
            f"Attribute-router index_dim metadata/state mismatch: {index_dim} != {detected_index_dim}."
        )
    if (
        detected_attribute_hidden_dim is not None
        and hidden_dim != int(detected_attribute_hidden_dim)
    ):
        raise ValueError(
            "Attribute-router hidden-dim metadata/state mismatch: "
            f"{hidden_dim} != {detected_attribute_hidden_dim}."
        )
    index_alpha = _validate_native_reference_routing_float(
        config["index_alpha"], name="index_alpha", source=source
    )
    attribute_alpha = _validate_native_reference_routing_float(
        config["attribute_alpha"], name="attribute_alpha", source=source
    )
    return {
        "attribute_routing_version": version,
        "attribute_schema_version": schema_version,
        "attribute_vocab": list(REFERENCE_ATTRIBUTE_VOCAB),
        "index_dim": index_dim,
        "attribute_hidden_dim": hidden_dim,
        "index_alpha": index_alpha,
        "attribute_alpha": attribute_alpha,
    }


def _attribute_configs_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in ANIMA_NATIVE_REFERENCE_ATTRIBUTE_CONFIG_KEYS
    )


def _validate_native_reference_context_state_shapes(
    shapes: Dict[str, tuple[int, ...]],
    *,
    source: str,
    expected_model_dim: Optional[int] = None,
    expected_router_dim: Optional[int] = None,
) -> int:
    expected = set(ANIMA_NATIVE_REFERENCE_CONTEXT_REQUIRED_STATE_KEYS)
    actual = set(shapes)
    if actual != expected:
        raise ValueError(
            f"Incomplete native context state from {source}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    visual = shapes[
        "native_reference_context_conditioner.instance_conditioner.visual_proj.weight"
    ]
    router = shapes[
        "native_reference_context_conditioner.instance_conditioner.router_proj.weight"
    ]
    role_visual = shapes[
        "native_reference_context_conditioner.role_visual_proj.weight"
    ]
    role_router = shapes[
        "native_reference_context_conditioner.role_router_proj.weight"
    ]
    if any(len(shape) != 2 for shape in (visual, router, role_visual, role_router)):
        raise ValueError(f"Native context tensors from {source} must all be matrices.")
    model_dim, index_dim = visual
    router_dim, router_index_dim = router
    if index_dim != router_index_dim or index_dim <= 0 or index_dim % 2:
        raise ValueError(f"Invalid native context index width from {source}: {index_dim}.")
    if role_visual != (model_dim, 2) or role_router != (router_dim, 2):
        raise ValueError(
            f"Native context role projection shapes from {source} are inconsistent."
        )
    if expected_model_dim is not None and model_dim != int(expected_model_dim):
        raise ValueError(
            f"Native context model width from {source} is {model_dim}, expected {expected_model_dim}."
        )
    if expected_router_dim is not None and router_dim != int(expected_router_dim):
        raise ValueError(
            f"Native context router width from {source} is {router_dim}, expected {expected_router_dim}."
        )
    return int(index_dim)


def _normalise_native_reference_context_config(
    config: Dict[str, Any],
    *,
    source: str,
    detected_index_dim: Optional[int] = None,
) -> Dict[str, Any]:
    missing = [
        key for key in ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_KEYS if key not in config
    ]
    if missing:
        raise ValueError(
            f"Incomplete native reference context config from {source}: missing={missing}."
        )
    version = str(config["context_routing_version"])
    if version != ANIMA_NATIVE_REFERENCE_CONTEXT_ROUTING_VERSION:
        raise ValueError(
            f"Unsupported context_routing_version from {source}: {version!r}."
        )
    try:
        index_dim = int(config["context_index_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid context_index_dim from {source}.") from exc
    if index_dim <= 0 or index_dim % 2:
        raise ValueError(f"context_index_dim from {source} must be positive and even.")
    if detected_index_dim is not None and index_dim != int(detected_index_dim):
        raise ValueError(
            f"Native context index metadata/state mismatch: {index_dim} != {detected_index_dim}."
        )
    alpha = _validate_native_reference_routing_float(
        config["context_alpha"], name="context_alpha", source=source
    )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("A published native_context_v1 checkpoint requires context_alpha in (0,1].")
    return {
        "context_routing_version": version,
        "context_index_dim": index_dim,
        "context_alpha": alpha,
    }


def _context_configs_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return all(left[key] == right[key] for key in ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_KEYS)


def _normalise_maskless_edit_scope_config(
    config: Dict[str, Any],
    *,
    source: str,
    detected_hidden_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate the complete checkpoint-integrated implicit-scope contract."""

    missing = [key for key in ANIMA_MASKLESS_EDIT_SCOPE_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Incomplete maskless edit-scope config from {source}: missing={missing}")
    version = str(config["scope_version"])
    if version != ANIMA_MASKLESS_EDIT_SCOPE_VERSION:
        raise ValueError(
            f"Unsupported maskless edit-scope version from {source}: {version!r}; "
            f"expected {ANIMA_MASKLESS_EDIT_SCOPE_VERSION!r}."
        )
    try:
        hidden_dim = int(config["scope_hidden_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid scope_hidden_dim from {source}: {config['scope_hidden_dim']!r}"
        ) from exc
    if hidden_dim <= 0:
        raise ValueError(f"scope_hidden_dim from {source} must be positive, got {hidden_dim}.")
    if detected_hidden_dim is not None and hidden_dim != int(detected_hidden_dim):
        raise ValueError(
            "Maskless edit-scope metadata/state mismatch: "
            f"scope_hidden_dim={hidden_dim}, tensor_hidden_dim={detected_hidden_dim}."
        )
    try:
        probability = float(config["scope_initial_change_probability"])
        alpha = float(config["scope_alpha"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid maskless edit-scope numeric config from {source}.") from exc
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(
            "scope_initial_change_probability must be finite and strictly in (0,1), "
            f"got {probability!r}."
        )
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"scope_alpha must be finite and in [0,1], got {alpha!r}.")
    return {
        "scope_version": version,
        "scope_hidden_dim": hidden_dim,
        "scope_initial_change_probability": probability,
        "scope_alpha": alpha,
    }


def inspect_anima_checkpoint(model_files: Union[str, List[str]]) -> Dict[str, Any]:
    """Inspect Anima checkpoint structure without loading its large tensors.

    V1/V2 native-reference checkpoints predate text-slot routing metadata and
    are valid legacy checkpoints.  A V4 checkpoint is different: the narrow
    clause-pooler state marker and all duplicated routing metadata must agree.
    This fail-closed rule prevents silently loading trained router tensors into
    a legacy graph (or constructing an untrained router for a mislabeled file).
    """

    metadata: Dict[str, str] = {}
    native_keys: List[str] = []
    text_slot_router_keys: List[str] = []
    attribute_router_keys: List[str] = []
    context_router_keys: List[str] = []
    edit_scope_keys: List[str] = []
    edit_scope_hidden_dim: Optional[int] = None
    slot_rows: Optional[int] = None
    state_architecture: Optional[str] = None
    state_rank: Optional[int] = None
    state_router_dim: Optional[int] = None
    state_head_count: Optional[int] = None
    state_spatial_rows: Optional[int] = None
    attribute_router_shapes: Dict[str, tuple[int, ...]] = {}
    context_router_shapes: Dict[str, tuple[int, ...]] = {}
    for checkpoint_file in _as_checkpoint_file_list(model_files):
        if not str(checkpoint_file).lower().endswith(".safetensors"):
            continue
        with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
            metadata.update(handle.metadata() or {})
            for raw_key in handle.keys():
                key = _normalise_anima_state_key(raw_key)
                if is_native_reference_state_key(key):
                    native_keys.append(key)
                if is_native_reference_text_slot_router_state_key(key):
                    text_slot_router_keys.append(key)
                if key.startswith(ANIMA_NATIVE_REFERENCE_INDEX_STATE_PREFIX) or key.startswith(
                    ANIMA_NATIVE_REFERENCE_ATTRIBUTE_STATE_PREFIX
                ):
                    attribute_router_keys.append(key)
                    attribute_router_shapes[key] = tuple(
                        int(value) for value in handle.get_slice(raw_key).get_shape()
                    )
                if key.startswith(ANIMA_NATIVE_REFERENCE_CONTEXT_STATE_PREFIX):
                    context_router_keys.append(key)
                    context_router_shapes[key] = tuple(
                        int(value) for value in handle.get_slice(raw_key).get_shape()
                    )
                if key.startswith(ANIMA_MASKLESS_EDIT_SCOPE_STATE_PREFIX):
                    edit_scope_keys.append(key)
                    if key == "edit_scope_predictor.output.weight":
                        edit_scope_hidden_dim = int(handle.get_slice(raw_key).get_shape()[1])
                if key == "reference_slot_embeddings.weight":
                    slot_rows = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.k_down.weight"):
                    state_architecture = "v2"
                    state_rank = int(handle.get_slice(raw_key).get_shape()[0])
                elif key.endswith(".native_reference_attn.k_proj.weight") and state_architecture is None:
                    state_architecture = "v1"
                if key.endswith(".native_reference_attn.prompt_key.weight"):
                    state_router_dim = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.head_gate.weight"):
                    state_head_count = int(handle.get_slice(raw_key).get_shape()[0])
                if key.endswith(".native_reference_attn.target_spatial.weight"):
                    state_spatial_rows = int(handle.get_slice(raw_key).get_shape()[0])

    raw_config = metadata.get(ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY)
    metadata_native_config = _parse_metadata_dict(raw_config, ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY)
    config: Dict[str, Any] = dict(metadata_native_config)

    metadata_enabled = _metadata_bool(metadata.get(ANIMA_NATIVE_REFERENCE_METADATA_KEY, False))
    enabled = bool(native_keys) or metadata_enabled
    if slot_rows is not None:
        config["max_reference_images"] = slot_rows
    if state_architecture is not None:
        # Tensor layout remains authoritative for historical V1/V2 metadata.
        config["architecture"] = state_architecture
    if state_rank is not None:
        config["rank"] = state_rank
    if state_router_dim is not None:
        config["router_dim"] = state_router_dim
    if state_head_count and state_spatial_rows and state_spatial_rows % state_head_count == 0:
        config["gate_dim"] = state_spatial_rows // state_head_count
    config.setdefault("architecture", "v1")
    config.setdefault("rank", 64)
    config.setdefault("max_reference_images", 8)
    config.setdefault("router_dim", 128)
    config.setdefault("gate_dim", 8)
    config.setdefault("initial_gate", 0.5 if config["architecture"] == "v2" else 0.01)
    detected_version = 2 if config["architecture"] == "v2" else ANIMA_NATIVE_REFERENCE_VERSION

    has_edit_scope_state = bool(edit_scope_keys)
    scope_metadata_fields = {
        key: metadata_native_config[key]
        for key in ANIMA_MASKLESS_EDIT_SCOPE_CONFIG_KEYS
        if key in metadata_native_config
    }
    if has_edit_scope_state:
        expected_scope_keys = set(ANIMA_MASKLESS_EDIT_SCOPE_REQUIRED_STATE_KEYS)
        actual_scope_keys = set(edit_scope_keys)
        if actual_scope_keys != expected_scope_keys:
            raise ValueError(
                "Incomplete maskless edit-scope checkpoint state: "
                f"missing={sorted(expected_scope_keys - actual_scope_keys)}, "
                f"unexpected={sorted(actual_scope_keys - expected_scope_keys)}."
            )
        if not enabled or config.get("architecture") != "v2":
            raise ValueError(
                "Maskless edit-scope state requires integrated native-reference architecture='v2'."
            )
        scope_config = _normalise_maskless_edit_scope_config(
            scope_metadata_fields,
            source=ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY,
            detected_hidden_dim=edit_scope_hidden_dim,
        )
        config.update(scope_config)
    elif scope_metadata_fields:
        raise ValueError(
            "Checkpoint metadata declares maskless edit scope but no edit_scope_predictor state is present."
        )
    else:
        scope_config = None

    native_routing_fields = {
        key: metadata_native_config[key]
        for key in _ANIMA_NATIVE_REFERENCE_ROUTING_CONFIG_KEYS
        if key in metadata_native_config
    }
    top_level_routing_fields = {}
    top_level_metadata_keys = {
        "routing_mode": ANIMA_NATIVE_REFERENCE_ROUTING_MODE_METADATA_KEY,
        "routing_alpha": ANIMA_NATIVE_REFERENCE_ROUTING_ALPHA_METADATA_KEY,
        "router_temperature": ANIMA_NATIVE_REFERENCE_ROUTER_TEMPERATURE_METADATA_KEY,
        "router_null_enabled": ANIMA_NATIVE_REFERENCE_ROUTER_NULL_ENABLED_METADATA_KEY,
    }
    for config_key, metadata_key in top_level_metadata_keys.items():
        if metadata_key in metadata:
            top_level_routing_fields[config_key] = metadata[metadata_key]

    router_config_metadata_present = ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY in metadata
    router_config_metadata = (
        _parse_metadata_dict(
            metadata.get(ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY),
            ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY,
        )
        if router_config_metadata_present
        else {}
    )

    routing_sources: List[tuple[str, Dict[str, Any]]] = []
    if native_routing_fields:
        routing_sources.append((ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY, native_routing_fields))
    if top_level_routing_fields:
        routing_sources.append(("top-level routing metadata", top_level_routing_fields))
    if router_config_metadata_present:
        routing_sources.append((ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY, router_config_metadata))

    normalised_routing_sources = [
        (source, _normalise_native_reference_routing_config(source_config, source=source))
        for source, source_config in routing_sources
    ]
    has_text_slot_router_state = bool(text_slot_router_keys)
    has_context_router_state = bool(context_router_keys)
    if has_text_slot_router_state or normalised_routing_sources:
        present_sources = {source for source, _ in normalised_routing_sources}
        required_sources = {
            ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY,
            "top-level routing metadata",
            ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY,
        }
        missing_sources = sorted(required_sources - present_sources)
        if missing_sources:
            raise ValueError(
                "Incomplete native reference routing metadata: "
                f"missing complete source(s) {missing_sources}; V4 checkpoints fail closed."
            )

        routing_config = normalised_routing_sources[0][1]
        for source, candidate in normalised_routing_sources[1:]:
            if not _routing_configs_equal(routing_config, candidate):
                raise ValueError(
                    "Inconsistent native reference routing metadata: "
                    f"{normalised_routing_sources[0][0]}={routing_config!r}, {source}={candidate!r}."
                )

        if has_text_slot_router_state:
            expected_mode = (
                "native_context_v1"
                if has_context_router_state
                else "competitive_text_slot_v1"
            )
            if routing_config["routing_mode"] != expected_mode:
                raise ValueError(
                    "Checkpoint competitive/context state disagrees with metadata routing_mode: "
                    f"expected {expected_mode!r}, got {routing_config['routing_mode']!r}."
                )
            if metadata_native_config.get("architecture") != "v2":
                raise ValueError(
                    "V4 text-slot router checkpoint metadata must explicitly declare native-reference architecture='v2'."
                )
            if state_architecture is not None and state_architecture != "v2":
                raise ValueError("V4 text-slot router keys are inconsistent with the checkpoint native-reference tensor layout.")
        elif routing_config["routing_mode"] in {
            "competitive_text_slot_v1",
            "native_context_v1",
        }:
            raise ValueError(
                "Checkpoint metadata declares a competitive routing mode but no "
                f"{ANIMA_NATIVE_REFERENCE_TEXT_SLOT_ROUTER_STATE_MARKER!r} state key is present."
            )
        elif routing_config["routing_alpha"] != 0.0:
            raise ValueError("Legacy routing metadata must use routing_alpha=0.0 because no text-slot router is present.")
    else:
        # Published V1/V2 checkpoints intentionally have no routing metadata.
        routing_config = {
            "routing_mode": "legacy",
            "routing_alpha": 0.0,
            "router_temperature": 1.0,
            "router_null_enabled": True,
        }

    if normalised_routing_sources:
        config.update(routing_config)

    nested_context_fields = {
        key: metadata_native_config[key]
        for key in ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_KEYS
        if key in metadata_native_config
    }
    separate_context_metadata_present = (
        ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY in metadata
    )
    separate_context_fields = (
        _parse_metadata_dict(
            metadata.get(ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY),
            ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY,
        )
        if separate_context_metadata_present
        else {}
    )
    if has_context_router_state:
        if not enabled or config.get("architecture") != "v2":
            raise ValueError(
                "Native context state requires integrated native-reference architecture='v2'."
            )
        if not has_text_slot_router_state or routing_config["routing_mode"] != "native_context_v1":
            raise ValueError(
                "Native context state requires the integrated competitive visual carrier."
            )
        state_context_index_dim = _validate_native_reference_context_state_shapes(
            context_router_shapes,
            source="checkpoint",
            expected_router_dim=int(config["router_dim"]),
        )
        if not nested_context_fields or not separate_context_metadata_present:
            raise ValueError(
                "Native context checkpoints require complete config in both native-reference "
                "metadata and anima_native_reference_context_config."
            )
        nested_context_config = _normalise_native_reference_context_config(
            nested_context_fields,
            source=ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY,
            detected_index_dim=state_context_index_dim,
        )
        separate_context_config = _normalise_native_reference_context_config(
            separate_context_fields,
            source=ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY,
            detected_index_dim=state_context_index_dim,
        )
        if not _context_configs_equal(
            nested_context_config, separate_context_config
        ):
            raise ValueError("Inconsistent duplicated native context checkpoint metadata.")
        context_config = nested_context_config
        config.update(context_config)
    elif nested_context_fields or separate_context_metadata_present:
        raise ValueError(
            "Checkpoint metadata declares native context routing, but its state is absent."
        )
    else:
        context_config = None

    has_attribute_router_state = bool(attribute_router_keys)
    if has_attribute_router_state and has_context_router_state:
        raise ValueError(
            "A final checkpoint cannot contain both retired V6 attribute state and V7 native context state."
        )
    nested_attribute_fields = {
        key: metadata_native_config[key]
        for key in ANIMA_NATIVE_REFERENCE_ATTRIBUTE_CONFIG_KEYS
        if key in metadata_native_config
    }
    separate_attribute_metadata_present = (
        ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY in metadata
    )
    separate_attribute_fields = (
        _parse_metadata_dict(
            metadata.get(ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY),
            ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY,
        )
        if separate_attribute_metadata_present
        else {}
    )
    if has_attribute_router_state:
        expected_attribute_keys = set(
            ANIMA_NATIVE_REFERENCE_ATTRIBUTE_REQUIRED_STATE_KEYS
        )
        actual_attribute_keys = set(attribute_router_keys)
        if actual_attribute_keys != expected_attribute_keys:
            raise ValueError(
                "Incomplete native reference attribute-router checkpoint state: "
                f"missing={sorted(expected_attribute_keys - actual_attribute_keys)}, "
                f"unexpected={sorted(actual_attribute_keys - expected_attribute_keys)}."
            )
        if not enabled or config.get("architecture") != "v2":
            raise ValueError(
                "Attribute-router state requires integrated native-reference architecture='v2'."
            )
        if not has_text_slot_router_state or routing_config["routing_mode"] != "competitive_text_slot_v1":
            raise ValueError(
                "Attribute-router state requires the integrated competitive text-slot carrier."
            )
        state_index_dim, state_attribute_hidden_dim = (
            _validate_native_reference_attribute_state_shapes(
                attribute_router_shapes,
                source="checkpoint",
                expected_router_dim=int(config["router_dim"]),
            )
        )
        if not nested_attribute_fields or not separate_attribute_metadata_present:
            raise ValueError(
                "Attribute-router checkpoints require complete config in both native-reference "
                "metadata and anima_native_reference_attribute_router_config."
            )
        nested_attribute_config = _normalise_native_reference_attribute_config(
            nested_attribute_fields,
            source=ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY,
            detected_index_dim=state_index_dim,
            detected_attribute_hidden_dim=state_attribute_hidden_dim,
        )
        separate_attribute_config = _normalise_native_reference_attribute_config(
            separate_attribute_fields,
            source=ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY,
            detected_index_dim=state_index_dim,
            detected_attribute_hidden_dim=state_attribute_hidden_dim,
        )
        if not _attribute_configs_equal(
            nested_attribute_config, separate_attribute_config
        ):
            raise ValueError("Inconsistent duplicated attribute-router checkpoint metadata.")
        attribute_config = nested_attribute_config
        config.update(attribute_config)
    elif nested_attribute_fields or separate_attribute_metadata_present:
        raise ValueError(
            "Checkpoint metadata declares native reference attribute routing, but its state is absent."
        )
    else:
        attribute_config = None

    public_config = dict(config)
    # Preserve the exact public schema of historical V1 checkpoints. V2 adds
    # architecture/rank only when those fields are semantically required.
    if public_config["architecture"] == "v1":
        public_config.pop("architecture", None)
        public_config.pop("rank", None)

    return {
        "native_reference_conditioning": enabled,
        "native_reference_version": int(metadata.get("anima_native_reference_version", detected_version)),
        "native_reference_config": public_config,
        "native_reference_keys": tuple(native_keys),
        "native_reference_text_slot_router": has_text_slot_router_state,
        "native_reference_text_slot_router_keys": tuple(text_slot_router_keys),
        "native_reference_routing_mode": routing_config["routing_mode"],
        "native_reference_routing_alpha": routing_config["routing_alpha"],
        "native_reference_router_temperature": routing_config["router_temperature"],
        "native_reference_router_null_enabled": routing_config["router_null_enabled"],
        "native_reference_router_config": dict(routing_config),
        "native_reference_attribute_routing": has_attribute_router_state,
        "native_reference_attribute_router_keys": tuple(attribute_router_keys),
        "native_reference_attribute_router_config": (
            None if attribute_config is None else dict(attribute_config)
        ),
        "native_reference_context_routing": has_context_router_state,
        "native_reference_context_keys": tuple(context_router_keys),
        "native_reference_context_config": (
            None if context_config is None else dict(context_config)
        ),
        "maskless_edit_scope": has_edit_scope_state,
        "maskless_edit_scope_keys": tuple(edit_scope_keys),
        "maskless_edit_scope_config": None if scope_config is None else dict(scope_config),
        "metadata": metadata,
    }


def add_anima_architecture_metadata(
    metadata: Optional[Dict[str, Any]],
    state_dict: Dict[str, torch.Tensor],
    native_reference_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return safetensors metadata describing integrated native-ref weights.

    Historical V1/V2 saves retain their historical metadata schema unless a
    routing config is explicitly supplied.  V4 router tensors, however, may
    never be saved without a complete, duplicated routing contract.
    """

    result = {str(k): str(v) for k, v in (metadata or {}).items()}
    native_keys = [key for key in state_dict if is_native_reference_state_key(key)]
    if not native_keys:
        return result

    explicit_config = dict(native_reference_config or {})
    config: Dict[str, Any] = {
        "architecture": "v1",
        "rank": 64,
        "max_reference_images": 8,
        "router_dim": 128,
        "gate_dim": 8,
        "initial_gate": 0.01,
    }
    config.update(explicit_config)
    has_text_slot_router_state = any(is_native_reference_text_slot_router_state_key(key) for key in state_dict)
    edit_scope_keys = {
        _normalise_anima_state_key(key)
        for key in state_dict
        if _normalise_anima_state_key(key).startswith(ANIMA_MASKLESS_EDIT_SCOPE_STATE_PREFIX)
    }
    attribute_router_keys = {
        _normalise_anima_state_key(key)
        for key in state_dict
        if _normalise_anima_state_key(key).startswith(
            ANIMA_NATIVE_REFERENCE_INDEX_STATE_PREFIX
        )
        or _normalise_anima_state_key(key).startswith(
            ANIMA_NATIVE_REFERENCE_ATTRIBUTE_STATE_PREFIX
        )
    }
    context_router_keys = {
        _normalise_anima_state_key(key)
        for key in state_dict
        if _normalise_anima_state_key(key).startswith(
            ANIMA_NATIVE_REFERENCE_CONTEXT_STATE_PREFIX
        )
    }
    context_explicit_fields = {
        key: explicit_config[key]
        for key in ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_KEYS
        if key in explicit_config
    }
    context_config: Optional[Dict[str, Any]] = None
    if context_router_keys:
        if attribute_router_keys:
            raise ValueError(
                "Refusing to save both retired V6 attribute state and V7 native context state."
            )
        context_shapes = {
            _normalise_anima_state_key(key): tuple(int(value) for value in tensor.shape)
            for key, tensor in state_dict.items()
            if _normalise_anima_state_key(key) in context_router_keys
        }
        state_context_index_dim = _validate_native_reference_context_state_shapes(
            context_shapes,
            source="state_dict",
            expected_router_dim=int(config["router_dim"]),
        )
        context_config = _normalise_native_reference_context_config(
            context_explicit_fields,
            source="native_reference_config",
            detected_index_dim=state_context_index_dim,
        )
        config.update(context_config)
    elif context_explicit_fields:
        raise ValueError(
            "Native context config was supplied, but the state dict has no context-conditioner keys."
        )
    attribute_explicit_fields = {
        key: explicit_config[key]
        for key in ANIMA_NATIVE_REFERENCE_ATTRIBUTE_CONFIG_KEYS
        if key in explicit_config
    }
    attribute_config: Optional[Dict[str, Any]] = None
    if attribute_router_keys:
        attribute_shapes = {
            _normalise_anima_state_key(key): tuple(int(value) for value in tensor.shape)
            for key, tensor in state_dict.items()
            if _normalise_anima_state_key(key) in attribute_router_keys
        }
        state_index_dim, state_attribute_hidden_dim = (
            _validate_native_reference_attribute_state_shapes(
                attribute_shapes,
                source="state_dict",
                expected_router_dim=int(config["router_dim"]),
            )
        )
        attribute_config = _normalise_native_reference_attribute_config(
            attribute_explicit_fields,
            source="native_reference_config",
            detected_index_dim=state_index_dim,
            detected_attribute_hidden_dim=state_attribute_hidden_dim,
        )
        config.update(attribute_config)
    elif attribute_explicit_fields:
        raise ValueError(
            "Attribute-router config was supplied, but the state dict has no attribute-router keys."
        )
    scope_explicit_fields = {
        key: explicit_config[key]
        for key in ANIMA_MASKLESS_EDIT_SCOPE_CONFIG_KEYS
        if key in explicit_config
    }
    if edit_scope_keys:
        expected_scope_keys = set(ANIMA_MASKLESS_EDIT_SCOPE_REQUIRED_STATE_KEYS)
        if edit_scope_keys != expected_scope_keys:
            raise ValueError(
                "Refusing to save incomplete maskless edit-scope state: "
                f"missing={sorted(expected_scope_keys - edit_scope_keys)}, "
                f"unexpected={sorted(edit_scope_keys - expected_scope_keys)}."
            )
        output_weight = next(
            tensor
            for key, tensor in state_dict.items()
            if _normalise_anima_state_key(key) == "edit_scope_predictor.output.weight"
        )
        scope_config = _normalise_maskless_edit_scope_config(
            scope_explicit_fields,
            source="native_reference_config",
            detected_hidden_dim=int(output_weight.shape[1]),
        )
        config.update(scope_config)
    elif scope_explicit_fields:
        raise ValueError(
            "Maskless edit-scope config was supplied, but the state dict has no edit_scope_predictor keys."
        )
    for key, tensor in state_dict.items():
        normalised_key = _normalise_anima_state_key(key)
        if normalised_key == "reference_slot_embeddings.weight":
            config["max_reference_images"] = int(tensor.shape[0])
        if normalised_key.endswith(".native_reference_attn.k_down.weight"):
            config["architecture"] = "v2"
            config["rank"] = int(tensor.shape[0])
    if edit_scope_keys and config.get("architecture") != "v2":
        raise ValueError("Maskless edit-scope state can only be saved with architecture='v2'.")
    if config["architecture"] == "v2" and "initial_gate" not in explicit_config:
        config["initial_gate"] = 0.5

    explicit_routing_fields = {
        key: explicit_config[key]
        for key in _ANIMA_NATIVE_REFERENCE_ROUTING_CONFIG_KEYS
        if key in explicit_config
    }
    stale_top_level_routing_metadata = any(
        key in result
        for key in (
            ANIMA_NATIVE_REFERENCE_ROUTING_MODE_METADATA_KEY,
            ANIMA_NATIVE_REFERENCE_ROUTING_ALPHA_METADATA_KEY,
            ANIMA_NATIVE_REFERENCE_ROUTER_TEMPERATURE_METADATA_KEY,
            ANIMA_NATIVE_REFERENCE_ROUTER_NULL_ENABLED_METADATA_KEY,
            ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY,
        )
    )
    routing_config: Optional[Dict[str, Any]] = None
    if has_text_slot_router_state or explicit_routing_fields:
        routing_config = _normalise_native_reference_routing_config(
            explicit_routing_fields,
            source="native_reference_config",
        )
        if has_text_slot_router_state:
            expected_mode = (
                "native_context_v1" if context_config is not None else "competitive_text_slot_v1"
            )
            if routing_config["routing_mode"] != expected_mode:
                raise ValueError(
                    "Competitive visual-router state/config mismatch: "
                    f"expected routing_mode={expected_mode!r}."
                )
            if config.get("architecture") != "v2":
                raise ValueError("V4 text-slot router state must be saved with native-reference architecture='v2'.")
        elif routing_config["routing_mode"] in {
            "competitive_text_slot_v1",
            "native_context_v1",
        }:
            raise ValueError(
                "A competitive routing mode was provided, but the checkpoint state has no "
                f"{ANIMA_NATIVE_REFERENCE_TEXT_SLOT_ROUTER_STATE_MARKER!r} key."
            )
        elif routing_config["routing_alpha"] != 0.0:
            raise ValueError("Legacy routing metadata must use routing_alpha=0.0.")
    elif stale_top_level_routing_metadata:
        raise ValueError(
            "Routing metadata was supplied without an explicit native_reference_config; refusing to save a "
            "potentially mislabeled checkpoint."
        )

    if routing_config is not None:
        config.update(routing_config)
    if attribute_config is not None:
        if config.get("architecture") != "v2":
            raise ValueError(
                "Attribute-router state can only be saved with native-reference architecture='v2'."
            )
        if not has_text_slot_router_state or routing_config is None or routing_config[
            "routing_mode"
        ] != "competitive_text_slot_v1":
            raise ValueError(
                "Attribute-router state requires a saved competitive text-slot routing graph."
            )
    if context_config is not None:
        if config.get("architecture") != "v2":
            raise ValueError(
                "Native context state can only be saved with native-reference architecture='v2'."
            )
        if not has_text_slot_router_state or routing_config is None or routing_config[
            "routing_mode"
        ] != "native_context_v1":
            raise ValueError(
                "Native context state requires a saved competitive visual-routing graph."
            )
    if config["architecture"] == "v1":
        config.pop("architecture", None)
        config.pop("rank", None)

    result[ANIMA_NATIVE_REFERENCE_METADATA_KEY] = "true"
    version = (
        4
        if context_config is not None
        else 3
        if attribute_config is not None
        else (2 if config.get("architecture") == "v2" else ANIMA_NATIVE_REFERENCE_VERSION)
    )
    result["anima_native_reference_version"] = str(version)
    result[ANIMA_NATIVE_REFERENCE_CONFIG_METADATA_KEY] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    if routing_config is not None:
        result[ANIMA_NATIVE_REFERENCE_ROUTING_MODE_METADATA_KEY] = routing_config["routing_mode"]
        result[ANIMA_NATIVE_REFERENCE_ROUTING_ALPHA_METADATA_KEY] = str(routing_config["routing_alpha"])
        result[ANIMA_NATIVE_REFERENCE_ROUTER_TEMPERATURE_METADATA_KEY] = str(
            routing_config["router_temperature"]
        )
        result[ANIMA_NATIVE_REFERENCE_ROUTER_NULL_ENABLED_METADATA_KEY] = str(
            routing_config["router_null_enabled"]
        ).lower()
        result[ANIMA_NATIVE_REFERENCE_ROUTER_CONFIG_METADATA_KEY] = json.dumps(
            routing_config,
            sort_keys=True,
            separators=(",", ":"),
        )
    if attribute_config is not None:
        result[ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY] = json.dumps(
            attribute_config,
            sort_keys=True,
            separators=(",", ":"),
        )
    if context_config is not None:
        # Training metadata may originate from the V6 warm-start file.  The
        # migrated state tree no longer owns that classifier, so never carry
        # its duplicated config forward into the V7 artifact.
        result.pop(ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTER_CONFIG_METADATA_KEY, None)
        result[ANIMA_NATIVE_REFERENCE_CONTEXT_CONFIG_METADATA_KEY] = json.dumps(
            context_config,
            sort_keys=True,
            separators=(",", ":"),
        )
    # Explicitly documents that numbered slots carry no fixed semantic role.
    result["anima_reference_slot_semantics"] = (
        "structural_image_instance_not_semantic_role"
        if context_config is not None
        else "ordered_image_index"
    )
    if context_config is not None:
        result["anima_prompt_contract"] = "exact_user_prompt_no_hidden_reference_rewrite"
        result["anima_reference_routing_inputs"] = "full_prompt+visual_candidate+target+instance+source_role"
    result["anima_checkpoint_layout"] = "integrated_single_checkpoint"
    result["anima_external_adapter_required"] = "false"
    # NetworkTrainer normally emits an adapter ModelSpec because it owns a
    # network object.  Native-reference publication instead contains the full
    # frozen base plus the integrated route, so remove that misleading suffix.
    architecture = result.get("modelspec.architecture")
    if architecture and architecture.endswith("/lora"):
        result["modelspec.architecture"] = architecture[: -len("/lora")]
    return result


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    enable_ip_adapter: bool = False,
    ip_adapter_scale: float = 1.0,
    ip_adapter_feature_dim: Optional[int] = None,
    ip_adapter_num_tokens: int = 4,
    ip_adapter_linear_adapter: bool = False,
    ip_adapter_mlp_adapter: bool = False,
    ip_adapter_norm_linear_adapter: bool = False,
    ip_adapter_omni_adapter: bool = False,
    enable_native_reference_conditioning: Optional[bool] = None,
    native_reference_max_images: Optional[int] = None,
    native_reference_router_dim: Optional[int] = None,
    native_reference_gate_dim: Optional[int] = None,
    native_reference_initial_gate: Optional[float] = None,
    native_reference_architecture: Optional[str] = None,
    native_reference_rank: Optional[int] = None,
    native_reference_scale: float = 1.0,
    native_reference_routing_mode: Optional[str] = None,
    native_reference_routing_alpha: Optional[float] = None,
    native_reference_router_temperature: Optional[float] = None,
    enable_native_reference_attribute_routing: Optional[bool] = None,
    native_reference_index_dim: Optional[int] = None,
    native_reference_attribute_hidden_dim: Optional[int] = None,
    native_reference_index_alpha: Optional[float] = None,
    native_reference_attribute_alpha: Optional[float] = None,
    enable_native_reference_context_routing: Optional[bool] = None,
    native_reference_context_index_dim: Optional[int] = None,
    native_reference_context_alpha: Optional[float] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # The Comfy V7 overlay is intentionally narrower than the training loader:
    # it accepts one audited, integrated BF16 checkpoint and has no merging or
    # quantization branch.  Keeping those branches in the private vendor tree
    # would reintroduce adapter/LoRA/FP8 dependencies that the public runtime
    # explicitly forbids.
    if fp8_scaled:
        raise ValueError("The Native Context V7 runtime does not support FP8 loading.")
    if lora_weights_list is not None or lora_multipliers is not None:
        raise ValueError("The Native Context V7 runtime does not load or merge LoRA weights.")
    if enable_ip_adapter:
        raise ValueError("The Native Context V7 runtime does not enable IP-Adapter.")
    if dit_weight_dtype is not torch.bfloat16:
        raise ValueError("The Native Context V7 runtime requires BF16 DiT weights.")
    if not os.path.isfile(dit_path) or not dit_path.lower().endswith(".safetensors"):
        raise ValueError("The Native Context V7 runtime requires one .safetensors checkpoint file.")

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    checkpoint_info = inspect_anima_checkpoint(dit_path)
    checkpoint_has_native_reference = checkpoint_info["native_reference_conditioning"]
    if enable_native_reference_conditioning is False and checkpoint_has_native_reference:
        raise ValueError(
            "Checkpoint contains integrated native reference weights, but "
            "enable_native_reference_conditioning=False was requested."
        )
    use_native_reference = (
        checkpoint_has_native_reference
        if enable_native_reference_conditioning is None
        else enable_native_reference_conditioning
    )
    native_config = dict(checkpoint_info["native_reference_config"])
    native_config.setdefault("architecture", "v1")
    native_config.setdefault("rank", 64)
    explicit_native_config = {
        "max_reference_images": native_reference_max_images,
        "router_dim": native_reference_router_dim,
        "gate_dim": native_reference_gate_dim,
        "initial_gate": native_reference_initial_gate,
        "architecture": native_reference_architecture,
        "rank": native_reference_rank,
    }
    if checkpoint_has_native_reference:
        for config_key, explicit_value in explicit_native_config.items():
            if explicit_value is None:
                continue
            detected_value = native_config[config_key]
            if explicit_value != detected_value:
                raise ValueError(
                    f"Native reference checkpoint config {config_key}={detected_value!r}, "
                    f"but {explicit_value!r} was requested."
                )
    native_config.update({key: value for key, value in explicit_native_config.items() if value is not None})

    checkpoint_has_text_slot_router = bool(checkpoint_info["native_reference_text_slot_router"])
    checkpoint_routing_mode = str(checkpoint_info["native_reference_routing_mode"])
    checkpoint_routing_alpha = float(checkpoint_info["native_reference_routing_alpha"])
    checkpoint_router_temperature = float(checkpoint_info["native_reference_router_temperature"])
    checkpoint_router_null_enabled = bool(checkpoint_info["native_reference_router_null_enabled"])
    checkpoint_has_edit_scope = bool(checkpoint_info.get("maskless_edit_scope", False))
    checkpoint_edit_scope_config = checkpoint_info.get("maskless_edit_scope_config")
    checkpoint_has_attribute_routing = bool(
        checkpoint_info.get("native_reference_attribute_routing", False)
    )
    checkpoint_attribute_config = checkpoint_info.get(
        "native_reference_attribute_router_config"
    )
    checkpoint_has_context_routing = bool(
        checkpoint_info.get("native_reference_context_routing", False)
    )
    checkpoint_context_config = checkpoint_info.get(
        "native_reference_context_config"
    )
    if checkpoint_has_edit_scope and not use_native_reference:
        raise ValueError("A maskless edit-scope checkpoint requires native reference conditioning.")
    if checkpoint_has_attribute_routing and not use_native_reference:
        raise ValueError("An attribute-router checkpoint requires native reference conditioning.")
    if checkpoint_has_context_routing and not use_native_reference:
        raise ValueError("A native-context checkpoint requires native reference conditioning.")
    if (
        enable_native_reference_attribute_routing is False
        and checkpoint_has_attribute_routing
        and native_reference_routing_mode != "native_context_v1"
    ):
        raise ValueError(
            "Checkpoint contains integrated attribute-router state, but it was explicitly disabled."
        )

    if native_reference_routing_mode is None:
        requested_routing_mode = checkpoint_routing_mode
    else:
        requested_routing_mode = _validate_native_reference_routing_mode(
            native_reference_routing_mode,
            source="load_anima_model(native_reference_routing_mode)",
        )
    requested_routing_alpha = (
        checkpoint_routing_alpha
        if native_reference_routing_alpha is None
        else _validate_native_reference_routing_float(
            native_reference_routing_alpha,
            name="routing_alpha",
            source="load_anima_model(native_reference_routing_alpha)",
        )
    )
    requested_router_temperature = (
        checkpoint_router_temperature
        if native_reference_router_temperature is None
        else _validate_native_reference_routing_float(
            native_reference_router_temperature,
            name="router_temperature",
            source="load_anima_model(native_reference_router_temperature)",
        )
    )
    use_attribute_routing = (
        checkpoint_has_attribute_routing
        if enable_native_reference_attribute_routing is None
        else bool(enable_native_reference_attribute_routing)
    )
    if checkpoint_has_attribute_routing:
        if checkpoint_attribute_config is None:
            raise RuntimeError("Checkpoint attribute-router state has no validated config.")
        attribute_config = dict(checkpoint_attribute_config)
        for key, explicit_value in (
            ("index_dim", native_reference_index_dim),
            ("attribute_hidden_dim", native_reference_attribute_hidden_dim),
        ):
            if explicit_value is not None and int(explicit_value) != int(attribute_config[key]):
                raise ValueError(
                    f"Attribute-router checkpoint config {key}={attribute_config[key]!r}, "
                    f"but {explicit_value!r} was requested."
                )
    else:
        attribute_config = {
            "attribute_routing_version": ANIMA_NATIVE_REFERENCE_ATTRIBUTE_ROUTING_VERSION,
            "attribute_schema_version": REFERENCE_ATTRIBUTE_SCHEMA_VERSION,
            "attribute_vocab": list(REFERENCE_ATTRIBUTE_VOCAB),
            "index_dim": 64 if native_reference_index_dim is None else int(native_reference_index_dim),
            "attribute_hidden_dim": (
                256
                if native_reference_attribute_hidden_dim is None
                else int(native_reference_attribute_hidden_dim)
            ),
            "index_alpha": 1.0 if enable_native_reference_attribute_routing is True else 0.0,
            "attribute_alpha": 1.0 if enable_native_reference_attribute_routing is True else 0.0,
        }
    requested_index_alpha = (
        float(attribute_config["index_alpha"])
        if native_reference_index_alpha is None
        else _validate_native_reference_routing_float(
            native_reference_index_alpha,
            name="index_alpha",
            source="load_anima_model(native_reference_index_alpha)",
        )
    )
    requested_attribute_alpha = (
        float(attribute_config["attribute_alpha"])
        if native_reference_attribute_alpha is None
        else _validate_native_reference_routing_float(
            native_reference_attribute_alpha,
            name="attribute_alpha",
            source="load_anima_model(native_reference_attribute_alpha)",
        )
    )

    use_context_routing = (
        checkpoint_has_context_routing
        if enable_native_reference_context_routing is None
        else bool(enable_native_reference_context_routing)
    )
    # Selecting the graph mode is itself an explicit opt-in for a V4/V6 warm
    # start; callers do not need a redundant second boolean.
    if requested_routing_mode == "native_context_v1":
        use_context_routing = True
    if checkpoint_has_context_routing:
        if checkpoint_context_config is None:
            raise RuntimeError("Checkpoint native-context state has no validated config.")
        context_config = dict(checkpoint_context_config)
        if (
            native_reference_context_index_dim is not None
            and int(native_reference_context_index_dim)
            != int(context_config["context_index_dim"])
        ):
            raise ValueError(
                "Native-context checkpoint index width is authoritative and differs from the request."
            )
    else:
        inherited_index_dim = (
            int(attribute_config["index_dim"])
            if checkpoint_has_attribute_routing
            else 64
        )
        context_config = {
            "context_routing_version": ANIMA_NATIVE_REFERENCE_CONTEXT_ROUTING_VERSION,
            "context_index_dim": (
                inherited_index_dim
                if native_reference_context_index_dim is None
                else int(native_reference_context_index_dim)
            ),
            "context_alpha": 1.0,
        }
    requested_context_alpha = (
        float(context_config["context_alpha"])
        if native_reference_context_alpha is None
        else _validate_native_reference_routing_float(
            native_reference_context_alpha,
            name="context_alpha",
            source="load_anima_model(native_reference_context_alpha)",
        )
    )
    if use_context_routing and not 0.0 < requested_context_alpha <= 1.0:
        raise ValueError("native_context_v1 requires context_alpha in (0,1].")

    if (
        checkpoint_has_text_slot_router
        and requested_routing_mode != checkpoint_routing_mode
        and requested_routing_mode != "native_context_v1"
    ):
        raise ValueError(
            f"Text-slot router checkpoint routing_mode={checkpoint_routing_mode!r}, "
            f"but {requested_routing_mode!r} was requested. A V4 graph cannot be loaded as legacy."
        )
    if requested_routing_mode == "legacy":
        if native_reference_routing_alpha is not None and requested_routing_alpha != 0.0:
            raise ValueError("native_reference_routing_alpha requires a competitive routing mode.")
        if native_reference_router_temperature is not None and requested_router_temperature != 1.0:
            raise ValueError("native_reference_router_temperature requires a competitive routing mode.")
    elif not use_native_reference:
        raise ValueError("Competitive routing requires native reference conditioning to be enabled.")
    if use_attribute_routing:
        if not use_native_reference or requested_routing_mode not in {
            "competitive_text_slot_v1",
            "native_context_v1",
        }:
            raise ValueError(
                "Native reference attribute routing requires competitive_text_slot_v1."
            )
    if use_context_routing and (
        not use_native_reference or requested_routing_mode != "native_context_v1"
    ):
        raise ValueError(
            "Native reference context routing requires routing_mode='native_context_v1'."
        )

    # The graph mode is checkpoint-authoritative.  A published V2 checkpoint
    # explicitly upgraded through the CLI is still constructed as legacy here,
    # loaded exactly, and receives freshly initialized router modules only after
    # assign=True has populated every checkpoint tensor.
    checkpoint_graph_routing_mode = (
        "native_context_v1"
        if checkpoint_has_context_routing
        else "competitive_text_slot_v1"
        if checkpoint_has_text_slot_router
        else "legacy"
    )

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
        # For an integrated checkpoint, build these modules in the empty-weight
        # context so state_dict assignment can populate them without a second
        # 2B-model allocation.  For a base checkpoint used to start new native
        # training, enable them after the base weights have been assigned.
        "native_reference_conditioning": bool(checkpoint_has_native_reference),
        "native_reference_max_images": int(native_config["max_reference_images"]),
        "native_reference_router_dim": int(native_config["router_dim"]),
        "native_reference_gate_dim": int(native_config["gate_dim"]),
        "native_reference_initial_gate": float(native_config["initial_gate"]),
        "native_reference_architecture": str(native_config["architecture"]),
        "native_reference_rank": int(native_config["rank"]),
        "native_reference_routing_mode": checkpoint_graph_routing_mode,
        "native_reference_routing_alpha": (
            checkpoint_routing_alpha if checkpoint_has_text_slot_router else 0.0
        ),
        "native_reference_router_temperature": (
            checkpoint_router_temperature if checkpoint_has_text_slot_router else 1.0
        ),
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if checkpoint_has_edit_scope:
            if checkpoint_edit_scope_config is None:
                raise RuntimeError("Checkpoint edit-scope state has no validated configuration.")
            model.enable_maskless_edit_scope(
                hidden_dim=int(checkpoint_edit_scope_config["scope_hidden_dim"]),
                initial_change_probability=float(
                    checkpoint_edit_scope_config["scope_initial_change_probability"]
                ),
                initial_mix_scale=float(checkpoint_edit_scope_config["scope_alpha"]),
            )
        if checkpoint_has_attribute_routing:
            model.enable_native_reference_attribute_routing(
                index_dim=int(attribute_config["index_dim"]),
                attribute_hidden_dim=int(attribute_config["attribute_hidden_dim"]),
                index_alpha=requested_index_alpha,
                attribute_alpha=requested_attribute_alpha,
            )
        if checkpoint_has_context_routing:
            model.enable_native_reference_context_routing(
                index_dim=int(context_config["context_index_dim"]),
                context_alpha=requested_context_alpha,
            )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # Load the one already-validated integrated checkpoint.  ``assign=True``
    # below moves these tensors into the meta-constructed graph without a
    # second initialized 2B-parameter allocation.
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    sd = load_file(dit_path, device=str(loading_device))
    for key in tuple(sd):
        if key.startswith("net."):
            sd[key[len("net.") :]] = sd.pop(key)

    # Some published V4 checkpoints contain a historical pointer head. The RF
    # runtime keeps it inert, but materializes the exact parameter names/shapes
    # after the text-router graph is assigned so those files remain loadable.
    pointer_state = {
        key: sd.pop(key)
        for key in list(sd)
        if ".native_reference_attn.pointer_head." in key
    }

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if use_native_reference and not checkpoint_has_native_reference:
        # New training from a legacy base checkpoint: create initialized native
        # modules only after assigning the base state, otherwise their missing
        # tensors would remain on the meta device.
        model.enable_native_reference_conditioning(
            max_reference_images=int(native_config["max_reference_images"]),
            router_dim=int(native_config["router_dim"]),
            gate_dim=int(native_config["gate_dim"]),
            initial_gate=float(native_config["initial_gate"]),
            architecture=str(native_config["architecture"]),
            rank=int(native_config["rank"]),
        )
    if use_native_reference and requested_routing_mode in {
        "competitive_text_slot_v1",
        "native_context_v1",
    }:
        if not checkpoint_has_text_slot_router:
            # Exact V2/base assignment is complete at this point.  New V4
            # tensors are therefore real initialized tensors, never meta tensors,
            # and cannot perturb the published checkpoint load path.
            model.enable_native_reference_text_slot_router(
                routing_alpha=requested_routing_alpha,
                router_temperature=requested_router_temperature,
            )
        else:
            # Alpha and temperature are explicit runtime controls.  The graph and
            # learned router weights remain checkpoint-authoritative, while CLI
            # values may intentionally override the published sampling defaults.
            if native_reference_routing_alpha is not None:
                model.set_native_reference_routing_alpha(requested_routing_alpha)
            if native_reference_router_temperature is not None:
                model.set_native_reference_router_temperature(requested_router_temperature)
        # Unlike alpha/temperature, null-route enablement has no CLI override.
        # Restore it from integrated checkpoint metadata or evaluation silently
        # changes graph semantics (the constructor defaults this switch to True).
        if checkpoint_has_text_slot_router:
            if not hasattr(model, "set_native_reference_router_null_enabled"):
                raise RuntimeError("Loaded V4 model lacks the router-null runtime setter.")
            model.set_native_reference_router_null_enabled(checkpoint_router_null_enabled)

    if use_attribute_routing:
        if not checkpoint_has_attribute_routing:
            # V4 warm-start upgrade: materialize only after all checkpoint
            # tensors have left the meta graph.
            model.enable_native_reference_attribute_routing(
                index_dim=int(attribute_config["index_dim"]),
                attribute_hidden_dim=int(attribute_config["attribute_hidden_dim"]),
                index_alpha=requested_index_alpha,
                attribute_alpha=requested_attribute_alpha,
            )
        elif (
            native_reference_index_alpha is not None
            or native_reference_attribute_alpha is not None
        ):
            model.set_native_reference_attribute_routing_alphas(
                index_alpha=requested_index_alpha,
                attribute_alpha=requested_attribute_alpha,
            )

    if use_context_routing:
        if not checkpoint_has_context_routing:
            model.enable_native_reference_context_routing(
                index_dim=int(context_config["context_index_dim"]),
                context_alpha=requested_context_alpha,
                migrate_v6_index=checkpoint_has_attribute_routing,
                drop_v6_attribute_router=checkpoint_has_attribute_routing,
            )
        else:
            model.set_native_reference_context_alpha(requested_context_alpha)
        # A V6 warm start is migrated in place above.  The migration can copy
        # the structural index projection, but it deliberately deregisters the
        # retired semantic attribute router.  Keep subsequent reporting tied
        # to the actual graph rather than the pre-migration request variable.
        use_attribute_routing = bool(
            getattr(model, "native_reference_attribute_routing_enabled", False)
        )

    if pointer_state and requested_routing_mode == "native_context_v1":
        # V7 deliberately retires the historical pointer head.  It never
        # participated in the RF forward path, and carrying it into the new
        # checkpoint would preserve dead V4 compatibility state indefinitely.
        # The tensors were removed from ``sd`` before strict assignment, so a
        # V6 -> V7 warm start can safely discard them here.
        logger.info(
            "Discarding %s inert legacy pointer tensors during native_context_v1 migration.",
            len(pointer_state),
        )
        pointer_state = {}

    if pointer_state:
        if not use_native_reference or requested_routing_mode not in {
            "competitive_text_slot_v1",
            "native_context_v1",
        }:
            raise RuntimeError("Checkpoint contains legacy pointer tensors without a competitive V4 graph.")
        if not hasattr(model, "materialize_native_reference_legacy_pointer_state"):
            raise RuntimeError("Runtime cannot materialize integrated legacy pointer state.")
        model.materialize_native_reference_legacy_pointer_state(max_slots=2)
        expected_pointer = {
            key
            for key in model.state_dict()
            if ".native_reference_attn.pointer_head." in key
        }
        if set(pointer_state) != expected_pointer:
            missing_pointer = sorted(expected_pointer - set(pointer_state))
            unexpected_pointer = sorted(set(pointer_state) - expected_pointer)
            raise RuntimeError(
                "Incomplete legacy V4 pointer state: "
                f"missing={missing_pointer[:8]}, unexpected={unexpected_pointer[:8]}"
            )
        pointer_load = model.load_state_dict(pointer_state, strict=False, assign=True)
        if pointer_load.unexpected_keys:
            raise RuntimeError(f"Unexpected legacy pointer keys: {pointer_load.unexpected_keys[:8]}")

    if use_native_reference:
        model.set_native_reference_scale(native_reference_scale)
        logger.info(
            "Native reference conditioning enabled: architecture=%s, rank=%s, max_images=%s, "
            "router_dim=%s, gate_dim=%s, scale=%s, routing_mode=%s, routing_alpha=%s, "
            "router_temperature=%s, router_null_enabled=%s",
            native_config["architecture"],
            native_config["rank"],
            native_config["max_reference_images"],
            native_config["router_dim"],
            native_config["gate_dim"],
            native_reference_scale,
            requested_routing_mode,
            requested_routing_alpha,
            requested_router_temperature,
            checkpoint_router_null_enabled,
        )
        if checkpoint_has_edit_scope:
            logger.info(
                "Integrated maskless edit scope enabled: version=%s, hidden_dim=%s, alpha=%s",
                checkpoint_edit_scope_config["scope_version"],
                checkpoint_edit_scope_config["scope_hidden_dim"],
                checkpoint_edit_scope_config["scope_alpha"],
            )
        if use_attribute_routing:
            logger.info(
                "Integrated attribute routing enabled: version=%s, index_dim=%s, "
                "attribute_hidden_dim=%s, index_alpha=%s, attribute_alpha=%s",
                attribute_config["attribute_routing_version"],
                attribute_config["index_dim"],
                attribute_config["attribute_hidden_dim"],
                requested_index_alpha,
                requested_attribute_alpha,
            )
        if use_context_routing:
            logger.info(
                "Integrated native context enabled: version=%s, index_dim=%s, "
                "context_alpha=%s, prompt_contract=exact/full/no-rewrite",
                context_config["context_routing_version"],
                context_config["context_index_dim"],
                requested_context_alpha,
            )

    if enable_ip_adapter:
        model.enable_ip_adapter(
            ip_adapter_scale,
            feature_dim=ip_adapter_feature_dim,
            num_feature_tokens=ip_adapter_num_tokens,
            linear_adapter=ip_adapter_linear_adapter,
            mlp_adapter=ip_adapter_mlp_adapter,
            norm_linear_adapter=ip_adapter_norm_linear_adapter,
            omni_adapter=ip_adapter_omni_adapter,
        )
        ip_adapter_dtype = dit_weight_dtype
        if model.visual_condition_adapter is not None:
            model.visual_condition_adapter.to(device=loading_device, dtype=ip_adapter_dtype)
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if lora_weights is not None or lora_multipliers is not None:
        raise ValueError("The Native Context V7 runtime does not load or merge text-encoder LoRA weights.")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            state_dict = load_file(qwen3_path, device="cpu")
        else:
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str,
    dit_state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, Any]],
    dtype: Optional[torch.dtype] = None,
    native_reference_config: Optional[Dict[str, Any]] = None,
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        metadata: Metadata dict to include in the safetensors file
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            # v = v.to(dtype)
            v = v.detach().clone().to("cpu").to(dtype)  # Reduce GPU memory usage during save
        prefixed_sd["net." + k] = v.contiguous()

    metadata = add_anima_architecture_metadata(metadata, dit_state_dict, native_reference_config)
    metadata["format"] = "pt"  # For compatibility with the official .safetensors file

    save_file(prefixed_sd, save_path, metadata=metadata)  # safetensors.save_file consumes a lot of memory, but Anima is small enough
    logger.info(f"Saved Anima model to {save_path}")
