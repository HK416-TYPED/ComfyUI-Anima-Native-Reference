from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

from PIL import Image, ImageOps
import pytest
from safetensors.torch import save_file
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import runtime as rtmod  # noqa: E402


def test_v2_v4_public_generate_defaults_do_not_cross_contaminate() -> None:
    import inspect

    v2 = inspect.signature(rtmod.AnimaNativeReferenceV2Runtime.generate).parameters
    v4 = inspect.signature(rtmod.AnimaNativeReferenceV4Runtime.generate).parameters
    assert v2["steps"].default == rtmod.DEFAULT_STEPS == 40
    assert v2["guidance_scale"].default == rtmod.DEFAULT_GUIDANCE_SCALE == 1.0
    assert v4["steps"].default == rtmod.V4_DEFAULT_STEPS == 30
    assert (
        v4["guidance_scale"].default
        == rtmod.V4_DEFAULT_GUIDANCE_SCALE
        == 3.5
    )


def _load_pure_vendor_module(module_leaf: str) -> ModuleType:
    name = f"_anima_v4_pure_test_{module_leaf}"
    sys.modules.pop(name, None)
    path = (
        PACKAGE_ROOT
        / "vendor"
        / "anima_edit"
        / "library"
        / f"{module_leaf}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BINDING = _load_pure_vendor_module("anima_reference_binding")
TEXT_CONDITIONING = _load_pure_vendor_module("anima_text_conditioning")


def _tiny_v4_shape_contract():
    shapes = {
        "reference_slot_embeddings.weight": (2, 3),
        "reference_type_embedding": (3,),
    }
    for block_index in range(28):
        stem = f"blocks.{block_index}.native_reference_attn"
        for extra in range(33):
            shapes[f"{stem}.test_native_{extra:02d}"] = (1,)
    assert len(shapes) == rtmod.V4_EXPECTED_NATIVE_TENSORS
    return shapes


def _fake_v4_state_dict(shape_contract):
    tensors = {
        f"net.{key}": torch.zeros(shape, dtype=torch.bfloat16)
        for key, shape in shape_contract.items()
    }
    for index in range(rtmod.V4_EXPECTED_TOTAL_TENSORS - len(tensors)):
        tensors[f"net.test_base_tensor_{index:04d}"] = torch.zeros(
            (1,), dtype=torch.bfloat16
        )
    assert len(tensors) == rtmod.V4_EXPECTED_TOTAL_TENSORS
    assert (
        sum(rtmod._is_native_reference_key(key) for key in tensors)
        == rtmod.V4_EXPECTED_NATIVE_TENSORS
    )
    return tensors


@pytest.fixture(scope="module")
def fake_v4(tmp_path_factory):
    path = tmp_path_factory.mktemp("v4") / "fake-v4.safetensors"
    metadata = dict(rtmod.V4_REQUIRED_METADATA)
    metadata["anima_native_reference_config"] = json.dumps(
        dict(rtmod.V4_EXPECTED_CONFIG), sort_keys=True, separators=(",", ":")
    )
    metadata["anima_native_reference_router_config"] = json.dumps(
        {
            "router_null_enabled": True,
            "router_temperature": 1.0,
            "routing_alpha": 1.0,
            "routing_mode": "competitive_text_slot_v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    shapes = _tiny_v4_shape_contract()
    save_file(_fake_v4_state_dict(shapes), str(path), metadata=metadata)
    return path, shapes


def test_final_v4_shape_contract_is_complete_and_release_specific():
    shapes = rtmod._v4_required_native_shapes()
    assert len(shapes) == 926
    assert shapes["reference_slot_embeddings.weight"] == (2, 2048)
    assert (
        shapes[
            "blocks.27.native_reference_attn.clause_pooler.key_proj.weight"
        ]
        == (64, 1024)
    )
    assert (
        shapes[
            "blocks.27.native_reference_attn.competitive_router.target_query.weight"
        ]
        == (64, 2048)
    )
    assert (
        shapes["blocks.27.native_reference_attn.pointer_head.weight"] == (2, 64)
    )


def test_final_v4_validator_accepts_exact_layout(fake_v4, monkeypatch):
    path, shapes = fake_v4
    monkeypatch.setattr(rtmod, "V4_EXPECTED_FILE_SIZE", path.stat().st_size)
    monkeypatch.setattr(rtmod, "_v4_required_native_shapes", lambda: dict(shapes))
    info = rtmod.validate_v4_checkpoint(path)
    assert info.config == dict(rtmod.V4_EXPECTED_CONFIG)
    assert info.tensor_count == 1614
    assert info.native_tensor_count == 926
    assert info.dtype_counts == {"BF16": 1614}
    assert info.metadata["anima_native_reference_version"] == "2"
    assert info.metadata["anima_native_reference_routing_mode"] == (
        "competitive_text_slot_v1"
    )


def test_final_v4_validator_rejects_v2_or_tampered_router(fake_v4, monkeypatch):
    path, shapes = fake_v4
    monkeypatch.setattr(rtmod, "V4_EXPECTED_FILE_SIZE", path.stat().st_size)
    monkeypatch.setattr(rtmod, "_v4_required_native_shapes", lambda: dict(shapes))
    required = dict(rtmod.V4_REQUIRED_METADATA)
    required["anima_native_reference_routing_alpha"] = "0.0"
    monkeypatch.setattr(rtmod, "V4_REQUIRED_METADATA", required)
    with pytest.raises(rtmod.CheckpointValidationError, match="routing_alpha"):
        rtmod.validate_v4_checkpoint(path)


def test_private_vendor_import_exposes_all_v4_contract_modules_in_subprocess():
    code = """
import runtime
import torch
m = runtime.load_runtime_modules()
assert hasattr(m.anima_models.Anima, 'set_native_reference_fixed12ref_vectorized')
assert hasattr(m.strategy_anima, 'preprocess_anima_reference_image')
assert hasattr(m.strategy_anima.AnimaTokenizeStrategy, 'tokenize_reference_clause_masks')
assert hasattr(m.anima_text_conditioning, 'prepare_anima_prompt_conditioning')
assert hasattr(m.anima_reference_binding, 'parse_reference_bindings')
qwen = m.anima_utils.load_qwen3_tokenizer('tokenizer-config-only.safetensors')
t5 = m.anima_utils.load_t5_tokenizer(None)
strategy = m.strategy_anima.AnimaTokenizeStrategy(
    qwen3_tokenizer=qwen,
    t5_tokenizer=t5,
    qwen3_max_length=512,
    t5_max_length=512,
)
cases = [
    ('Use Image 1 as the character identity reference. Generate a new illustration.', [[0]], [True, False]),
    ('Use Image 2 as the character identity reference. Generate a new illustration.', [[1]], [False, True]),
    ('Use the pose and composition from Image 1; use the character appearance from Image 2.', [[0, 1]], [True, True]),
]
for prompt, slots, occupied in cases:
    batch = strategy.tokenize_reference_clause_masks(
        [prompt], slots, max_slots=2, require_canonical=True
    )
    assert batch.clause_masks.shape == (1, 2, 512)
    assert batch.clause_masks.dtype is torch.bool
    assert batch.clause_masks.any(dim=-1).tolist() == [occupied]
    assert bool(batch.binding_valid.all())
print('V4_VENDOR_IMPORT_OK')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "V4_VENDOR_IMPORT_OK" in completed.stdout


class CharacterTokenizeStrategy:
    """Small deterministic tokenizer implementing the V4 strategy surface."""

    def __init__(self, *, qwen_length: int = 512, t5_length: int = 512):
        self.qwen_length = qwen_length
        self.t5_length = t5_length
        self.calls = []

    @staticmethod
    def _encode(text: str, length: int):
        active = min(len(text), length)
        ids = [index + 1 for index in range(active)] + [0] * (length - active)
        mask = [1] * active + [0] * (length - active)
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([mask], dtype=torch.long),
        )

    def tokenize(self, text):
        assert isinstance(text, str)
        self.calls.append(text)
        qwen_ids, qwen_mask = self._encode(text, self.qwen_length)
        t5_ids, t5_mask = self._encode(text, self.t5_length)
        return [qwen_ids, qwen_mask, t5_ids, t5_mask]

    def tokenize_reference_clause_masks(
        self,
        texts,
        expected_slot_ids_per_sample,
        *,
        max_slots,
        require_canonical,
    ):
        if isinstance(texts, str):
            texts = [texts]
        masks = torch.zeros(
            len(texts), max_slots, self.qwen_length, dtype=torch.bool
        )
        valid = []
        statuses = []
        reasons = []
        canonicals = []
        for row, (text, slots) in enumerate(
            zip(texts, expected_slot_ids_per_sample, strict=True)
        ):
            parsed = BINDING.parse_reference_bindings(
                text, expected_slot_ids=tuple(slots)
            )
            canonical = parsed.canonical_instruction
            row_valid = bool(parsed.binding_valid)
            reason = parsed.reason
            if require_canonical and canonical != text:
                row_valid = False
                reason = "non_canonical_instruction"
            if row_valid:
                active = min(len(text), self.qwen_length)
                offsets = [
                    (index, index + 1) for index in range(active)
                ] + [(0, 0)] * (self.qwen_length - active)
                attention = [1] * active + [0] * (self.qwen_length - active)
                try:
                    by_slot = BINDING.build_slot_clause_token_masks(
                        parsed,
                        offsets,
                        attention_mask=attention,
                    )
                    for slot_id, values in by_slot.items():
                        masks[row, slot_id] = torch.tensor(values, dtype=torch.bool)
                except BINDING.SpanAlignmentError:
                    row_valid = False
                    reason = "truncated_reference_clause"
                    masks[row].zero_()
            valid.append(row_valid)
            statuses.append(parsed.status.value)
            reasons.append(reason)
            canonicals.append(canonical)
        return SimpleNamespace(
            clause_masks=masks,
            binding_valid=torch.tensor(valid, dtype=torch.bool),
            statuses=tuple(statuses),
            reasons=tuple(reasons),
            canonical_instructions=tuple(canonicals),
        )


class FakeTextEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(self, *, input_ids, attention_mask):
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 8)
        return SimpleNamespace(last_hidden_state=hidden.to(torch.bfloat16))


class FakeV4Anima:
    patch_spatial = 2
    use_llm_adapter = True
    native_reference_conditioning_enabled = True
    native_reference_architecture = "v2"
    native_reference_max_images = 2
    native_reference_routing_alpha = 1.0
    native_reference_routing_mode = "competitive_text_slot_v1"

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.calls = []
        self.scales = []
        self.fixed12 = []

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def set_native_reference_fixed12ref_vectorized(self, enabled):
        self.fixed12.append(bool(enabled))

    def set_native_reference_scale(self, scale):
        self.scales.append(float(scale))

    def _preprocess_text_embeds(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("V4 runtime must not adapt/discard raw Qwen early")

    def __call__(self, latents, timestep, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(latents)


class FakeVAE:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.encode_calls = 0

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def encode_pixels_to_latents(self, pixels):
        self.encode_calls += 1
        height = max(1, pixels.shape[-2] // 8)
        width = max(1, pixels.shape[-1] // 8)
        marker = pixels[:, :1].mean().to(torch.bfloat16)
        return marker.view(1, 1, 1, 1).expand(1, 16, height, width).contiguous()

    def decode_to_pixels(self, latent):
        height = int(latent.shape[-2]) * 8
        width = int(latent.shape[-1]) * 8
        return torch.zeros((1, 3, height, width), dtype=latent.dtype)


class FakeHunyuan:
    @staticmethod
    def get_timesteps_sigmas(steps, flow_shift, device):
        return torch.linspace(1000, 1, steps, device=device), torch.linspace(
            1, 0, steps + 1, device=device
        )

    @staticmethod
    def step(latents, noise_pred, sigmas, index):
        return latents


class SpyPreprocess:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        image,
        *,
        max_area,
        multiple_of,
        target_size_hw=None,
        flipped=False,
    ):
        self.calls.append(
            {
                "max_area": max_area,
                "multiple_of": multiple_of,
                "target_size_hw": target_size_hw,
                "flipped": flipped,
            }
        )
        image = image.convert("RGB")
        if target_size_hw is not None:
            target_h, target_w = target_size_hw
            return ImageOps.fit(
                image,
                (target_w, target_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        width = (image.width // multiple_of) * multiple_of
        height = (image.height // multiple_of) * multiple_of
        if width <= 0 or height <= 0:
            raise ValueError("too small")
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))


def _bare_v4_runtime(tmp_path: Path):
    obj = object.__new__(rtmod.AnimaNativeReferenceV4Runtime)
    obj.lock = threading.RLock()
    obj._closed = False
    obj.device = torch.device("cpu")
    obj.dtype = torch.bfloat16
    obj.offload_mode = "balanced"
    obj._anima = FakeV4Anima()
    obj._vae = FakeVAE()
    obj._text_encoder = FakeTextEncoder()
    obj._qwen_tokenizer = object()
    obj._t5_tokenizer = object()
    obj._tokenize_strategy = CharacterTokenizeStrategy()
    preprocess = SpyPreprocess()
    obj._modules = SimpleNamespace(
        qwen_image_autoencoder_kl=SimpleNamespace(SCALE_FACTOR=8),
        anima_models=SimpleNamespace(Anima=SimpleNamespace(LATENT_CHANNELS=16)),
        hunyuan_image_utils=FakeHunyuan,
        strategy_anima=SimpleNamespace(
            preprocess_anima_reference_image=preprocess
        ),
        anima_text_conditioning=TEXT_CONDITIONING,
    )
    vae_file = tmp_path / "vae.safetensors"
    vae_file.write_bytes(b"fake-vae")
    obj.vae_fingerprint = rtmod.FileFingerprint.from_path(vae_file)
    obj._prompt_cache = rtmod._BoundedLRU(8)
    obj._reference_cache = rtmod._BoundedLRU(8)
    obj._prompt_hits = 0
    obj._prompt_misses = 0
    obj._reference_hits = 0
    obj._reference_misses = 0
    obj._model_loads = 1
    return obj, preprocess


def test_raw_qwen_and_neutral_t5_four_tensor_cache_preserves_unicode(tmp_path):
    runtime, _ = _bare_v4_runtime(tmp_path)
    prompt = "Use Image 1 as 角色参考。"
    encoded = runtime._encode_text_uncached_v4(prompt)
    assert encoded.raw_qwen_context.shape == (1, 512, 8)
    assert encoded.source_attention_mask.shape == (1, 512)
    assert encoded.target_input_ids.shape == (1, 512)
    assert encoded.target_attention_mask.shape == (1, 512)
    assert runtime._tokenize_strategy.calls == [prompt]
    assert all(tensor.device.type == "cpu" for tensor in encoded.as_sequence())
    assert not hasattr(runtime._anima, "last_preprocessed")

    first, negative = runtime._get_raw_text_encodings(prompt, "")
    second, _ = runtime._get_raw_text_encodings(prompt, "")
    assert first is second
    assert first.raw_qwen_context.shape[-1] == 8
    assert negative.raw_qwen_context.shape[-1] == 8
    stats = runtime.cache_statistics()
    assert stats.prompt_misses == 2
    assert stats.prompt_hits == 2


@pytest.mark.parametrize(
    ("prompt", "slots", "occupied"),
    [
        (
            "Use Image 1 as the character identity reference. "
            "Generate a new illustration.",
            (0,),
            (True, False),
        ),
        (
            "Use Image 2 as the character identity reference. "
            "Generate a new illustration.",
            (1,),
            (False, True),
        ),
        (
            "Use the pose and composition from Image 1; "
            "use the character appearance from Image 2.",
            (0, 1),
            (True, True),
        ),
    ],
)
def test_canonical_clause_masks_cover_single_slot0_slot1_and_dual(
    tmp_path, prompt, slots, occupied
):
    runtime, _ = _bare_v4_runtime(tmp_path)
    positive, negative = runtime._prepare_text_conditionings(
        prompt, "", reference_slot_ids=slots
    )
    masks = positive.reference_clause_masks
    assert masks.dtype is torch.bool
    assert masks.shape == (1, 2, 512)
    assert tuple(bool(masks[0, slot].any()) for slot in range(2)) == occupied
    assert negative.reference_clause_masks is None


@pytest.mark.parametrize(
    ("prompt", "slots"),
    [
        ("Use both refs.", (0, 1)),
        ("Use Image 1.", (0, 1)),
        ("  Use Image 1.", (0,)),
        ("Use Image 2.", (0,)),
    ],
)
def test_noncanonical_ambiguous_missing_or_wrong_slot_fails_closed(
    tmp_path, prompt, slots
):
    runtime, _ = _bare_v4_runtime(tmp_path)
    with pytest.raises(rtmod.InputValidationError, match="Invalid V4 reference prompt"):
        runtime._prepare_text_conditionings(
            prompt, "", reference_slot_ids=slots
        )
    assert runtime._anima.calls == []
    assert runtime._vae.encode_calls == 0


def test_cfg_negative_keeps_negative_standard_text_but_shares_positive_router(
    tmp_path,
):
    runtime, _ = _bare_v4_runtime(tmp_path)
    prompt = (
        "Use the pose and composition from Image 1; "
        "use the character appearance from Image 2."
    )
    positive, negative = runtime._prepare_text_conditionings(
        prompt, "bad anatomy", reference_slot_ids=(0, 1)
    )
    refs = [[torch.zeros(1, 16, 2, 2), torch.ones(1, 16, 2, 2)]]
    result = runtime._denoise_v4(
        positive_conditioning=positive,
        negative_conditioning=negative,
        reference_latents=refs,
        reference_slot_ids=(0, 1),
        width=16,
        height=16,
        seed=123,
        steps=2,
        guidance_scale=3.5,
        flow_shift=5.0,
        progress_callback=None,
        interrupt_callback=None,
    )
    assert result.shape == (1, 16, 1, 2, 2)
    assert len(runtime._anima.calls) == 4
    conditional, unconditional = runtime._anima.calls[:2]
    assert not torch.equal(conditional["context"], unconditional["context"])
    assert not torch.equal(
        conditional["target_input_ids"], unconditional["target_input_ids"]
    )
    assert torch.equal(
        conditional["raw_qwen_context"], unconditional["raw_qwen_context"]
    )
    assert torch.equal(
        conditional["router_source_attention_mask"],
        unconditional["router_source_attention_mask"],
    )
    assert torch.equal(
        conditional["reference_clause_masks"],
        unconditional["reference_clause_masks"],
    )
    assert conditional["reference_slot_ids"] == [[0, 1]]
    assert conditional["reference_latents"] is refs
    assert conditional["use_ip_adapter"] is False
    assert conditional["ip_adapter_latents"] is None
    assert conditional["ip_adapter_embeds"] is None

    runtime._anima.calls.clear()
    runtime._denoise_v4(
        positive_conditioning=positive,
        negative_conditioning=negative,
        reference_latents=refs,
        reference_slot_ids=(0, 1),
        width=16,
        height=16,
        seed=123,
        steps=2,
        guidance_scale=1.0,
        flow_shift=5.0,
        progress_callback=None,
        interrupt_callback=None,
    )
    assert len(runtime._anima.calls) == 2


def test_single_slot1_carrier_has_one_physical_latent_and_independent_geometry(
    tmp_path,
):
    runtime, preprocess = _bare_v4_runtime(tmp_path)
    ref = torch.zeros((1, 35, 51, 3), dtype=torch.float32)
    output = runtime.generate(
        [ref],
        [1],
        (
            "Use Image 2 as the character identity reference. "
            "Generate a new illustration."
        ),
        width=32,
        height=32,
        steps=1,
        preprocess_mode="independent_reference",
    )
    assert output.shape == (1, 32, 32, 3)
    assert preprocess.calls[-1]["target_size_hw"] is None
    assert runtime._vae.encode_calls == 1
    call = runtime._anima.calls[-1]
    assert call["reference_slot_ids"] == [[1]]
    assert len(call["reference_latents"]) == 1
    assert len(call["reference_latents"][0]) == 1


def test_single_slot0_edit_geometry_matches_output_and_cache_key_is_geometry_safe(
    tmp_path,
):
    runtime, preprocess = _bare_v4_runtime(tmp_path)
    ref = torch.zeros((1, 35, 51, 3), dtype=torch.float32)
    prompt = (
        "Use Image 1 as the source image reference. "
        "Generate a new illustration."
    )
    output = runtime.generate(
        [ref],
        [0],
        prompt,
        width=48,
        height=32,
        steps=1,
        preprocess_mode="match_output_edit",
    )
    assert output.shape == (1, 32, 48, 3)
    assert preprocess.calls[-1]["target_size_hw"] == (32, 48)
    assert runtime._anima.calls[-1]["reference_slot_ids"] == [[0]]
    assert len(runtime._anima.calls[-1]["reference_latents"][0]) == 1
    first_encode_count = runtime._vae.encode_calls

    runtime.generate(
        [ref],
        [0],
        prompt,
        width=48,
        height=32,
        steps=1,
        preprocess_mode="match_output_edit",
    )
    assert runtime._vae.encode_calls == first_encode_count

    runtime.generate(
        [ref],
        [0],
        prompt,
        width=64,
        height=32,
        steps=1,
        preprocess_mode="match_output_edit",
    )
    assert runtime._vae.encode_calls == first_encode_count + 1
    assert preprocess.calls[-1]["target_size_hw"] == (32, 64)


def test_reference_request_rejects_fake_second_duplicate_or_dual_edit_mode():
    image = torch.zeros((1, 16, 16, 3))
    with pytest.raises(rtmod.InputValidationError, match="one or two"):
        rtmod.AnimaNativeReferenceV4Runtime._normalise_reference_request(
            [], [], preprocess_mode="independent_reference"
        )
    with pytest.raises(rtmod.InputValidationError, match="unique"):
        rtmod.AnimaNativeReferenceV4Runtime._normalise_reference_request(
            [image, image], [0, 0], preprocess_mode="independent_reference"
        )
    with pytest.raises(rtmod.InputValidationError, match="one-reference edit"):
        rtmod.AnimaNativeReferenceV4Runtime._normalise_reference_request(
            [image, image], [0, 1], preprocess_mode="match_output_edit"
        )
