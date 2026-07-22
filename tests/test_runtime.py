from __future__ import annotations

import ast
import gc
import json
from pathlib import Path
import sys
import threading
import time
import warnings
from types import ModuleType, SimpleNamespace
import weakref

import pytest
from safetensors.torch import save_file
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import runtime as rtmod  # noqa: E402


def test_vendored_runtime_manifest_and_hashes_are_pinned():
    assert rtmod.validate_vendored_runtime() == rtmod.VENDORED_ANIMA_COMMIT
    manifest = json.loads(
        (PACKAGE_ROOT / "vendor" / "VENDOR_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["import_namespace"] == "_anima_native_ref_vendor"
    assert manifest["file_count"] == 107
    assert len(manifest["vendored_tree_sha256"]) == 64


def test_vendored_python_imports_use_only_private_namespace():
    vendor_root = rtmod.default_runtime_root()
    internal_imports = []
    for path in vendor_root.rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                internal_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("_anima_native_ref_vendor.")
                )
                assert all(
                    alias.name.split(".", 1)[0] not in {"library", "networks"}
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("_anima_native_ref_vendor."):
                    internal_imports.append(node.module)
                assert node.module.split(".", 1)[0] not in {"library", "networks"}
    assert len(internal_imports) == 207


def test_loader_preserves_sys_path_and_foreign_generic_packages(tmp_path, monkeypatch):
    root = tmp_path / "vendored-runtime"
    required_files = (
        "__init__.py",
        "library/__init__.py",
        "library/anima_utils.py",
        "library/anima_models.py",
        "library/hunyuan_image_utils.py",
        "library/qwen_image_autoencoder_kl.py",
        "configs/qwen3_06b/config.json",
        "configs/t5_old/spiece.model",
        "networks/__init__.py",
        "networks/loha.py",
        "networks/lokr.py",
        "LICENSE.md",
    )
    for relative in required_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    private_prefix = rtmod._VENDOR_PACKAGE_NAME
    cache_key = str(root.resolve())
    for name in tuple(sys.modules):
        if name == private_prefix or name.startswith(f"{private_prefix}."):
            sys.modules.pop(name, None)
    rtmod._RUNTIME_MODULES_BY_ROOT.pop(cache_key, None)

    foreign_library = ModuleType("library")
    foreign_networks = ModuleType("networks")
    monkeypatch.setitem(sys.modules, "library", foreign_library)
    monkeypatch.setitem(sys.modules, "networks", foreign_networks)
    monkeypatch.setattr(rtmod, "validate_vendored_runtime", lambda runtime_root: "ok")

    imported_names = []

    def fake_import(name):
        assert name.startswith(f"{private_prefix}.")
        imported_names.append(name)
        relative = name.removeprefix(f"{private_prefix}.").replace(".", "/")
        module = ModuleType(name)
        module.__file__ = str(root / f"{relative}.py")
        sys.modules[name] = module
        return module

    monkeypatch.setattr(rtmod.importlib, "import_module", fake_import)
    original_sys_path = list(sys.path)
    try:
        first = rtmod.load_runtime_modules(root)
        second = rtmod.load_runtime_modules(root)
        assert first is second
        assert sys.path == original_sys_path
        assert sys.modules["library"] is foreign_library
        assert sys.modules["networks"] is foreign_networks
        assert all(name.startswith(f"{private_prefix}.") for name in imported_names)
        assert "library.anima_utils" not in sys.modules
        assert "networks.loha" not in sys.modules
    finally:
        rtmod._RUNTIME_MODULES_BY_ROOT.pop(cache_key, None)
        for name in tuple(sys.modules):
            if name == private_prefix or name.startswith(f"{private_prefix}."):
                sys.modules.pop(name, None)


def _fake_e180_state_dict():
    tensors = {
        "net.reference_slot_embeddings.weight": torch.zeros(
            (2, 2048), dtype=torch.bfloat16
        ),
        "net.reference_type_embedding": torch.zeros((2048,), dtype=torch.bfloat16),
    }
    for block in range(28):
        stem = f"net.blocks.{block}.native_reference_attn"
        tensors[f"{stem}.k_down.weight"] = torch.zeros((64, 2048), dtype=torch.bfloat16)
        tensors[f"{stem}.prompt_key.weight"] = torch.zeros(
            (64, 1024), dtype=torch.bfloat16
        )
        tensors[f"{stem}.head_gate.weight"] = torch.zeros(
            (16, 64), dtype=torch.bfloat16
        )
        tensors[f"{stem}.target_spatial.weight"] = torch.zeros(
            (64, 2048), dtype=torch.bfloat16
        )
        # 19 native tensors per block: four authoritative tensors + 15 other V2 tensors.
        for extra in range(15):
            tensors[f"{stem}.test_extra_{extra}.weight"] = torch.zeros(
                (1,), dtype=torch.bfloat16
            )
    assert sum(rtmod._is_native_reference_key(key) for key in tensors) == 534
    for index in range(rtmod.E180_EXPECTED_TOTAL_TENSORS - len(tensors)):
        tensors[f"net.test_base_tensor_{index:04d}"] = torch.zeros(
            (1,), dtype=torch.bfloat16
        )
    assert len(tensors) == rtmod.E180_EXPECTED_TOTAL_TENSORS
    return tensors


@pytest.fixture(scope="module")
def fake_e180(tmp_path_factory):
    path = tmp_path_factory.mktemp("e180") / "fake-e180.safetensors"
    metadata = dict(rtmod.E180_REQUIRED_METADATA)
    metadata["anima_native_reference_config"] = json.dumps(
        dict(rtmod.E180_EXPECTED_CONFIG), sort_keys=True, separators=(",", ":")
    )
    save_file(_fake_e180_state_dict(), str(path), metadata=metadata)
    return path


def test_strict_e180_metadata_layout_and_dtype_validation(fake_e180, monkeypatch):
    monkeypatch.setattr(rtmod, "E180_EXPECTED_FILE_SIZE", fake_e180.stat().st_size)
    info = rtmod.validate_e180_checkpoint(fake_e180)
    assert info.config == dict(rtmod.E180_EXPECTED_CONFIG)
    assert info.tensor_count == 1222
    assert info.native_tensor_count == 534
    assert info.dtype_counts == {"BF16": 1222}
    assert info.sha256 is None


def test_strict_e180_metadata_mismatch_fails_closed(fake_e180, monkeypatch):
    monkeypatch.setattr(rtmod, "E180_EXPECTED_FILE_SIZE", fake_e180.stat().st_size)
    required = dict(rtmod.E180_REQUIRED_METADATA)
    required["ss_epoch"] = "181"
    monkeypatch.setattr(rtmod, "E180_REQUIRED_METADATA", required)
    with pytest.raises(rtmod.CheckpointValidationError, match="ss_epoch"):
        rtmod.validate_e180_checkpoint(fake_e180)


def test_auxiliary_model_layout_validation_and_wrong_selection_rejection(tmp_path):
    auxiliary = tmp_path / "auxiliary.safetensors"
    save_file(
        {
            "embed.weight": torch.zeros((3, 4), dtype=torch.bfloat16),
            "norm.weight": torch.zeros((4,), dtype=torch.bfloat16),
        },
        str(auxiliary),
    )
    fingerprint = rtmod._validate_auxiliary_safetensors(
        auxiliary,
        label="test auxiliary",
        expected_file_size=auxiliary.stat().st_size,
        expected_tensor_count=2,
        required_shapes={"embed.weight": (3, 4), "norm.weight": (4,)},
        expected_sha256="unused-without-full-verification",
        verify_sha256=False,
    )
    assert fingerprint.path == str(auxiliary.resolve())

    with pytest.raises(rtmod.CheckpointValidationError, match="shape mismatch"):
        rtmod._validate_auxiliary_safetensors(
            auxiliary,
            label="test auxiliary",
            expected_file_size=auxiliary.stat().st_size,
            expected_tensor_count=2,
            required_shapes={"embed.weight": (4, 3)},
            expected_sha256="unused",
            verify_sha256=False,
        )

    wrong = tmp_path / "wrong.safetensors"
    wrong.write_bytes(b"not-the-published-qwen")
    with pytest.raises(rtmod.CheckpointValidationError, match="byte size mismatch"):
        rtmod.validate_qwen_text_encoder(wrong)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, return_tensors, truncation, padding, max_length):
        self.calls.append(list(texts))
        assert return_tensors == "pt"
        assert truncation is True
        assert padding == "max_length"
        ids = torch.zeros((1, max_length), dtype=torch.long)
        mask = torch.zeros((1, max_length), dtype=torch.long)
        ids[:, :4] = torch.tensor([[1, 2, 3, 4]])
        mask[:, :4] = 1
        return {"input_ids": ids, "attention_mask": mask}


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
        hidden = torch.ones(
            (*input_ids.shape, 8), dtype=torch.bfloat16, device=input_ids.device
        )
        return SimpleNamespace(last_hidden_state=hidden)


class FakeAnima:
    patch_spatial = 2

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.calls = []
        self.scales = []

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def set_native_reference_fixed2ref_vectorized(self, enabled):
        assert enabled is False

    def set_native_reference_scale(self, scale):
        self.scales.append(float(scale))

    def _preprocess_text_embeds(
        self,
        *,
        source_hidden_states,
        target_input_ids,
        target_attention_mask,
        source_attention_mask,
    ):
        self.last_target_ids = target_input_ids.detach().cpu()
        self.last_source_mask = source_attention_mask.detach().cpu()
        # Match the model contract's [B,512,1024]-like output with a tiny width.
        return source_hidden_states.to(self.device)

    def __call__(self, latents, timestep, embed, **kwargs):
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
        # Red reference -> positive marker; blue -> negative marker.
        marker = (pixels[:, 0].mean() - pixels[:, 2].mean()).to(torch.bfloat16)
        return marker.view(1, 1, 1, 1).expand(1, 16, 2, 2).contiguous()

    def decode_to_pixels(self, latent):
        value = latent[:, :3, 0, :1, :1].expand(1, 3, 16, 16)
        return value.clamp(-1, 1)


class FakeHunyuan:
    @staticmethod
    def get_timesteps_sigmas(steps, flow_shift, device):
        return torch.linspace(1000, 1, steps, device=device), torch.linspace(
            1, 0, steps + 1, device=device
        )

    @staticmethod
    def step(latents, noise_pred, sigmas, index):
        return latents - noise_pred * 0


def _bare_runtime(tmp_path: Path):
    obj = object.__new__(rtmod.AnimaNativeReferenceV2Runtime)
    obj.lock = threading.RLock()
    obj._closed = False
    obj.device = torch.device("cpu")
    obj.dtype = torch.bfloat16
    obj.offload_mode = "balanced"
    obj._anima = FakeAnima()
    obj._vae = FakeVAE()
    obj._text_encoder = FakeTextEncoder()
    obj._qwen_tokenizer = FakeTokenizer()
    obj._t5_tokenizer = FakeTokenizer()
    obj._modules = SimpleNamespace(
        qwen_image_autoencoder_kl=SimpleNamespace(SCALE_FACTOR=8),
        anima_models=SimpleNamespace(Anima=SimpleNamespace(LATENT_CHANNELS=16)),
        hunyuan_image_utils=FakeHunyuan,
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
    return obj


def test_unicode_prompt_is_passed_to_both_tokenizers_without_mojibake(tmp_path):
    runtime = _bare_runtime(tmp_path)
    prompt = "动漫キャラクター，保持角色特征"
    encoded = runtime._encode_text_uncached(prompt)
    assert runtime._qwen_tokenizer.calls == [[prompt]]
    assert runtime._t5_tokenizer.calls == [[prompt]]
    assert encoded.shape == (1, 512, 8)
    assert encoded.device.type == "cpu"


def test_reference_order_and_second_layer_latent_cache(tmp_path):
    runtime = _bare_runtime(tmp_path)
    ref1 = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    ref1[..., 0] = 1.0  # red -> +2 marker
    ref2 = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    ref2[..., 2] = 1.0  # blue -> -2 marker

    first = runtime._encode_reference_pair(ref1, ref2, reference_max_area=65536)
    second = runtime._encode_reference_pair(ref1, ref2, reference_max_area=65536)
    assert first[0][0].float().mean() > 0
    assert first[0][1].float().mean() < 0
    assert torch.equal(first[0][0], second[0][0])
    assert torch.equal(first[0][1], second[0][1])
    assert runtime._vae.encode_calls == 2
    stats = runtime.cache_statistics()
    assert (stats.reference_misses, stats.reference_hits, stats.reference_entries) == (
        2,
        2,
        2,
    )


def test_prompt_cache_deduplicates_equal_positive_and_negative(tmp_path):
    runtime = _bare_runtime(tmp_path)
    positive, negative = runtime._get_text_embeddings("same", "same")
    assert torch.equal(positive, negative)
    assert runtime._qwen_tokenizer.calls == [["same"]]
    runtime._get_text_embeddings("same", "same")
    stats = runtime.cache_statistics()
    assert stats.prompt_misses == 1
    assert stats.prompt_hits == 1


def test_denoise_uses_fixed_ordered_slots_and_never_legacy_ip_route(tmp_path):
    runtime = _bare_runtime(tmp_path)
    reference_latents = [[torch.ones((1, 16, 2, 2)), torch.full((1, 16, 2, 2), 2.0)]]
    progress = []
    result = runtime._denoise(
        context=torch.zeros((1, 4, 8)),
        negative_context=torch.zeros((1, 4, 8)),
        reference_latents=reference_latents,
        width=16,
        height=16,
        seed=123,
        steps=3,
        guidance_scale=1.0,
        flow_shift=5.0,
        progress_callback=lambda done, total: progress.append((done, total)),
        interrupt_callback=None,
    )
    assert result.shape == (1, 16, 1, 2, 2)
    expected_generator = torch.Generator(device="cpu").manual_seed(123)
    expected_noise = torch.randn(
        result.shape,
        generator=expected_generator,
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert torch.equal(result, expected_noise)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(runtime._anima.calls) == 3
    for call in runtime._anima.calls:
        assert call["reference_latents"] is reference_latents
        assert call["reference_slot_ids"] == [[0, 1]]
        assert call["use_reference_sequence"] is True
        assert call["use_ip_adapter"] is False
        assert call["ip_adapter_latents"] is None
        assert call["ip_adapter_embeds"] is None


def test_cfg_matches_formal_evaluator_and_runs_for_every_nonunit_scale(tmp_path):
    runtime = _bare_runtime(tmp_path)
    reference_latents = [[torch.zeros((1, 16, 2, 2)), torch.zeros((1, 16, 2, 2))]]
    common = {
        "context": torch.zeros((1, 4, 8)),
        "negative_context": torch.ones((1, 4, 8)),
        "reference_latents": reference_latents,
        "width": 16,
        "height": 16,
        "seed": 123,
        "steps": 2,
        "flow_shift": 5.0,
        "progress_callback": None,
        "interrupt_callback": None,
    }

    runtime._denoise(guidance_scale=0.5, **common)
    assert len(runtime._anima.calls) == 4

    runtime._anima.calls.clear()
    runtime._denoise(guidance_scale=3.5, **common)
    assert len(runtime._anima.calls) == 4


def test_generate_is_serialized_updates_cached_scale_and_returns_raw_bhwc(
    tmp_path, monkeypatch
):
    runtime = _bare_runtime(tmp_path)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_refs(*args, **kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with active_lock:
            active -= 1
        return [[torch.zeros((1, 16, 2, 2)), torch.zeros((1, 16, 2, 2))]]

    monkeypatch.setattr(runtime, "_encode_reference_pair", fake_refs)
    monkeypatch.setattr(
        runtime, "_get_text_embeddings", lambda *a: (torch.zeros(1), torch.zeros(1))
    )
    monkeypatch.setattr(runtime, "_denoise", lambda **k: torch.zeros((1, 16, 1, 2, 2)))
    monkeypatch.setattr(
        runtime, "_decode_latent", lambda latent: torch.full((1, 16, 16, 3), 0.25)
    )
    image = torch.zeros((1, 16, 16, 3))
    outputs = []

    def worker(scale):
        outputs.append(
            runtime.generate(
                image,
                image,
                "角色参考",
                width=16,
                height=16,
                native_reference_scale=scale,
            )
        )

    threads = [threading.Thread(target=worker, args=(scale,)) for scale in (0.5, 1.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max_active == 1
    assert sorted(runtime._anima.scales) == [0.5, 1.0]
    assert len(outputs) == 2
    assert all(tuple(output.shape) == (1, 16, 16, 3) for output in outputs)
    assert all(torch.all(output == 0.25) for output in outputs)


def test_process_model_cache_loads_identical_fingerprint_once(tmp_path, monkeypatch):
    rtmod.clear_model_cache(close=False)
    model = tmp_path / "model.safetensors"
    text = tmp_path / "text.safetensors"
    vae = tmp_path / "vae.safetensors"
    root = tmp_path / "runtime-root"
    root.mkdir()
    for path, payload in ((model, b"m"), (text, b"t"), (vae, b"v")):
        path.write_bytes(payload)

    created = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            self._closed = False
            self.limits = []
            self.verify_calls = 0
            created.append((args, kwargs))

        def close(self):
            self._closed = True

        def set_condition_cache_limits(self, **kwargs):
            self.limits.append(kwargs)

        def verify_release_sha256(self):
            self.verify_calls += 1

    monkeypatch.setattr(rtmod, "AnimaNativeReferenceV2Runtime", FakeRuntime)
    one = rtmod.get_or_create_runtime(model, text, vae, runtime_root=root, device="cpu")
    two = rtmod.get_or_create_runtime(model, text, vae, runtime_root=root, device="cpu")
    assert one is two
    assert len(created) == 1

    # The process LRU, not an incidental loader output, must own a strong
    # reference so Comfy --cache-none cannot trigger a second disk load.
    runtime_ref = weakref.ref(one)
    del one, two
    gc.collect()
    assert runtime_ref() is not None
    three = rtmod.get_or_create_runtime(
        model, text, vae, runtime_root=root, device="cpu"
    )
    assert three is runtime_ref()
    assert len(created) == 1

    # Condition-cache capacity is part of the process cache contract.
    four = rtmod.get_or_create_runtime(
        model, text, vae, runtime_root=root, device="cpu", prompt_cache_entries=7
    )
    assert four is three
    assert four.limits[-1] == {
        "prompt_cache_entries": 7,
        "reference_cache_entries": 16,
    }
    assert len(created) == 1

    five = rtmod.get_or_create_runtime(
        model, text, vae, runtime_root=root, device="cpu", verify_checkpoint_sha256=True
    )
    assert five is three
    assert five.verify_calls == 1
    assert len(created) == 1
    rtmod.clear_model_cache(close=True)


def test_process_model_cache_is_bounded_lru_and_clear_closes(tmp_path, monkeypatch):
    rtmod.clear_model_cache(close=False)
    text = tmp_path / "text.safetensors"
    vae = tmp_path / "vae.safetensors"
    root = tmp_path / "runtime-root"
    root.mkdir()
    text.write_bytes(b"t")
    vae.write_bytes(b"v")
    models = []
    for index in range(rtmod.PROCESS_MODEL_CACHE_CAPACITY + 1):
        model = tmp_path / f"model-{index}.safetensors"
        model.write_bytes(f"m{index}".encode())
        models.append(model)

    created = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            self._closed = False
            created.append(self)

        def close(self):
            self._closed = True

        def set_condition_cache_limits(self, **kwargs):
            self.limits = kwargs

        def verify_release_sha256(self):
            self.verified = True

    monkeypatch.setattr(rtmod, "AnimaNativeReferenceV2Runtime", FakeRuntime)
    live = [
        rtmod.get_or_create_runtime(model, text, vae, runtime_root=root, device="cpu")
        for model in models
    ]
    assert len(rtmod._MODEL_CACHE) == rtmod.PROCESS_MODEL_CACHE_CAPACITY
    assert len(created) == rtmod.PROCESS_MODEL_CACHE_CAPACITY + 1

    # model-0 was least recently used and must require reconstruction.
    rebuilt = rtmod.get_or_create_runtime(
        models[0], text, vae, runtime_root=root, device="cpu"
    )
    assert rebuilt is not live[0]
    assert len(created) == rtmod.PROCESS_MODEL_CACHE_CAPACITY + 2
    assert len(rtmod._MODEL_CACHE) == rtmod.PROCESS_MODEL_CACHE_CAPACITY

    cached = list(rtmod._MODEL_CACHE.values())
    rtmod.clear_model_cache(close=True)
    assert not rtmod._MODEL_CACHE
    assert all(runtime._closed for runtime in cached)


def test_comfy_image_rejects_nonfinite_and_batch_greater_than_one():
    invalid = torch.zeros((1, 16, 16, 3))
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(rtmod.InputValidationError, match="NaN or Inf"):
        rtmod.AnimaNativeReferenceV2Runtime._comfy_image_to_pil(invalid, name="ref")
    with pytest.raises(rtmod.InputValidationError, match="fixed B=1"):
        rtmod.AnimaNativeReferenceV2Runtime._comfy_image_to_pil(
            torch.zeros((2, 16, 16, 3)), name="ref"
        )
