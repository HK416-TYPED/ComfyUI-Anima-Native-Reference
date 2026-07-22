# Runtime backend test report

Date: 2026-07-20 (Asia/Shanghai)

## Local unit suite

Command:

```text
python -m pytest -q tests
```

Result:

```text
...............                                                          [100%]
15 passed in 3.59s
```

Covered contracts:

- exact E180 metadata/config/key-count/native-key-count/BF16/layout validation;
- published Qwen3/VAE byte-size, key-count, BF16 and authoritative-shape
  validation, with optional SHA-256 verification;
- pinned vendored source/config manifest and per-file SHA-256 validation;
- AST verification of all 207 vendored internal imports under the private
  `_anima_native_ref_vendor` namespace, plus a loader test proving that foreign
  top-level `library`/`networks` packages and `sys.path` remain untouched;
- fail-closed metadata mismatch;
- Unicode Chinese/Japanese prompt preservation through both tokenizers;
- ordered Ref1/Ref2 latent placement;
- bounded second-layer reference and prompt cache behavior;
- fixed `[[0,1]]` logical slots, native sequence enabled, legacy IP route disabled;
- formal-evaluator sampling parity for direct BF16 noise from the CPU generator;
- CLI CFG route parity: one conditional pass only at scale 1, positive and
  negative passes at every other scale;
- per-call native-reference scale update on a cached model;
- serialized concurrent execution with `RLock`;
- raw Comfy BHWC image return;
- process model-cache single construction for identical fingerprints;
- strong-reference, thread-safe, capacity-two LRU retention under simulated
  Comfy `--cache-none`, bounded eviction, and explicit close/clear behavior;
- NaN/Inf and `B>1` input rejection.

## Known-good Linux runtime import

The package was copied to an isolated development directory on the Pro6000
server (not into the formal training repository), then tested with:

```text
/root/autodl-tmp/envs/anima-edit/bin/python -m py_compile runtime.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "... load_runtime_modules() ..."
python -m pytest -q tests
```

Observed:

```text
vendored commit: 2ae811d296ff4159c6024c4a86415d19961a388c
anima_utils module: _anima_native_ref_vendor.library.anima_utils
anima_models module: _anima_native_ref_vendor.library.anima_models
foreign library/networks modules preserved: true
vendor root added to sys.path: false
VAE scale factor: 8
15 passed in 2.39s
```

## Real E180 header validation

The validator was run against the 4.27 GB server checkpoint without loading
model tensor data onto GPU. It returned:

```json
{
  "config": {
    "architecture": "v2",
    "gate_dim": 4,
    "initial_gate": 0.5,
    "max_reference_images": 2,
    "rank": 64,
    "router_dim": 64
  },
  "dtypes": {"BF16": 1222},
  "epoch": "180",
  "native_tensors": 534,
  "size": 4271362542,
  "steps": "64080",
  "tensors": 1222
}
```

## Real CUDA model load and generation

The current package was copied to an isolated directory on the Pro6000 and run
against the real E180, Qwen3-0.6B and Qwen Image VAE files. The formal training
repository and model files were not changed.

Observed results:

```text
E180 + Qwen + VAE load: 12.307 s
DiT state load:          0 missing, 0 unexpected
Qwen state load:         all keys matched
VAE state load:          all keys matched
process cache lookup:    same runtime object

2-step 256x256 smoke:    1.499 s
formal 40-step 256x256:  5.808 s
output:                  CPU float32 [1,256,256,3], finite, [0,1]
clear_model_cache:       runtime closed successfully
```

The 40-step run used the formal E180 contract: held-out pair `000052`, seed
`20260720`, CFG `1.0`, flow shift `5.0`, native-reference scale `1.0`, reference
max area `65536`, slots `[[0,1]]`, torch attention and the native sequence
route. The returned tensor contained only generated pixels.

## CLI/backend differential

The backend output above was quantized with the formal CLI's truncating uint8
save rule and compared against the existing formal CLI `generated_only/E180`
artifact for the same case. The comparison was pixel-exact:

```text
shape:          256x256 RGB
maximum error:  0
nonzero values: 0
pixel SHA-256:  20ba2661e8667bea3448c05021d5694d470474f7d39105ef1ee54096c394d04f
```

The complete 40-step differential was repeated after the private-namespace,
auxiliary-model validation and strong-LRU changes. It remained pixel-identical
(`max=0`, `nonzero=0`) and the process exited with no residual CUDA compute
process.

This differential validates the real model load, text path, ordered reference
encoding, seeded BF16 noise, scheduler, native V2 denoising and VAE decode—not
only the mocked unit contracts. A normal Comfy `SaveImage` may round differently
from the historical CLI serializer by one integer level; the in-memory float
tensor is the backend's authoritative output.
