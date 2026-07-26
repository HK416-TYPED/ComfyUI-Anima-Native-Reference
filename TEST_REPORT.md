# Anima Native Reference V4 — Engineering Acceptance Report

Date: 2026-07-24  
Scope: isolated release-candidate package and RTX PRO 6000 validation  
Hugging Face upload: **not performed**

## Release identity

The only public checkpoint path and the required local ComfyUI basename are:

```text
checkpoints/v4-scaled-49k/anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors
```

The local file belongs in:

```text
ComfyUI/models/diffusion_models/anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors
```

| Check | Exact value |
|---|---|
| Bytes | `4,302,295,014` |
| SHA256 | `4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f` |
| Tensor count | `1,614` BF16 |
| Native tensor count | `926` BF16 |
| Non-native/frozen tensor count | `688` BF16 |
| Native architecture metadata | `v2` |
| Routing mode | `competitive_text_slot_v1` |
| Routing alpha | `1.0` |
| Router temperature | `1.0` |
| Router null | enabled |
| Formal optimizer updates | `49,000` |
| Formal epochs | `2` |

The 926 native tensors comprise two model-level tensors and 33 tensors in each
of 28 DiT blocks. The validator checks exact metadata, JSON configs, key names,
shapes, counts, dtypes, size, and optionally the complete SHA256. The file is
self-contained: no external LoRA, IP-Adapter, or reference-adapter sidecar is
loaded.

## Validated public sampling contract

| Setting | Value |
|---|---:|
| Output size | `256 × 256` |
| Steps | `30` |
| CFG | `3.5` |
| Flow shift | `5.0` |
| Native reference scale | `1.0` |
| Independent reference area limit | `65,536` pixels |
| Attention mode | `torch` |

CFG `1.0` is retained as a non-CFG regression path, not advertised as the
validated public default. Legacy V2 E180 deliberately retains its historical
40-step / CFG-1 defaults.

## Implemented production contracts

### Text and routing

- raw Qwen hidden states and masks remain available to the competitive router;
- neutral-T5 supplies token IDs and masks only; no T5 encoder is loaded;
- no early `_preprocess_text_embeds` destroys the raw router context;
- `reference_clause_masks` are boolean `[1,2,512]`;
- single slot 0, single slot 1, and dual-slot occupancy are checked exactly;
- missing, wrong, ambiguous, non-canonical, and truncated bindings fail closed;
- CFG negative ordinary text stays negative;
- CFG negative router context/mask/clause masks are shared from the positive
  branch.

### Reference carrier and geometry

- one-reference requests carry one real VAE latent only, with logical `[[0]]`
  or `[[1]]`;
- dual requests carry two real latents with logical `[[0,1]]`;
- no duplicated/fabricated second latent and no IP-Adapter fallback;
- generation starts from pure noise;
- independent preprocessing and one-image target-fit preprocessing call the
  authoritative `strategy_anima.preprocess_anima_reference_image`;
- cache keys include mode, output geometry, area limit, alignment multiple,
  VAE fingerprint, and prepared pixels.

### Nodes and workflows

The package registers all five class IDs:

```text
AnimaNativeRefV2Loader
AnimaNativeRefV2Generate
AnimaNativeRefV4Loader
AnimaNativeRefV4Generate1Ref
AnimaNativeRefV4Generate2Refs
```

V4 uses a distinct `ANIMA_NATIVE_REF_V4_PIPELINE` socket type. Three UI-v1 and
three API workflows cover slot 0, slot 1, and dual refs. All six V4 workflows
use the exact release basename and the 30-step / CFG-3.5 public defaults. Their
loader instances set `verify_release_sha256=true`, so the published examples
fail closed on a truncated or replaced checkpoint, Qwen, or VAE. The loader
node's global default remains `false` for intentional custom/local weights.

## Local automated verification

Commands:

```bash
python -m pytest -q
python -m ruff check __init__.py nodes.py runtime.py tests tools
python -m compileall -q __init__.py nodes.py runtime.py tests tools vendor
```

Result:

```text
39 passed
ruff: All checks passed
compileall: passed
```

The suite includes:

- exact V4 and legacy E180 validator/tamper tests;
- a complete private-vendor subprocess import;
- raw Qwen plus neutral-T5-token caching with Unicode;
- slot-binding and fail-closed prompt tests;
- CFG positive-router sharing spies;
- one-physical-latent carrier spies;
- preprocessing/cache-isolation tests;
- UI/API link and model-URL validation;
- exact V2/V4 public-signature defaults:
  - V2: steps `40`, CFG `1.0`;
  - V4: steps `30`, CFG `3.5`.

## Direct dependency isolation test

`requirements.txt` now declares these direct runtime dependencies:

```text
toml>=0.10.2,<1
imagesize>=1.4.1,<2
```

A new clean virtual environment produced the expected sequence:

1. `import toml, imagesize` failed before installation;
2. installing only the two declared constraints succeeded;
3. imports then succeeded with `toml 0.10.2` and `imagesize 1.5.0`.

The Comfy validation environment independently contained `toml 0.10.2` and
`imagesize 1.4.1`, and `pip check` reported no broken requirements.

## RTX PRO 6000 direct-node CUDA parity

The isolated package was deployed under:

```text
ComfyUI-anima-ref-v4-final49k-preflight-20260724
```

It did not overwrite the historical V2 Comfy installation. The direct-node
test called the exported V4 Loader and Generate node classes with the exact
final checkpoint, Qwen, and VAE.

Evidence:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/direct_node/REPORT.json
SHA256 aee23d5f7e5ba210ba2c6644bdf8a486882bb2ef6f409058984aa106bb2c79b0
```

The parity subset used the already accepted structural minimal-CLI contract:
256×256, two denoising steps, CFG 3.5, flow shift 5, reference scale 1,
reference area 65,536, and seed 2026072401. Two steps are intentionally a
structural differential gate, not the public quality setting.

| Case | Logical slots | Accepted CLI parity |
|---|---:|---|
| Single reference, slot 0 | `[0]` | pixel exact; `0` differing values |
| Single reference, slot 1 | `[1]` | pixel exact; `0` differing values |
| Dual references | `[0,1]` | pixel exact; `0` differing values |

All three have `max_abs_u8=0` and `mean_abs_u8=0`. The same three cases at CFG
1.0 were finite and non-constant as regression coverage. Across all six
generations, the model loaded once; final cache counters were eight prompt hits
and five reference hits.

## Real ComfyUI HTTP `/prompt`

ComfyUI `0.19.1` was started from the isolated copy on temporary loopback port
8191. `/object_info` proved all five class IDs and the default split:

- V2 Generate: steps `40`, CFG `1.0`;
- V4 1Ref and 2Refs: steps `30`, CFG `3.5`.

Object-info evidence:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/live_api/OBJECT_INFO_REPORT.json
SHA256 57d7748164ccf62496bdb7343877d58b2a131739334fae2f66fada1c73ccd0d6
```

The dual-reference API workflow then ran twice at the exact public 30-step /
CFG-3.5 settings, with two different seeds. The prompt contained Chinese
Unicode text in addition to canonical Image 1/Image 2 clauses.

Evidence:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/live_api/LIVE_API_REPORT.json
SHA256 6c465ae4a0fab9a5d7ecd4b3072027f36a985261045c7fd05b9a5d2edcb77d73
```

| Run | Seed | Wall time | Saved PNG SHA256 |
|---|---:|---:|---|
| 1 | `2026072401` | `19.812 s` | `3ff01c1bebbdfce76283c5fb784c74a5d1fc71f1951c2587f86639ea1075e0df` |
| 2 | `2026072402` | `7.573 s` | `b1de18c7142006b3de111043cc36696113dd36faa434e47cf82ad1b36d391477` |

Both outputs were 256×256 RGB, finite, and non-constant. Unicode survived
request serialization, Comfy history, tokenization, and the Unicode
`SaveImage` filename. Run 2 cached nodes `1`, `2`, and `3` (both LoadImage
nodes and the V4 Loader), while the changed seed correctly forced generation.
The V4-only server log records exactly one DiT, Qwen, and VAE load across both
requests:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/live_api/SERVER_CACHE_REPORT.json
SHA256 ff2ed7280bd15451fc7efbea6a1c4c685cc973e16adcc065e843a77cd76fb796
```

The two published single-reference API workflows were also exercised through
real `/prompt` requests after enabling their release-hash option. Both used the
same physical reference and seed while assigning it to the distinct logical
slot selected by the workflow. Each request used the exact public 256×256,
30-step, CFG-3.5, flow-shift-5, reference-scale-1, and area-65,536 contract.

Evidence:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/live_api_single_slots/REPORT.json
SHA256 b12ac2a19a762f3e4e59a0fd0d0210516f353551297ebc36e63263296cb0ac9b
```

| Workflow | Logical slot | Wall time | Saved PNG SHA256 |
|---|---:|---:|---|
| Single slot 0 | `[0]` | `23.952 s` | `b13d7c6b2ad779e936538a604d5dfdba696dab7c22d92d89e0f68462dfe26c4b` |
| Single slot 1 | `[1]` | `8.334 s` | `12fdfc2c2a2245a2a3d7ae3d909d8544b7c45e840fe2ba680fe688ea402583f7` |

Both saved outputs were 256×256 RGB, finite, and non-constant. Comfy history
preserved the Unicode prompts, exact logical-slot selectors, and
`verify_release_sha256=true`. The second request cached its LoadImage and V4
Loader nodes but not its generator. With the same physical input and seed, the
two slot assignments produced different results (`47,983` differing RGB
values; maximum absolute difference `189`), and the server log again recorded
exactly one DiT, one Qwen, and one VAE load across both requests.

## Legacy V2 E180 real regression

The updated mixed V2/V4 package loaded the retained exact E180 checkpoint and
ran its accepted held-out case through a real `/prompt` request at legacy
40-step / CFG-1 settings.

Evidence:

```text
anima_v4_final49k_comfy_preflight_20260724_r4/legacy_v2/REPORT.json
SHA256 c3ca5d3c5138222d61a6fc6049358615000379eb60471719d94fa0cb8a48eda4
```

The generated RGB pixels were bit-identical to the accepted legacy CLI output:

```text
pixel_exact=true
differing_values=0
max_abs_u8=0
pixel_sha256=20ba2661e8667bea3448c05021d5694d470474f7d39105ef1ee54096c394d04f
```

The old production V2 package remained unchanged:

```text
runtime.py  6d20a28c0e33cca2ffa62346fd11205db8f6ed38ee04deb98211a8a528aea680
nodes.py    2e0550c7678aba023926d2b054c826087a5a72803ec0a226ff6bcd8243b6e1ad
```

## Shutdown and isolation

After acceptance:

- temporary Comfy port 8191 was closed;
- the temporary server process exited on SIGTERM;
- no GPU compute process remained;
- the historical V2 Comfy directory retained its original source hashes;
- no Hugging Face upload or remote repository mutation occurred.

## Vendored runtime integrity

Base commit:

```text
2ae811d296ff4159c6024c4a86415d19961a388c
```

Overlay:

```text
v4-scaled-mixture-final49k-20260724
```

`vendor/VENDOR_MANIFEST.json` covers 110 source/config/license files. Its
vendored-tree SHA256 is:

```text
67b25623eec3cb287c017b98255d9048bc8002869299a6a79a51f3bafabed251
```

All runtime imports use the collision-free private namespace
`_anima_native_ref_vendor.*`.

## Remaining limitations

- Comfy batch size is fixed to one output image per request;
- the release is trained and validated at nominal 256-pixel-area buckets;
- larger multiples of 16 are accepted by code but are not a quality promise;
- this report validates the release candidate but does not publish it.
