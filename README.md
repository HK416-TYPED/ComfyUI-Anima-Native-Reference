# ComfyUI — Anima Native Reference V4

ComfyUI custom nodes for the final, self-contained Anima competitive
text-slot-router checkpoint. The plugin supports:

- one real reference assigned to logical **Image 1 / slot 0**;
- one real reference assigned to logical **Image 2 / slot 1**; and
- two real references assigned to logical slots **0 + 1**.

Generation begins from pure noise. Visual information enters through native
Qwen-Image VAE reference-latent sequences and integrated DiT reference
attention. This is not img2img, so there is intentionally no denoise-strength
input.

The model file contains the learned reference route and competitive router.
No external LoRA, IP-Adapter, ControlNet, or reference-adapter sidecar is
mounted.

Existing V2 E180 node class IDs and workflows remain in the package for
backward compatibility.

## Nodes

| Node | Purpose |
|---|---|
| **Anima Reference V4 Loader (Final 49k)** | Loads and process-caches the exact final integrated checkpoint, Qwen3-0.6B text encoder, and Qwen-Image VAE. |
| **Anima Reference V4 Generate (1 Ref)** | Encodes one physical reference once, assigns it to logical slot 0 or 1, and supports character/reference or one-image-edit geometry. |
| **Anima Reference V4 Generate (2 Refs)** | Encodes Image 1 and Image 2 independently and binds prompt clauses to logical slots 0 and 1. |
| **Anima Reference V2 Loader (E180)** | Legacy E180 loader retained without changing its class ID. |
| **Anima Reference V2 Generate (2 Refs)** | Legacy two-reference E180 generation node. |

The V4 pipeline type is `ANIMA_NATIVE_REF_V4_PIPELINE`; it cannot be
accidentally connected to the legacy V2 generation node.

## Final checkpoint contract

Place this file in `ComfyUI/models/diffusion_models/`:

```text
anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors
```

| Property | Exact value |
|---|---|
| Bytes | `4,302,295,014` |
| SHA256 | `4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f` |
| Total tensors | `1,614` BF16 |
| Integrated native tensors | `926` BF16 |
| Routing mode | `competitive_text_slot_v1` |
| Routing alpha | `1.0` |
| Logical reference capacity | `2` |
| Formal optimizer updates | `49,000` |
| Formal epochs | `2` |

The metadata field `anima_native_reference_version` is deliberately still
`"2"`. “V4” is the project/release generation that adds competitive
text-slot routing; it does not rename the underlying integrated V2 reference
architecture. The loader validates the router config, all 926 native key names
and shapes, file size, metadata, tensor counts, and BF16 dtypes. Optional full
SHA256 verification reads the complete checkpoint once.

The final file is fully integrated: selecting the old V2 E180 file in the V4
loader fails closed rather than silently constructing a wrong graph.

## Required base assets

Place:

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors
    ├── text_encoders/
    │   └── qwen_3_06b_base.safetensors
    └── vae/
        └── qwen_image_vae.safetensors
```

| File | Bytes | SHA256 |
|---|---:|---|
| `qwen_3_06b_base.safetensors` | `1,192,135,096` | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| `qwen_image_vae.safetensors` | `253,806,246` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

The base assets are available from
[`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima).
The only release path for the final V4 checkpoint is
[`checkpoints/v4-scaled-49k/anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors`](https://huggingface.co/LAXMAYDAY/NOOB2-Project-Character-Reference-Bypass-Injector-Research/blob/main/checkpoints/v4-scaled-49k/anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors).
The local ComfyUI filename is the same basename. This engineering working copy
does not itself upload model weights.

Keep Hugging Face tokens outside workflow JSON, shell scripts, and the custom
node directory.

## Why both Qwen and “neutral T5” appear

No T5 encoder model is loaded.

The standard text path is:

1. the Qwen tokenizer creates Qwen input IDs and an attention mask;
2. Qwen3-0.6B produces raw hidden states `[B,Lqwen,1024]`;
3. the neutral-T5 **tokenizer only** creates target token IDs and a mask;
4. the integrated DiT `llm_adapter` maps raw Qwen states onto that neutral-T5
   token sequence for ordinary Anima cross-attention.

The V4 router additionally reads:

- the positive prompt's raw Qwen hidden states;
- the positive Qwen attention mask; and
- boolean `reference_clause_masks` aligned to the raw Qwen token sequence.

The runtime never pre-adapts and discards raw Qwen. That early-adaptation
behavior belongs only to the legacy E180 runtime.

## Canonical Image 1 / Image 2 clauses

V4 competitive routing is instruction bound, not role hard-coded.

The physical/logical mapping is explicit:

```text
one-ref slot-0 node:
  one physical latent -> logical Image 1 / slot 0

one-ref slot-1 node:
  one physical latent -> logical Image 2 / slot 1

two-ref node:
  Reference Image 1 -> logical slot 0
  Reference Image 2 -> logical slot 1
```

Neither slot permanently means scene, identity, pose, source, or style. The
prompt clause says what should be taken from each slot.

Safe defaults:

```text
Use Image 1 as the character identity reference. Generate a new illustration.
```

```text
Use Image 2 as the character identity reference. Generate a new illustration.
```

```text
Use the pose and composition from Image 1; use the character appearance from Image 2.
```

For alpha-positive V4 reference sampling, the prompt must:

- mention every selected logical slot explicitly;
- not mention a slot that is absent;
- use a supported indexed form such as `Image 1`, `Image 2`, `Ref 1`,
  `Ref 2`, `first image`, or `second image`;
- preserve byte-exact canonical whitespace; and
- keep the full bound clause inside the 512-token Qwen sequence.

Ambiguous text such as `use both refs`, a dual request that only mentions
Image 1, a slot-0 request that mentions Image 2, non-canonical leading/doubled
whitespace, and truncated clauses fail before VAE encoding or denoising. The
runtime never silently falls back to an unbound router.

`reference_clause_masks` always have shape `[1,2,512]`, boolean dtype, and
occupancy matching the selected logical slots.

## One-reference carrier

The one-reference node does not duplicate the input to satisfy a nominal
two-reference interface.

It sends:

```text
slot 0: reference_latents = [[one_real_latent]], reference_slot_ids = [[0]]
slot 1: reference_latents = [[one_real_latent]], reference_slot_ids = [[1]]
```

The dual node sends:

```text
reference_latents = [[latent_image_1, latent_image_2]]
reference_slot_ids = [[0, 1]]
```

All three cases use the audited
`set_native_reference_fixed12ref_vectorized(True)` carrier. Duplicate slots,
missing physical inputs, fabricated second latents, and the older generic
carrier fail closed.

## Reference preprocessing

The one-reference node exposes:

| Mode | Intended task | Geometry |
|---|---|---|
| `independent_reference` | character/reference generation | Preserves the reference aspect ratio, downsizes only above `reference_max_area`, then center-crops to the VAE × DiT patch multiple. |
| `match_output_edit` | one-image edit | Uses `ImageOps.fit` to center-fit the source to the requested output width/height. |

The two-reference node always uses independent preprocessing for both images.
It does not infer semantic task type merely because one or two sockets happen
to be connected.

Both online and cached paths call the authoritative
`strategy_anima.preprocess_anima_reference_image` helper. Reference-cache keys
include preprocessing mode, output geometry, area limit, alignment multiple,
VAE fingerprint, and final prepared pixels.

## CFG contract

At CFG `1.0`, one DiT forward is evaluated per denoising step.

At any CFG value other than `1.0`, the standard positive and negative text
paths remain different:

- positive branch: positive raw Qwen + positive T5 IDs/masks;
- negative branch: negative raw Qwen + negative T5 IDs/masks.

Router-only conditioning is deliberately shared from the positive branch:

- positive raw Qwen router context;
- positive router attention mask; and
- positive Image 1/Image 2 clause masks.

This prevents the unconditional branch from losing the selected slot semantics
while preserving normal negative-prompt CFG behavior.

## Installation

Requirements:

- current ComfyUI with Python 3.10 or newer;
- NVIDIA CUDA GPU with BF16 support;
- enough GPU/host memory for Anima 2B, Qwen3-0.6B, and the VAE;
- the exact files above.

Copy this directory to:

```text
ComfyUI/custom_nodes/ComfyUI-Anima-Native-Reference/
```

Then use the same Python interpreter that launches ComfyUI:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Anima-Native-Reference
python -m pip install -r requirements.txt
```

Do not replace a working ComfyUI CUDA `torch`/`torchvision` build. The
requirements intentionally do not install or pin PyTorch. They do directly
declare both `toml>=0.10.2,<1` and `imagesize>=1.4.1,<2`; do not rely on those
packages arriving accidentally through another custom node.

Restart ComfyUI.

## Example workflows

UI-v1 drag-and-drop workflows:

- [`example_workflows/anima_ref_v4_final_single_slot0.json`](example_workflows/anima_ref_v4_final_single_slot0.json)
- [`example_workflows/anima_ref_v4_final_single_slot1.json`](example_workflows/anima_ref_v4_final_single_slot1.json)
- [`example_workflows/anima_ref_v4_final_dual.json`](example_workflows/anima_ref_v4_final_dual.json)

Comfy `/prompt` API graphs:

- [`api_workflows/anima_ref_v4_final_single_slot0_api.json`](api_workflows/anima_ref_v4_final_single_slot0_api.json)
- [`api_workflows/anima_ref_v4_final_single_slot1_api.json`](api_workflows/anima_ref_v4_final_single_slot1_api.json)
- [`api_workflows/anima_ref_v4_final_dual_api.json`](api_workflows/anima_ref_v4_final_dual_api.json)

The API and UI workflow formats are intentionally separate.

All six published V4 workflows enable `verify_release_sha256` on the loader.
This performs a full integrity check of the release checkpoint, Qwen, and VAE
the first time that cached runtime is loaded. The loader node itself keeps the
option disabled by default so intentional custom/local weights remain usable.

Validated defaults:

| Setting | Value |
|---|---:|
| Output size | `256 × 256` |
| Steps | `30` |
| CFG | `3.5` |
| Flow shift | `5.0` |
| Native reference scale | `1.0` |
| Independent reference area limit | `65,536` pixels |
| Attention mode | `torch` |

The final run trained at 256-resolution buckets. Larger multiples of 16 are
accepted by the implementation but are not a validated quality promise.

## Memory modes and caching

| Mode | Behavior |
|---|---|
| `balanced` | Default. Keeps DiT on GPU and stages Qwen/VAE for their phases. |
| `high_vram` | Keeps DiT, Qwen, and VAE on GPU for faster repeated generation. |
| `text_encoder_cpu` | Encodes text on CPU as a low-VRAM, slower fallback. |

There are two model-cache layers:

1. ComfyUI may cache the Loader output;
2. the plugin maintains a thread-safe, strong-reference LRU for the two most
   recently used V2/V4 runtime identities.

Prompt raw four-tensor encodings and reference VAE latents use bounded caches.
Generation is serialized per runtime so mutable reference scale and device
staging cannot leak between concurrent requests.

## Runtime integrity

Before import, every vendored file is verified against
`vendor/VENDOR_MANIFEST.json`. The authoritative V4 overlay hashes are
recorded before the deterministic import-namespace transform.

`V4_IMPLEMENTATION_MANIFEST.json` records hashes for the production runtime,
nodes, V4 workflows, documentation, vendor manifest, and tests. It contains no
model weights or credentials.

Vendored imports exist only under:

```text
_anima_native_ref_vendor.library.*
_anima_native_ref_vendor.networks.*
```

The runtime does not append the vendor tree to `sys.path` and does not claim,
read, or overwrite generic top-level `library` or `networks` packages from
other ComfyUI extensions.

## CPU/mock verification

The package's test suite verifies:

- exact V4 metadata/config/count/dtype/key-shape validation and tamper failure;
- availability of the vendored binding/router/text/fixed12/preprocessing code;
- four-tensor raw Qwen + neutral-T5-token caching with Unicode;
- canonical slot-0, slot-1, and dual clause masks;
- ambiguous, wrong-slot, non-canonical, and missing-clause failure;
- CFG negative ordinary text plus positive-router sharing;
- one physical latent for one-ref slot 0/1 (no fake second);
- independent and target-output geometry/cache isolation;
- V2 class-ID compatibility;
- UI-v1 link consistency and API graph wiring for all V4 cases; and
- no denoise-strength input.

Run:

```bash
python -m pytest -q
python -m ruff check runtime.py nodes.py __init__.py tests tools
python -m compileall -q .
```

Real CUDA acceptance matched the accepted standalone CLI for slot-0, slot-1,
and dual cases at CFG `3.5`; CFG `1.0` was retained only as non-CFG regression
coverage. A live dual-reference `/prompt` run also passed. See
`TEST_REPORT.md`.

## Legacy V2 E180 compatibility

The following class IDs and files are unchanged:

```text
AnimaNativeRefV2Loader
AnimaNativeRefV2Generate
example_workflows/anima_ref_v2_e180.json
api_workflows/anima_ref_v2_e180_api.json
```

Legacy E180:

| Property | Value |
|---|---|
| File | `anima-native-ref-v2-e180-step64080-256px.safetensors` |
| Bytes | `4,271,362,542` |
| SHA256 | `1f970a7867dd7b65858d30b58135134ce84fd3b552ab27fc9f07e7f15209c6dd` |

The V2 node remains exactly two-reference and uses its accepted legacy text
and generic carrier path. Use the new V4 loader/nodes for the competitive
router and one-reference slots.

## Limitations

- Comfy batch size is currently fixed to one image/request (`B=1`);
- V4 publication accepts the exact final checkpoint, not arbitrary router
  experiments;
- CUDA BF16 only;
- no stock Comfy `MODEL`/`CLIP`/`VAE` outputs or stock KSampler;
- no dynamic N-reference sockets beyond the checkpoint's two logical slots;
- 256 is the trained/validated resolution;
- difficult pairs can still miss fine identity, pose, text, or compositional
  details;
- a custom inference implementation is still required even though learned
  weights are self-contained.

“No external adapter” means the learned model is one integrated checkpoint. It
does not mean unmodified upstream/stock Anima code knows the new competitive
router, clause masks, or fixed-one/two carrier.

## 中文快速说明

1. 最终 V4 是单一自包含模型文件，不再外挂 LoRA、Adapter 或 IP-Adapter。
2. 单参考节点可以把**唯一一张真实参考图**放到 `Image 1 / slot 0` 或
   `Image 2 / slot 1`；不会复制一张假装成第二参考图。
3. 双参考节点固定为 `Image 1 -> slot 0`、`Image 2 -> slot 1`，但槽位没有
   固定“场景/人物/姿势/风格”语义，具体取什么由 prompt 的对应子句决定。
4. prompt 必须明确、规范地写出所有正在使用的 `Image 1/Image 2`。缺槽、
   错槽、`use both refs`、非规范空格或截断都会在去噪前报错，不静默降级。
5. 文本路径是：Qwen 编码 raw hidden；neutral-T5 只负责 tokenizer IDs；
   DiT 内部 LLM adapter 做普通文本条件；V4 router 另外读取正向 raw Qwen
   和 `reference_clause_masks`。
6. CFG 的 negative 分支仍用自己的 negative prompt 做普通文本条件，但
   router 必须共享正向 prompt 的槽位子句。
7. 单参考 `independent_reference` 用于角色/参考生图；
   `match_output_edit` 用于把 edit 源图对齐到目标宽高。双参考始终独立保留
   两张图的比例。
8. 这是从纯噪声开始的 ref/edit 生成，不是 img2img，所以没有 denoise
   strength。
