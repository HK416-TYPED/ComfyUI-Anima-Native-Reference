# ComfyUI Anima Native Context V7 - 512 E4

Package version: **7.0.2**

Checkpoint: **`anima-native-context-v7-512-e4.safetensors`**

This package exposes the integrated Anima Native Context V7 checkpoint as five
ComfyUI nodes: loader, true no-image T2I, one-reference generation,
two-reference generation, and one-source maskless Edit. The checkpoint already
contains the complete visual-conditioning path. It does not load an external
LoRA, adapter, IP-Adapter, ControlNet, or other sidecar.

## Exact prompt contract

The prompt visible in the node is exactly the prompt encoded by the normal
Anima text path. The node/runtime never prepends `Use Image 1`, invents image
roles, parses prompt clauses, or rewrites an empty or non-empty prompt. Example
workflows contain ordinary visible example prompts only.

| Node | Structural image metadata |
|---|---|
| T2I | no visual sequence; hard bypass |
| 1 Reference | the sole physical image is structural instance `0` |
| 2 References | physical order is instance `0`, then instance `1` |
| Edit | the sole image is instance `0` plus `aligned_source=true` |

Instance indices distinguish simultaneous visual streams. They do not assign
hard-coded identity, pose, style, or background roles.


## 7.0.2 full-directory hotfix

7.0.2 fixes a Python method-binding regression that made all visual paths fail
before reference preprocessing (_normalise_reference_request,
_comfy_image_to_pil, and _validate_sample_inputs were state-free helpers but
were missing @staticmethod). It also makes the 512 defaults authoritative in
one place (512 x 512, reference area 262144).

**Upgrade atomically:** stop ComfyUI, delete the complete old
custom_nodes/ComfyUI-Anima-Native-Context directory, install the complete
7.0.2 directory, then restart ComfyUI. Do not overlay individual Python files;
ComfyUI keeps imported modules and cached node schemas until restart.

The ComfyUI-Danbooru-Gallery/py/metadata_collector/metadata_hook.py frame may
appear above a node error because that extension wraps ComfyUI execution. The
V7 7.0.1 failure it exposed was inside this package and is fixed here.

## Installation

Extract this directory to:

```text
ComfyUI/custom_nodes/ComfyUI-Anima-Native-Context
```

Install `requirements.txt` with the Python interpreter that launches ComfyUI.
Place the model files as follows:

```text
ComfyUI/models/diffusion_models/anima-native-context-v7-512-e4.safetensors
ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors
ComfyUI/models/vae/qwen_image_vae.safetensors
```

Restart ComfyUI. Nodes appear under **Anima / Native Context V7**.

## 512-stage defaults

- output: `512 x 512` (aspect-ratio sizes are also supported);
- maximum area per reference: `262144` pixels (`512 x 512`);
- steps: `30`;
- CFG: `3.5`;
- flow shift: `5.0`;
- single/multi-reference scale: `1.0`;
- Edit scale: `1.4`.

Reference generation starts from pure noise and has no denoise-strength
control. Edit has no inference mask. T2I supplies no image or fabricated blank
visual token.

## Fail-closed validation

The loader requires 1,569 BF16 integrated tensors, `native_context_v1`, the
exact-prompt contract, structural instances, native VAE-latent read-only
reference K/V attention, the integrated edit-scope predictor, and no retired
attribute/index/pointer router or external LoRA tensor.

Some trained V7 files omit three redundant boolean metadata keys. Version
7.0.2 accepts a missing key only when its stricter equivalent
contract is present. An explicit conflicting value still fails.

## Workflows and limitations

UI workflows are in `example_workflows/`; API graphs are in `api_workflows/`.
They all target `anima-native-context-v7-512-e4.safetensors`, use 512 defaults, and contain no hidden prompt
template or logical-slot selector.

This research checkpoint is strongest for anime single- and multi-reference
generation. Fine identity attributes can still simplify or leak, and maskless
Edit can change a broader region than requested.
