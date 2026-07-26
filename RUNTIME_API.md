# Runtime API

## Final V4

```python
from .runtime import get_or_create_v4_runtime

pipe = get_or_create_v4_runtime(
    checkpoint_path,
    text_encoder_path,
    vae_path,
    offload_mode="balanced",        # balanced | high_vram | text_encoder_cpu
    attn_mode="torch",              # fixed12 carrier supports torch only
    vae_chunk_size=64,
    vae_disable_cache=True,
    prompt_cache_entries=64,
    reference_cache_entries=16,
    verify_checkpoint_sha256=False, # when true: V4 + Qwen + VAE full SHA256
)
```

### One reference in logical slot 0

```python
image = pipe.generate(
    [reference_image],              # one Comfy IMAGE [1,H,W,C], [0,1]
    [0],                            # logical Image 1 / slot 0
    (
        "Use Image 1 as the character identity reference. "
        "Generate a new illustration."
    ),
    negative_prompt="",
    preprocess_mode="independent_reference",
    width=256,
    height=256,
    seed=0,
    steps=30,
    guidance_scale=3.5,
    flow_shift=5.0,
    native_reference_scale=1.0,
    reference_max_area=65536,
)
```

### One reference in logical slot 1

```python
image = pipe.generate(
    [reference_image],
    [1],                            # logical Image 2 / slot 1
    (
        "Use Image 2 as the character identity reference. "
        "Generate a new illustration."
    ),
    preprocess_mode="independent_reference",
)
```

The physical carrier contains one and only one reference latent in both cases.

### One-image edit geometry

```python
image = pipe.generate(
    [source_image],
    [0],
    "Use Image 1 as the source image reference. Generate a new illustration.",
    preprocess_mode="match_output_edit",
    width=768,
    height=512,
)
```

`match_output_edit` center-fits the source with `ImageOps.fit` to the requested
output geometry. It is rejected for two-reference generation.

### Two references

```python
image = pipe.generate(
    [reference_image_1, reference_image_2],
    [0, 1],
    (
        "Use the pose and composition from Image 1; "
        "use the character appearance from Image 2."
    ),
    negative_prompt="",
    preprocess_mode="independent_reference",
    width=256,
    height=256,
    seed=0,
    steps=30,
    guidance_scale=3.5,
    flow_shift=5.0,
    native_reference_scale=1.0,
    reference_max_area=65536,
    progress_callback=None,         # callable(done: int, total: int)
    interrupt_callback=None,        # zero-argument callable; raise to cancel
)
```

The return value is the raw generated image only: CPU `torch.float32`
`[1,H,W,3]` in `[0,1]`. It is not a contact sheet and is never written to a
temporary image.

### Text contract

The prompt cache stores exactly:

```text
raw_qwen_context
source_attention_mask
target_input_ids
target_attention_mask
```

No T5 encoder exists. The final two tensors are neutral-T5 tokenizer outputs
for the integrated DiT LLM adapter.

For every active V4 reference request,
`prepare_anima_prompt_conditioning(...)` builds boolean
`reference_clause_masks [1,2,512]`. Canonical parsing fails closed on missing,
wrong, ambiguous, non-canonical, or truncated bindings.

At CFG other than 1:

```python
positive_kwargs = positive.model_text_kwargs(...)
negative_kwargs = negative.model_text_kwargs(
    ...,
    router_conditioning=positive,
)
```

Thus negative ordinary text is negative, while router-only raw Qwen/mask/clause
conditioning is positive.

### Observability and lifecycle

```python
pipe.metadata
pipe.fingerprint
pipe.model_fingerprints
pipe.checkpoint_info
pipe.cache_statistics()
pipe.clear_condition_caches()
pipe.close()
```

`get_or_create_v4_runtime` uses a process-wide, thread-safe, strong-reference
LRU shared with the legacy V2 runtime. Runtime identity includes:

- runtime kind (`v4-final49k` versus `v2`);
- checkpoint/Qwen/VAE file fingerprints;
- private vendor root;
- CUDA device;
- offload mode;
- attention mode; and
- VAE chunk/cache settings.

Prompt/reference cache capacities and full SHA verification are policies on an
existing runtime rather than separate model identities.

## Legacy V2 E180

The backward-compatible API remains:

```python
from .runtime import get_or_create_runtime

legacy_pipe = get_or_create_runtime(
    e180_checkpoint_path,
    text_encoder_path,
    vae_path,
)

legacy_image = legacy_pipe.generate(
    reference_image_1,
    reference_image_2,
    prompt,
)
```

This remains exactly two-reference and does not implement V4 clause routing.

## Private vendor namespace

The embedded runtime imports only:

```text
_anima_native_ref_vendor.library.*
_anima_native_ref_vendor.networks.*
```

It does not modify `sys.path` and does not populate or overwrite generic
top-level `library` or `networks` modules.
