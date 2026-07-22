# Runtime API for `nodes.py`

```python
from .runtime import get_or_create_runtime

pipe = get_or_create_runtime(
    checkpoint_path,
    text_encoder_path,
    vae_path,
    offload_mode="balanced",       # balanced | high_vram | text_encoder_cpu
    attn_mode="torch",             # formal E180 mode; other values fail
    vae_chunk_size=64,
    vae_disable_cache=True,
    prompt_cache_entries=64,
    reference_cache_entries=16,
    verify_checkpoint_sha256=False,  # when true, verifies E180 + Qwen + VAE
)

image = pipe.generate(
    reference_image_1,              # Comfy IMAGE [1,H,W,C], [0,1]
    reference_image_2,              # Comfy IMAGE [1,H,W,C], [0,1]
    prompt,
    negative_prompt="",
    width=256,
    height=256,
    seed=0,
    steps=40,
    guidance_scale=1.0,
    flow_shift=5.0,
    native_reference_scale=1.0,
    reference_max_area=65536,
    progress_callback=None,         # callable(done: int, total: int)
    interrupt_callback=None,        # zero-argument callable; raise to cancel
)
```

The return value is the **raw generated image only**: CPU `torch.float32`
`[1,H,W,3]`, range `[0,1]`. It never contains the reference-preview panels and
is never written to a temporary PNG.

Checkpoint inspection and cache observability:

```python
pipe.metadata                 # validated safetensors metadata mapping
pipe.fingerprint              # E180 FileFingerprint(path,size,mtime_ns)
pipe.model_fingerprints       # checkpoint / text_encoder / VAE fingerprints
pipe.checkpoint_info          # strict E180 architecture/layout result
pipe.cache_statistics()       # model/prompt/reference hit/miss counters
pipe.clear_condition_caches()
pipe.close()
```

`get_or_create_runtime` is backed by a process-wide, thread-safe,
strong-reference LRU with capacity two. This deliberately retains the two most
recent model runtimes even under ComfyUI `--cache-none` or loader-output
eviction. `clear_model_cache(close=True)` explicitly closes and removes all
process-cached runtimes; `pipe.close()` closes one runtime, and the next lookup
for that key reconstructs it. Prompt/reference cache capacities and SHA
verification are runtime policies rather than model identities: changing them
resizes/verifies the existing runtime instead of loading a second DiT/Qwen/VAE
set.

The backend is intentionally fixed to two ordered references and emits logical
slot IDs `[[0,1]]`. Prompt text defines their semantics; neither slot is
hard-coded as scene, identity, or style.

The pinned implementation is loaded only under the private
`_anima_native_ref_vendor.library.*` and
`_anima_native_ref_vendor.networks.*` namespaces. Loading the node does not add
the vendor tree to `sys.path` and does not claim or overwrite generic
top-level `library` or `networks` packages used by other ComfyUI extensions.
