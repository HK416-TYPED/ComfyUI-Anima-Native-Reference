# ComfyUI Anima Native Reference V2 E180 integration report

**Result: PASS**  
Date: 2026-07-20 (Asia/Shanghai)

## Tested runtime

- ComfyUI `0.19.1`, git `8f374716ee98d378d403ebc61250e091ecd3a25c`
- Frontend `1.42.11`
- PyTorch `2.8.0+cu128`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (97,252 MiB)
- Custom node source: **122/122 files byte-identical** to the local shared artifact; manifest SHA-256 `82e750abf37ed8ccbb5c0b78d47e7de3239776b59519c3e9bd15f76cd04d2989`
- Temporary server listened only on `127.0.0.1:8189` and was stopped after verification.

## Registration and unit verification

- Live `/object_info` registered both `AnimaNativeRefV2Loader` and `AnimaNativeRefV2Generate`.
- Model dropdowns resolved the E180 diffusion model, Qwen3 text encoder and Qwen Image VAE symlinks.
- Post-integration suite: **18 passed, 0 failed** (`pytest_after_integration.log`). Four warnings are pre-existing Python string-literal deprecations in vendored files.

## Real API workflow

The API graph loaded held-out pair `000052` through standard `LoadImage` nodes and called the custom generate node with the formal contract:

- ordered slots `[0, 1]`
- `256 × 256`, 40 steps
- seed `20260720`
- CFG `1.0`, flow shift `5.0`
- native-reference scale `1.0`
- reference max area `65,536`

Run 1, including heavyweight model construction, completed in **19.170 s**. A second 40-step request with a new seed completed in **7.061 s**.

The saved output is RGB `256 × 256`; references are `895 × 1352` and `2848 × 4323`. Therefore the node returned only the raw generated frame, not a concatenated `Ref1 | Ref2 | Generated` diagnostic canvas.

## Formal CLI differential

Comfy output and the existing formal CLI `generated_only/E180/000052.png` are **pixel-exact**:

- pixel SHA-256: `20ba2661e8667bea3448c05021d5694d470474f7d39105ef1ee54096c394d04f`
- maximum absolute error: `0`
- nonzero channel values: `0 / 196608`

The PNG file hashes differ because Comfy embeds prompt metadata; decoded RGB pixels do not differ.

## Cache verification

Across two API executions, the server log contains exactly:

- one `Loading DiT model from ...`
- one `Loading Qwen3 text encoder from ...`
- one `Loading VAE from ...`

The second prompt executed without reloading any heavyweight model. This validates process model-cache reuse through the live Comfy executor, not just a mocked unit test.

## UI workflow V1 verification

- The workflow passes the latest official ComfyUI Workflow JSON V1 schema with zero errors.
- The live template index exposes `ComfyUI-Anima-Native-Reference / anima_ref_v2_e180`.
- The live static-template endpoint served the JSON successfully.
- Posting it to Comfy's user-workflow persistence endpoint and reading it back was byte-exact.
- All 7 nodes, 5 links, socket types and custom-node input order match live `/object_info` definitions.
- Source/served/round-trip SHA-256: `5ed34f1805c6cca513591f673a12f51f9e77f553f86cd1dd2d82c5f5b55c6e76`.

## Cleanup

- Temporary Comfy PID terminated: yes
- `127.0.0.1:8189` closed: yes
- remaining Comfy Python processes: 0
- remaining GPU compute processes after shutdown: 0
- held-out inputs removed from the shared Comfy input folder after copies were retained in this evidence directory.

## Environment-only warnings

1. Comfy reports that CUDA 13.0+ PyTorch is required for its newest optimized kernels. This isolated environment uses CUDA 12.8 PyTorch, so the validated eager/torch-attention path was used.
2. The isolated temporary user-directory emitted a SQLite initialization warning. Registration, template serving, prompt execution, history polling and image saving all succeeded; it is unrelated to this custom node.

## Principal evidence

- `comfy_server.log`
- `api_client.log`, `api_run_results.json`, `run1_history.json`, `run2_history.json`
- `pixel_comparison.json`
- `ui_v1_validation.json`
- `object_info_loader.json`, `object_info_generate.json`
- `pytest_after_integration.log`
- `source_sync_verification.json`
- `gpu_before_shutdown.txt`, `gpu_after_shutdown.txt`, `post_session_process_scan.json`
- `output/anima_ref_v2_e180_api_run1_00001_.png`
