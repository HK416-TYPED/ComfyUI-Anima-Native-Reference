# Vendored Anima inference runtime

This directory contains the code/configuration snapshot required to load both
the integrated Native Reference V2 E180 checkpoint and the final V4
competitive text-slot-router checkpoint in-process. It contains **no model
weights**.

- Source repository: <https://github.com/akatsuki-neo/anima-edit.git>
- Audited source commit: `2ae811d296ff4159c6024c4a86415d19961a388c`
- Audited branch: `codex/native-ref-v2-e120-cumulative-step`
- Audited final overlay: `v4-scaled-mixture-final49k-20260724`
- Source license: `anima_edit/LICENSE.md` (Apache-2.0)
- Core extraction archive SHA-256: `4df2ffc7dd5efde322f66696c2607a9aea096b7abab2c608f4412eb8d00d023c`
- Required `networks/` archive SHA-256: `8b4b5067f47649cf7d90c6212a910c8e3d8b5e7573889f80e6fa3d6a6c2a7358`
- Per-file hashes: `VENDOR_MANIFEST.json`

The upstream absolute imports were mechanically namespaced for embedding. At
runtime, the physical `anima_edit/` tree is registered only as the private
package `_anima_native_ref_vendor`; its modules therefore resolve as
`_anima_native_ref_vendor.library.*` and
`_anima_native_ref_vendor.networks.*`. `runtime.py` does **not** add this tree to
`sys.path`, and it neither reads nor populates generic top-level `library` or
`networks` modules. An existing foreign copy of the private package name is
still rejected unless its origin is this exact pinned tree.

`VENDOR_MANIFEST.json` records the upstream commit/base extraction archives,
the pre-transform hashes of every authoritative V4 overlay file, the
deterministic namespace transformation, and hashes for the final embedded
files. The loader verifies every final file before importing code.

The V4-specific contract is implemented by:

- `library/anima_reference_binding.py` (canonical Image 1/Image 2 clauses);
- `library/anima_text_conditioning.py` (raw Qwen + neutral-T5 IDs and CFG
  positive-router sharing);
- `library/anima_reference_router.py` (competitive route);
- the fixed-one/two carrier in `library/anima_models.py`; and
- the shared reference preprocessing helper in `library/strategy_anima.py`.
