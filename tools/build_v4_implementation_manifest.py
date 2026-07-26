"""Write hashes for the V4 implementation deliverables (never model weights)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "V4_IMPLEMENTATION_MANIFEST.json"

INCLUDED = [
    "__init__.py",
    "nodes.py",
    "runtime.py",
    "requirements.txt",
    "README.md",
    "RUNTIME_API.md",
    "TEST_REPORT.md",
    "vendor/README.md",
    "vendor/VENDOR_MANIFEST.json",
    "api_workflows/anima_ref_v4_final_single_slot0_api.json",
    "api_workflows/anima_ref_v4_final_single_slot1_api.json",
    "api_workflows/anima_ref_v4_final_dual_api.json",
    "example_workflows/anima_ref_v4_final_single_slot0.json",
    "example_workflows/anima_ref_v4_final_single_slot1.json",
    "example_workflows/anima_ref_v4_final_dual.json",
    "tests/test_runtime.py",
    "tests/test_runtime_v4.py",
    "tests/test_nodes_and_workflows.py",
    "tests/test_requirements_contract.py",
    "tools/build_vendor_manifest.py",
    "tools/build_v4_workflows.py",
    "tools/build_v4_implementation_manifest.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    vendor = json.loads(
        (ROOT / "vendor" / "VENDOR_MANIFEST.json").read_text(encoding="utf-8")
    )
    files = []
    for relative in INCLUDED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "format": "anima-native-reference-v4-comfy-implementation-manifest-v1",
        "created": "2026-07-24",
        "scope": "isolated implementation; no model weights and no HF upload",
        "final_checkpoint": {
            "filename": (
                "anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors"
            ),
            "hf_path": (
                "checkpoints/v4-scaled-49k/"
                "anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors"
            ),
            "size": 4_302_295_014,
            "sha256": (
                "4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f"
            ),
            "tensor_count": 1_614,
            "native_tensor_count": 926,
        },
        "validated_sampling_defaults": {
            "width": 256,
            "height": 256,
            "steps": 30,
            "cfg": 3.5,
            "flow_shift": 5.0,
            "native_reference_scale": 1.0,
            "reference_max_area": 65_536,
        },
        "vendor": {
            "source_commit": vendor["source_commit"],
            "source_overlay": vendor["source_overlay"],
            "file_count": vendor["file_count"],
            "vendored_tree_sha256": vendor["vendored_tree_sha256"],
        },
        "node_class_ids": [
            "AnimaNativeRefV2Loader",
            "AnimaNativeRefV2Generate",
            "AnimaNativeRefV4Loader",
            "AnimaNativeRefV4Generate1Ref",
            "AnimaNativeRefV4Generate2Refs",
        ],
        "verification": {
            "pytest": "39 passed",
            "ruff": "All checks passed",
            "compileall": "passed",
            "direct_dependency_clean_venv": (
                "PASS: imports absent before install; toml 0.10.2 and "
                "imagesize 1.5.0 import after install"
            ),
            "real_cuda_v4_pixel_parity": (
                "PASS: slot0/slot1/dual CFG3.5 pixel exact to accepted "
                "minimal CLI; CFG1.0 finite/nonconstant"
            ),
            "live_comfy_api": (
                "PASS: dual twice at 30 steps/CFG3.5; loader/input cache, "
                "Unicode prompt/history/filename, save and cleanup verified"
            ),
            "live_comfy_single_slots": (
                "PASS: published slot0 and slot1 API workflows at "
                "30 steps/CFG3.5; release SHA verification, Unicode "
                "prompt/history, logical slots, save, cache and cleanup verified"
            ),
            "legacy_v2_pixel_regression": (
                "PASS: real /prompt at legacy 40 steps/CFG1.0 is pixel exact"
            ),
        },
        "acceptance_evidence": {
            "evidence_set": "anima_v4_final49k_comfy_preflight_20260724_r4",
            "direct_node_report_sha256": (
                "aee23d5f7e5ba210ba2c6644bdf8a486"
                "882bb2ef6f409058984aa106bb2c79b0"
            ),
            "live_api_report_sha256": (
                "6c465ae4a0fab9a5d7ecd4b3072027f36"
                "a985261045c7fd05b9a5d2edcb77d73"
            ),
            "live_object_info_report_sha256": (
                "57d7748164ccf62496bdb7343877d58b2"
                "a131739334fae2f66fada1c73ccd0d6"
            ),
            "live_server_cache_report_sha256": (
                "ff2ed7280bd15451fc7efbea6a1c4c685"
                "cc973e16adcc065e843a77cd76fb796"
            ),
            "live_single_slots_report_sha256": (
                "b12ac2a19a762f3e4e59a0fd0d021051"
                "6f353551297ebc36e63263296cb0ac9b"
            ),
            "legacy_v2_report_sha256": (
                "c3ca5d3c5138222d61a6fc6049358615"
                "000379eb60471719d94fa0cb8a48eda4"
            ),
        },
        "files": files,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {OUTPUT} ({len(files)} hashed files)")


if __name__ == "__main__":
    main()
