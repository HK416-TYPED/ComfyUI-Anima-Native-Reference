"""Regenerate the strict embedded-runtime manifest deterministically."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "anima_edit"
MANIFEST_PATH = PACKAGE_ROOT / "vendor" / "VENDOR_MANIFEST.json"

SOURCE_COMMIT = "2ae811d296ff4159c6024c4a86415d19961a388c"
SOURCE_OVERLAY = "v4-scaled-mixture-final49k-20260724"

AUTHORITATIVE_OVERLAY_FILES = {
    "anima_minimal_inference.py": (
        "9aaf1b59e0820a1eb855e550341c79d184403c705dc15a137469061014b76c03"
    ),
    "library/anima_models.py": (
        "ec9df3c0322ab4b24e077066e0f6ebc96d33bcd7c10e2ffa20cf325def0bd60b"
    ),
    "library/anima_reference_binding.py": (
        "874d716914d5feaab3b6d75e680134c2dd56a6380d7e3faf461c54e7da86e723"
    ),
    "library/anima_reference_router.py": (
        "71720ad923d661d4fad08e5436cc246602b9864b943a5c5d3cbf27409cfb8fdd"
    ),
    "library/anima_text_conditioning.py": (
        "8c9512c74fc174db7fda2cf9940cb55e0230b57ba43d5dadd049f1181e0f4ed8"
    ),
    "library/anima_utils.py": (
        "69f288613b2e264ef37c7599c26452469ede4fbce7b577d6bbfb8b9dacf01aeb"
    ),
    "library/strategy_anima.py": (
        "41faf719292fd8e4dd566518f3789ff2872bd2cd0a75f4c721d6c662e6fc8643"
    ),
    "library/train_util.py": (
        "9909d1fbb3278ff956176a13bde8abe4b694569dc4b993b0e14bdd27e7033a25"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted(
        (
            path
            for path in VENDOR_ROOT.rglob("*")
            if path.is_file()
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(VENDOR_ROOT).as_posix(),
    )
    files = [
        {
            "path": path.relative_to(VENDOR_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    tree_payload = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in files
    ).encode("utf-8")
    manifest = {
        "source_repository": "https://github.com/akatsuki-neo/anima-edit.git",
        "source_commit": SOURCE_COMMIT,
        "source_branch": "codex/native-ref-v2-e120-cumulative-step",
        "source_overlay": SOURCE_OVERLAY,
        "source_overlay_note": (
            "Final competitive text-slot-router inference snapshot used by the "
            "formal scaled-mixture 49k run; hashes below are before the mechanical "
            "private-namespace import transform."
        ),
        "overlay_authoritative_files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(AUTHORITATIVE_OVERLAY_FILES.items())
        ],
        "source_archives": [
            {
                "scope": "base library, configs, inference, license, requirements",
                "sha256": (
                    "4df2ffc7dd5efde322f66696c2607a9aea096b7abab2c608f4412eb8d00d023c"
                ),
            },
            {
                "scope": "base networks",
                "sha256": (
                    "8b4b5067f47649cf7d90c6212a910c8e3d8b5e7573889f80e6fa3d6a6c2a7358"
                ),
            },
        ],
        "import_namespace": "_anima_native_ref_vendor",
        "vendoring_transforms": [
            {
                "type": "absolute_import_namespace",
                "source_packages": ["library", "networks"],
                "target_prefix": "_anima_native_ref_vendor",
                "scope": "Python import statements in the embedded source tree",
            },
            {
                "type": "newline_normalization",
                "value": "LF",
                "scope": "authoritative overlay Python files",
            },
            {
                "type": "package_initializers",
                "paths": ["__init__.py", "networks/__init__.py"],
            },
        ],
        "vendored_tree_hash_algorithm": (
            "sha256(concat(sorted('<file_sha256>  <posix_path>\\n')))"
        ),
        "vendored_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "file_count": len(files),
        "files": files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {MANIFEST_PATH} ({len(files)} files, "
        f"tree={manifest['vendored_tree_sha256']})"
    )


if __name__ == "__main__":
    main()
