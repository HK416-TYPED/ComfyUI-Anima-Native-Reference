from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_only_direct_dependencies_are_declared_and_importable() -> None:
    """Guard direct imports that are not guaranteed by ComfyUI itself."""

    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "toml>=0.10.2,<1" in requirements
    assert "imagesize>=1.4.1,<2" in requirements

    # This exercises the post-install state.  The separate clean-venv preflight
    # records that both imports fail before these two direct requirements are
    # installed and succeed afterwards.
    assert importlib.import_module("toml") is not None
    assert importlib.import_module("imagesize") is not None
