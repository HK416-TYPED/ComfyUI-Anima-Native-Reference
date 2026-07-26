"""ComfyUI nodes for legacy V2 E180 and final competitive-router V4.

The package exports the classic/V1 ComfyUI registration dictionaries for the
broadest compatibility.  Existing V2 class IDs are intentionally retained.
Model weights are never loaded during import.
"""

if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # pragma: no cover - pytest may collect this hyphenated folder as __init__
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
