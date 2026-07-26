"""Build deterministic UI-v1 and API workflows for every V4 slot case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api_workflows"
UI_DIR = ROOT / "example_workflows"

CHECKPOINT = "anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors"
CHECKPOINT_HF_PATH = f"checkpoints/v4-scaled-49k/{CHECKPOINT}"
CHECKPOINT_SHA256 = (
    "4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f"
)
TEXT_ENCODER = "qwen_3_06b_base.safetensors"
TEXT_ENCODER_SHA256 = (
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
)
VAE = "qwen_image_vae.safetensors"
VAE_SHA256 = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"

SINGLE_SLOT0_PROMPT = (
    "Use Image 1 as the character identity reference. Generate a new illustration."
)
SINGLE_SLOT1_PROMPT = (
    "Use Image 2 as the character identity reference. Generate a new illustration."
)
DUAL_PROMPT = (
    "Use the pose and composition from Image 1; "
    "use the character appearance from Image 2."
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def loader_inputs() -> dict[str, Any]:
    return {
        "checkpoint": CHECKPOINT,
        "text_encoder": TEXT_ENCODER,
        "vae": VAE,
        "offload_mode": "balanced",
        # Published workflows should fail closed if any downloaded release
        # weight is truncated or replaced.  The node-level default remains
        # False so custom/local checkpoints do not incur an unexpected hash
        # pass merely by adding the loader node.
        "verify_release_sha256": True,
    }


def sample_inputs(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "negative_prompt": "",
        "seed": 20260724,
        "width": 256,
        "height": 256,
        "steps": 30,
        "cfg": 3.5,
        "flow_shift": 5.0,
        "reference_scale": 1.0,
        "reference_max_area": 65536,
    }


def api_single(slot_id: int) -> dict[str, Any]:
    prompt = SINGLE_SLOT0_PROMPT if slot_id == 0 else SINGLE_SLOT1_PROMPT
    generate = {
        "pipeline": ["2", 0],
        "reference_image": ["1", 0],
        "logical_slot": f"Image {slot_id + 1} (slot {slot_id})",
        "preprocess_mode": "independent_reference",
        **sample_inputs(prompt),
    }
    return {
        "1": {
            "inputs": {"image": f"reference_image_{slot_id + 1}.png"},
            "class_type": "LoadImage",
            "_meta": {"title": f"Reference Image {slot_id + 1} / slot {slot_id}"},
        },
        "2": {
            "inputs": loader_inputs(),
            "class_type": "AnimaNativeRefV4Loader",
            "_meta": {"title": "Anima Reference V4 Loader (Final 49k)"},
        },
        "3": {
            "inputs": generate,
            "class_type": "AnimaNativeRefV4Generate1Ref",
            "_meta": {"title": f"Anima V4 1Ref / logical slot {slot_id}"},
        },
        "4": {
            "inputs": {"images": ["3", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview"},
        },
        "5": {
            "inputs": {
                "filename_prefix": f"anima_v4_single_slot{slot_id}",
                "images": ["3", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save"},
        },
    }


def api_dual() -> dict[str, Any]:
    return {
        "1": {
            "inputs": {"image": "reference_image_1.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Reference Image 1 / slot 0"},
        },
        "2": {
            "inputs": {"image": "reference_image_2.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Reference Image 2 / slot 1"},
        },
        "3": {
            "inputs": loader_inputs(),
            "class_type": "AnimaNativeRefV4Loader",
            "_meta": {"title": "Anima Reference V4 Loader (Final 49k)"},
        },
        "4": {
            "inputs": {
                "pipeline": ["3", 0],
                "reference_image_1": ["1", 0],
                "reference_image_2": ["2", 0],
                **sample_inputs(DUAL_PROMPT),
            },
            "class_type": "AnimaNativeRefV4Generate2Refs",
            "_meta": {"title": "Anima V4 2Refs / slots 0+1"},
        },
        "5": {
            "inputs": {"images": ["4", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview"},
        },
        "6": {
            "inputs": {
                "filename_prefix": "anima_v4_dual",
                "images": ["4", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save"},
        },
    }


def input_slot(name: str, type_name: str, link: int | None, *, widget=False):
    value: dict[str, Any] = {"name": name, "type": type_name, "link": link}
    if widget:
        value["widget"] = {"name": name}
    return value


def load_image_node(
    node_id: int,
    *,
    slot_id: int,
    link_id: int,
    pos_y: int,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "LoadImage",
        "pos": [20, pos_y],
        "size": [330, 360],
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [link_id],
                "slot_index": 0,
            },
            {
                "name": "MASK",
                "type": "MASK",
                "links": None,
                "slot_index": 1,
            },
        ],
        "title": f"Reference Image {slot_id + 1} — logical slot {slot_id}",
        "properties": {
            "Node name for S&R": "LoadImage",
            "cnr_id": "comfy-core",
        },
        "widgets_values": [f"reference_image_{slot_id + 1}.png", "image"],
    }


def loader_node(node_id: int, link_id: int, *, order: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "AnimaNativeRefV4Loader",
        "pos": [430, 80],
        "size": [430, 270],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            input_slot("checkpoint", "COMBO", None, widget=True),
            input_slot("text_encoder", "COMBO", None, widget=True),
            input_slot("vae", "COMBO", None, widget=True),
            input_slot("offload_mode", "COMBO", None, widget=True),
            input_slot("verify_release_sha256", "BOOLEAN", None, widget=True),
        ],
        "outputs": [
            {
                "name": "pipeline",
                "type": "ANIMA_NATIVE_REF_V4_PIPELINE",
                "links": [link_id],
                "slot_index": 0,
            }
        ],
        "properties": {
            "Node name for S&R": "AnimaNativeRefV4Loader",
            "cnr_id": "ComfyUI-Anima-Native-Reference",
        },
        "widgets_values": [
            CHECKPOINT,
            TEXT_ENCODER,
            VAE,
            "balanced",
            True,
        ],
    }


def output_nodes(generate_id: int, start_id: int, links: tuple[int, int], prefix: str):
    preview_id, save_id = start_id, start_id + 1
    return [
        {
            "id": preview_id,
            "type": "PreviewImage",
            "pos": [1510, 80],
            "size": [410, 410],
            "flags": {},
            "order": preview_id - 1,
            "mode": 0,
            "inputs": [input_slot("images", "IMAGE", links[0])],
            "outputs": [],
            "properties": {
                "Node name for S&R": "PreviewImage",
                "cnr_id": "comfy-core",
            },
            "widgets_values": [],
        },
        {
            "id": save_id,
            "type": "SaveImage",
            "pos": [1510, 540],
            "size": [410, 300],
            "flags": {},
            "order": save_id - 1,
            "mode": 0,
            "inputs": [input_slot("images", "IMAGE", links[1])],
            "outputs": [],
            "properties": {
                "Node name for S&R": "SaveImage",
                "cnr_id": "comfy-core",
            },
            "widgets_values": [prefix],
        },
    ]


def note_node(node_id: int, text: str, *, order: int):
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": [430, 390],
        "size": [430, 430],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {
            "Node name for S&R": "MarkdownNote",
            "cnr_id": "comfy-core",
        },
        "widgets_values": [text],
        "color": "#222",
        "bgcolor": "#000",
    }


def models() -> list[dict[str, str]]:
    return [
        {
            "name": CHECKPOINT,
            # Intended release location.  Building workflows does not upload it.
            "url": (
                "https://huggingface.co/LAXMAYDAY/"
                "NOOB2-Project-Character-Reference-Bypass-Injector-Research/"
                f"resolve/main/{CHECKPOINT_HF_PATH}"
            ),
            "hash": CHECKPOINT_SHA256,
            "hash_type": "sha256",
            "directory": "diffusion_models",
        },
        {
            "name": TEXT_ENCODER,
            "url": (
                "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
                f"split_files/text_encoders/{TEXT_ENCODER}"
            ),
            "hash": TEXT_ENCODER_SHA256,
            "hash_type": "sha256",
            "directory": "text_encoders",
        },
        {
            "name": VAE,
            "url": (
                "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
                f"split_files/vae/{VAE}"
            ),
            "hash": VAE_SHA256,
            "hash_type": "sha256",
            "directory": "vae",
        },
    ]


def ui_base(
    *,
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    name: str,
    description: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "state": {
            "lastGroupid": 0,
            "lastNodeId": max(node["id"] for node in nodes),
            "lastLinkId": max(link["id"] for link in links),
            "lastRerouteId": 0,
        },
        "nodes": nodes,
        "links": links,
        "reroutes": [],
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.72, "offset": [80, 30]},
            "info": {
                "name": name,
                "author": "LAXMAYDAY / engineering reproduction",
                "description": description,
                "version": "0.2.0",
                "created": "2026-07-24",
                "modified": "2026-07-24",
                "software": "ComfyUI",
            },
        },
        "models": models(),
    }


def ui_single(slot_id: int) -> dict[str, Any]:
    prompt = SINGLE_SLOT0_PROMPT if slot_id == 0 else SINGLE_SLOT1_PROMPT
    logical = f"Image {slot_id + 1} (slot {slot_id})"
    nodes = [
        load_image_node(1, slot_id=slot_id, link_id=1, pos_y=80),
        loader_node(2, 2, order=1),
        {
            "id": 3,
            "type": "AnimaNativeRefV4Generate1Ref",
            "pos": [900, 80],
            "size": [550, 830],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
                input_slot("pipeline", "ANIMA_NATIVE_REF_V4_PIPELINE", 2),
                input_slot("reference_image", "IMAGE", 1),
                input_slot("logical_slot", "COMBO", None, widget=True),
                input_slot("preprocess_mode", "COMBO", None, widget=True),
                input_slot("prompt", "STRING", None, widget=True),
                input_slot("negative_prompt", "STRING", None, widget=True),
                input_slot("seed", "INT", None, widget=True),
                input_slot("width", "INT", None, widget=True),
                input_slot("height", "INT", None, widget=True),
                input_slot("steps", "INT", None, widget=True),
                input_slot("cfg", "FLOAT", None, widget=True),
                input_slot("flow_shift", "FLOAT", None, widget=True),
                input_slot("reference_scale", "FLOAT", None, widget=True),
                input_slot("reference_max_area", "INT", None, widget=True),
            ],
            "outputs": [
                {
                    "name": "image",
                    "type": "IMAGE",
                    "links": [3, 4],
                    "slot_index": 0,
                }
            ],
            "properties": {
                "Node name for S&R": "AnimaNativeRefV4Generate1Ref",
                "cnr_id": "ComfyUI-Anima-Native-Reference",
            },
            "widgets_values": [
                logical,
                "independent_reference",
                prompt,
                "",
                20260724,
                "fixed",
                256,
                256,
                30,
                3.5,
                5.0,
                1.0,
                65536,
            ],
        },
        *output_nodes(3, 4, (3, 4), f"anima_v4_single_slot{slot_id}"),
        note_node(
            6,
            (
                "## Anima Native Reference V4 — one reference\n\n"
                f"- One real physical image is assigned to **{logical}**; no "
                "second latent is copied or fabricated.\n"
                "- The prompt must explicitly mention that same numbered image "
                "in a byte-exact canonical clause.\n"
                "- `independent_reference` is the character/reference path. Use "
                "`match_output_edit` only when the source should be fitted to "
                "the requested output geometry.\n"
                "- Raw Qwen feeds the competitive router; neutral-T5 is tokenizer-only.\n"
                "- Pure-noise sampling; there is no img2img denoise strength."
            ),
            order=5,
        ),
    ]
    links = [
        {
            "id": 1,
            "origin_id": 1,
            "origin_slot": 0,
            "target_id": 3,
            "target_slot": 1,
            "type": "IMAGE",
        },
        {
            "id": 2,
            "origin_id": 2,
            "origin_slot": 0,
            "target_id": 3,
            "target_slot": 0,
            "type": "ANIMA_NATIVE_REF_V4_PIPELINE",
        },
        {
            "id": 3,
            "origin_id": 3,
            "origin_slot": 0,
            "target_id": 4,
            "target_slot": 0,
            "type": "IMAGE",
        },
        {
            "id": 4,
            "origin_id": 3,
            "origin_slot": 0,
            "target_id": 5,
            "target_slot": 0,
            "type": "IMAGE",
        },
    ]
    return ui_base(
        nodes=nodes,
        links=links,
        name=f"Anima V4 final — one reference in logical slot {slot_id}",
        description=(
            "Final competitive-router Anima generation with one real reference "
            f"assigned to logical slot {slot_id}."
        ),
    )


def ui_dual() -> dict[str, Any]:
    nodes = [
        load_image_node(1, slot_id=0, link_id=1, pos_y=80),
        load_image_node(2, slot_id=1, link_id=2, pos_y=500),
        loader_node(3, 3, order=2),
        {
            "id": 4,
            "type": "AnimaNativeRefV4Generate2Refs",
            "pos": [900, 80],
            "size": [550, 790],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
                input_slot("pipeline", "ANIMA_NATIVE_REF_V4_PIPELINE", 3),
                input_slot("reference_image_1", "IMAGE", 1),
                input_slot("reference_image_2", "IMAGE", 2),
                input_slot("prompt", "STRING", None, widget=True),
                input_slot("negative_prompt", "STRING", None, widget=True),
                input_slot("seed", "INT", None, widget=True),
                input_slot("width", "INT", None, widget=True),
                input_slot("height", "INT", None, widget=True),
                input_slot("steps", "INT", None, widget=True),
                input_slot("cfg", "FLOAT", None, widget=True),
                input_slot("flow_shift", "FLOAT", None, widget=True),
                input_slot("reference_scale", "FLOAT", None, widget=True),
                input_slot("reference_max_area", "INT", None, widget=True),
            ],
            "outputs": [
                {
                    "name": "image",
                    "type": "IMAGE",
                    "links": [4, 5],
                    "slot_index": 0,
                }
            ],
            "properties": {
                "Node name for S&R": "AnimaNativeRefV4Generate2Refs",
                "cnr_id": "ComfyUI-Anima-Native-Reference",
            },
            "widgets_values": [
                DUAL_PROMPT,
                "",
                20260724,
                "fixed",
                256,
                256,
                30,
                3.5,
                5.0,
                1.0,
                65536,
            ],
        },
        *output_nodes(4, 5, (4, 5), "anima_v4_dual"),
        note_node(
            7,
            (
                "## Anima Native Reference V4 — two references\n\n"
                "- Image 1 is logical slot 0; Image 2 is logical slot 1.\n"
                "- Slots have no hard-coded scene/identity/style role. Separate "
                "canonical clauses say what to use from each.\n"
                "- Both images preserve their independent aspect ratios under "
                "the area limit.\n"
                "- CFG negative ordinary text stays negative, while its router "
                "uses the positive Image 1/Image 2 clauses.\n"
                "- One self-contained checkpoint; no LoRA/adapter sidecar."
            ),
            order=6,
        ),
    ]
    links = [
        {
            "id": 1,
            "origin_id": 1,
            "origin_slot": 0,
            "target_id": 4,
            "target_slot": 1,
            "type": "IMAGE",
        },
        {
            "id": 2,
            "origin_id": 2,
            "origin_slot": 0,
            "target_id": 4,
            "target_slot": 2,
            "type": "IMAGE",
        },
        {
            "id": 3,
            "origin_id": 3,
            "origin_slot": 0,
            "target_id": 4,
            "target_slot": 0,
            "type": "ANIMA_NATIVE_REF_V4_PIPELINE",
        },
        {
            "id": 4,
            "origin_id": 4,
            "origin_slot": 0,
            "target_id": 5,
            "target_slot": 0,
            "type": "IMAGE",
        },
        {
            "id": 5,
            "origin_id": 4,
            "origin_slot": 0,
            "target_id": 6,
            "target_slot": 0,
            "type": "IMAGE",
        },
    ]
    return ui_base(
        nodes=nodes,
        links=links,
        name="Anima V4 final — two references in logical slots 0+1",
        description=(
            "Final competitive-router Anima generation with two independently "
            "preprocessed references and explicit prompt binding."
        ),
    )


def main() -> None:
    outputs = {
        API_DIR / "anima_ref_v4_final_single_slot0_api.json": api_single(0),
        API_DIR / "anima_ref_v4_final_single_slot1_api.json": api_single(1),
        API_DIR / "anima_ref_v4_final_dual_api.json": api_dual(),
        UI_DIR / "anima_ref_v4_final_single_slot0.json": ui_single(0),
        UI_DIR / "anima_ref_v4_final_single_slot1.json": ui_single(1),
        UI_DIR / "anima_ref_v4_final_dual.json": ui_dual(),
    }
    for path, graph in outputs.items():
        write_json(path, graph)
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
