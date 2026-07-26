from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = "anima-native-ref-v2-e180-step64080-256px.safetensors"
TEXT_ENCODER = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
V4_CHECKPOINT = "anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors"
V4_SHA256 = "4500a4aad657e0d8e821607afe050b09931f84bf601ea1447ce2a52cca782e2f"
V4_HF_URL = (
    "https://huggingface.co/LAXMAYDAY/"
    "NOOB2-Project-Character-Reference-Bypass-Injector-Research/resolve/main/"
    "checkpoints/v4-scaled-49k/"
    "anima-v4-scaled-mix-50-30-20-e2-step49000-256area.safetensors"
)


def _load_plugin(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    names = {
        "diffusion_models": [CHECKPOINT],
        "text_encoders": [TEXT_ENCODER],
        "vae": [VAE],
    }
    folder_paths.get_filename_list = lambda category: list(names[category])
    folder_paths.get_full_path_or_raise = lambda category, filename: str(ROOT / filename)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    package_name = "_anima_native_ref_node_test"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(package_name + "."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)
    return package


def test_node_registration_and_formal_defaults(monkeypatch):
    package = _load_plugin(monkeypatch)
    assert set(package.NODE_CLASS_MAPPINGS) == {
        "AnimaNativeRefV2Loader",
        "AnimaNativeRefV2Generate",
        "AnimaNativeRefV4Loader",
        "AnimaNativeRefV4Generate1Ref",
        "AnimaNativeRefV4Generate2Refs",
    }

    loader = package.NODE_CLASS_MAPPINGS["AnimaNativeRefV2Loader"]
    generator = package.NODE_CLASS_MAPPINGS["AnimaNativeRefV2Generate"]
    assert loader.RETURN_TYPES == ("ANIMA_NATIVE_REF_V2_PIPELINE",)
    assert generator.RETURN_TYPES == ("IMAGE",)

    loader_inputs = loader.INPUT_TYPES()
    assert loader_inputs["required"]["checkpoint"][0] == [CHECKPOINT]
    assert loader_inputs["required"]["text_encoder"][0] == [TEXT_ENCODER]
    assert loader_inputs["required"]["vae"][0] == [VAE]
    assert "verify_release_sha256" in loader_inputs["optional"]

    required = generator.INPUT_TYPES()["required"]
    assert required["width"][1]["default"] == 256
    assert required["height"][1]["default"] == 256
    assert required["steps"][1]["default"] == 40
    assert required["cfg"][1]["default"] == 1.0
    assert required["flow_shift"][1]["default"] == 5.0
    assert required["reference_scale"][1]["default"] == 1.0
    assert required["reference_max_area"][1]["default"] == 65536
    assert "denoise" not in required

    v4_loader = package.NODE_CLASS_MAPPINGS["AnimaNativeRefV4Loader"]
    v4_single = package.NODE_CLASS_MAPPINGS["AnimaNativeRefV4Generate1Ref"]
    v4_dual = package.NODE_CLASS_MAPPINGS["AnimaNativeRefV4Generate2Refs"]
    assert v4_loader.RETURN_TYPES == ("ANIMA_NATIVE_REF_V4_PIPELINE",)
    assert v4_single.RETURN_TYPES == v4_dual.RETURN_TYPES == ("IMAGE",)

    single = v4_single.INPUT_TYPES()["required"]
    assert single["logical_slot"][0] == [
        "Image 1 (slot 0)",
        "Image 2 (slot 1)",
    ]
    assert single["preprocess_mode"][0] == [
        "independent_reference",
        "match_output_edit",
    ]
    assert single["prompt"][1]["default"] == (
        "Use Image 1 as the character identity reference. "
        "Generate a new illustration."
    )
    assert single["steps"][1]["default"] == 30
    assert single["cfg"][1]["default"] == 3.5
    assert single["flow_shift"][1]["default"] == 5.0
    assert single["reference_scale"][1]["default"] == 1.0
    assert single["reference_max_area"][1]["default"] == 65536
    assert "reference_image_2" not in single
    assert "denoise" not in single

    dual = v4_dual.INPUT_TYPES()["required"]
    assert dual["prompt"][1]["default"] == (
        "Use the pose and composition from Image 1; "
        "use the character appearance from Image 2."
    )
    assert dual["steps"][1]["default"] == 30
    assert dual["cfg"][1]["default"] == 3.5
    assert "preprocess_mode" not in dual
    assert "logical_slot" not in dual
    assert "denoise" not in dual


def _node_by_id(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


def test_ui_workflow_v1_graph_and_model_contract():
    path = ROOT / "example_workflows" / "anima_ref_v2_e180.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 1
    assert {node["type"] for node in workflow["nodes"]} >= {
        "LoadImage",
        "AnimaNativeRefV2Loader",
        "AnimaNativeRefV2Generate",
        "PreviewImage",
        "SaveImage",
    }

    node_ids = [node["id"] for node in workflow["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    link_ids = [link["id"] for link in workflow["links"]]
    assert len(link_ids) == len(set(link_ids))
    for link in workflow["links"]:
        origin = nodes[link["origin_id"]]
        target = nodes[link["target_id"]]
        assert origin["outputs"][link["origin_slot"]]["type"] == link["type"]
        assert target["inputs"][link["target_slot"]]["type"] == link["type"]
        assert target["inputs"][link["target_slot"]]["link"] == link["id"]
        assert link["id"] in origin["outputs"][link["origin_slot"]]["links"]

    loader = _node_by_id(workflow, 3)
    generate = _node_by_id(workflow, 4)
    assert loader["widgets_values"] == [CHECKPOINT, TEXT_ENCODER, VAE, "balanced", False]
    # seed-control mode is inserted immediately after the seed in Comfy workflows.
    assert generate["widgets_values"][3:11] == [
        "fixed",
        256,
        256,
        40,
        1.0,
        5.0,
        1.0,
        65536,
    ]
    assert generate["inputs"][0]["name"] == "pipeline"
    assert generate["inputs"][1]["name"] == "reference_image_1"
    assert generate["inputs"][2]["name"] == "reference_image_2"

    models = {model["name"]: model for model in workflow["models"]}
    assert models[CHECKPOINT]["hash"] == (
        "1f970a7867dd7b65858d30b58135134ce84fd3b552ab27fc9f07e7f15209c6dd"
    )
    assert models[TEXT_ENCODER]["hash"] == (
        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
    )
    assert models[VAE]["hash"] == (
        "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
    )


def test_api_workflow_connections_and_formal_defaults():
    path = ROOT / "api_workflows" / "anima_ref_v2_e180_api.json"
    graph = json.loads(path.read_text(encoding="utf-8"))
    assert graph["3"]["class_type"] == "AnimaNativeRefV2Loader"
    assert graph["4"]["class_type"] == "AnimaNativeRefV2Generate"
    assert graph["4"]["inputs"]["pipeline"] == ["3", 0]
    assert graph["4"]["inputs"]["reference_image_1"] == ["1", 0]
    assert graph["4"]["inputs"]["reference_image_2"] == ["2", 0]
    assert graph["5"]["inputs"]["images"] == ["4", 0]
    assert graph["6"]["inputs"]["images"] == ["4", 0]

    loader = graph["3"]["inputs"]
    assert loader == {
        "checkpoint": CHECKPOINT,
        "text_encoder": TEXT_ENCODER,
        "vae": VAE,
        "offload_mode": "balanced",
        "verify_release_sha256": False,
    }
    generate = graph["4"]["inputs"]
    assert generate["width"] == generate["height"] == 256
    assert generate["steps"] == 40
    assert generate["cfg"] == 1.0
    assert generate["flow_shift"] == 5.0
    assert generate["reference_scale"] == 1.0
    assert generate["reference_max_area"] == 65536
    assert not any("denoise" in key for key in generate)

    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph


def _assert_ui_links_are_consistent(workflow: dict) -> None:
    assert workflow["version"] == 1
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    link_ids = [link["id"] for link in workflow["links"]]
    assert len(link_ids) == len(set(link_ids))
    for link in workflow["links"]:
        origin = nodes[link["origin_id"]]
        target = nodes[link["target_id"]]
        assert origin["outputs"][link["origin_slot"]]["type"] == link["type"]
        assert target["inputs"][link["target_slot"]]["type"] == link["type"]
        assert target["inputs"][link["target_slot"]]["link"] == link["id"]
        assert link["id"] in origin["outputs"][link["origin_slot"]]["links"]


@pytest.mark.parametrize(
    ("stem", "slot_id"),
    [
        ("anima_ref_v4_final_single_slot0", 0),
        ("anima_ref_v4_final_single_slot1", 1),
    ],
)
def test_v4_single_ui_and_api_workflows(stem, slot_id):
    ui = json.loads(
        (ROOT / "example_workflows" / f"{stem}.json").read_text(encoding="utf-8")
    )
    _assert_ui_links_are_consistent(ui)
    node_types = {node["type"] for node in ui["nodes"]}
    assert {
        "LoadImage",
        "AnimaNativeRefV4Loader",
        "AnimaNativeRefV4Generate1Ref",
        "PreviewImage",
        "SaveImage",
    } <= node_types
    generate = next(
        node for node in ui["nodes"] if node["type"] == "AnimaNativeRefV4Generate1Ref"
    )
    loader = next(
        node for node in ui["nodes"] if node["type"] == "AnimaNativeRefV4Loader"
    )
    assert loader["widgets_values"][-1] is True
    assert generate["inputs"][0]["type"] == "ANIMA_NATIVE_REF_V4_PIPELINE"
    assert generate["inputs"][1]["name"] == "reference_image"
    assert generate["widgets_values"][0] == f"Image {slot_id + 1} (slot {slot_id})"
    assert generate["widgets_values"][1] == "independent_reference"
    assert f"Image {slot_id + 1}" in generate["widgets_values"][2]
    assert generate["widgets_values"][8:13] == [30, 3.5, 5.0, 1.0, 65536]
    assert "denoise" not in {item["name"] for item in generate["inputs"]}

    models = {model["name"]: model for model in ui["models"]}
    assert models[V4_CHECKPOINT]["hash"] == V4_SHA256
    assert models[V4_CHECKPOINT]["directory"] == "diffusion_models"
    assert models[V4_CHECKPOINT]["url"] == V4_HF_URL

    api = json.loads(
        (ROOT / "api_workflows" / f"{stem}_api.json").read_text(encoding="utf-8")
    )
    assert api["2"]["class_type"] == "AnimaNativeRefV4Loader"
    assert api["2"]["inputs"]["checkpoint"] == V4_CHECKPOINT
    assert api["2"]["inputs"]["verify_release_sha256"] is True
    assert api["3"]["class_type"] == "AnimaNativeRefV4Generate1Ref"
    assert api["3"]["inputs"]["pipeline"] == ["2", 0]
    assert api["3"]["inputs"]["reference_image"] == ["1", 0]
    assert api["3"]["inputs"]["logical_slot"] == (
        f"Image {slot_id + 1} (slot {slot_id})"
    )
    assert api["3"]["inputs"]["preprocess_mode"] == "independent_reference"
    assert api["3"]["inputs"]["steps"] == 30
    assert api["3"]["inputs"]["cfg"] == 3.5
    assert "denoise" not in api["3"]["inputs"]


def test_v4_dual_ui_and_api_workflows():
    ui = json.loads(
        (ROOT / "example_workflows" / "anima_ref_v4_final_dual.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_ui_links_are_consistent(ui)
    generate = next(
        node for node in ui["nodes"] if node["type"] == "AnimaNativeRefV4Generate2Refs"
    )
    loader = next(
        node for node in ui["nodes"] if node["type"] == "AnimaNativeRefV4Loader"
    )
    assert loader["widgets_values"][-1] is True
    assert [item["name"] for item in generate["inputs"][:3]] == [
        "pipeline",
        "reference_image_1",
        "reference_image_2",
    ]
    assert "Image 1" in generate["widgets_values"][0]
    assert "Image 2" in generate["widgets_values"][0]
    assert generate["widgets_values"][6:11] == [30, 3.5, 5.0, 1.0, 65536]
    assert "denoise" not in {item["name"] for item in generate["inputs"]}
    models = {model["name"]: model for model in ui["models"]}
    assert models[V4_CHECKPOINT]["hash"] == V4_SHA256
    assert models[V4_CHECKPOINT]["url"] == V4_HF_URL

    api = json.loads(
        (ROOT / "api_workflows" / "anima_ref_v4_final_dual_api.json").read_text(
            encoding="utf-8"
        )
    )
    assert api["3"]["class_type"] == "AnimaNativeRefV4Loader"
    assert api["3"]["inputs"]["checkpoint"] == V4_CHECKPOINT
    assert api["3"]["inputs"]["verify_release_sha256"] is True
    assert api["4"]["class_type"] == "AnimaNativeRefV4Generate2Refs"
    assert api["4"]["inputs"]["pipeline"] == ["3", 0]
    assert api["4"]["inputs"]["reference_image_1"] == ["1", 0]
    assert api["4"]["inputs"]["reference_image_2"] == ["2", 0]
    assert api["4"]["inputs"]["steps"] == 30
    assert api["4"]["inputs"]["cfg"] == 3.5
    assert "Image 1" in api["4"]["inputs"]["prompt"]
    assert "Image 2" in api["4"]["inputs"]["prompt"]
    assert "preprocess_mode" not in api["4"]["inputs"]
    assert "denoise" not in api["4"]["inputs"]
