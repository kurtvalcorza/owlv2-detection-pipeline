"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on owlv2_detection_colab (OWD-*).

They need only CI's dependencies; the model-backed acceptance check for OWD-M2 (two adaptations, the same frozen
epoch 0) lives in tests/test_model_backed.py and runs where the snapshot is staged.
"""
# ruff: noqa: E501  -- notebook source fragments are kept on single lines

from __future__ import annotations

import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from owlv2_detection_pipeline import Owlv2DetectionPipeline, min_split_images, split_dataset, validate_dataset
from owlv2_detection_pipeline.samples import MIN_RECORDS, load_byod_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load("notebook_template_review", ROOT / "tools" / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [s for s in _code(nb) if marker in s]
    assert len(found) == 1, marker
    return found[0]


def _records(n: int) -> list[dict]:
    """n distinct 32 x 32 images, one box each."""
    return [
        {"id": f"r{i:03d}", "image": Image.new("RGB", (32, 32), (i % 256, (i * 7) % 256, (i * 13) % 256)), "boxes": [{"prompt": "a cell", "box": [2, 2, 20, 20]}]}
        for i in range(n)
    ]


# --- OWD-M2: adapt() no longer leaves tuned heads behind as the "frozen" model -------------------------------------


def test_owd_m2_reset_to_base_restores_the_frozen_heads_and_drops_the_adapter() -> None:
    torch = pytest.importorskip("torch")

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.class_head = torch.nn.Linear(3, 2)
            self.box_head = torch.nn.Linear(3, 4)
            self.tower = torch.nn.Linear(3, 3)

    model = _Model()
    pipe = Owlv2DetectionPipeline(None, "cpu", model, object(), None, None)  # type: ignore[arg-type]
    pipe._base_heads = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(("class_head.", "box_head."))}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    tower = model.tower.weight.detach().clone()
    pipe.adapter = {"best_epoch": 3}
    assert pipe.reset_to_base() == {"reset": True, "was_adapted": True}
    assert pipe.adapter is None
    assert all(torch.equal(p, pipe._base_heads[n]) for n, p in model.named_parameters() if n in pipe._base_heads)
    assert torch.equal(model.tower.weight, tower)  # only the heads are restored
    assert pipe.reset_to_base() == {"reset": True, "was_adapted": False}


def test_owd_m2_reset_refuses_without_a_head_snapshot() -> None:
    pipe = Owlv2DetectionPipeline(None, "cpu", object(), object(), None, {"best_epoch": 1})  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no frozen head snapshot"):
        pipe.reset_to_base()


def test_owd_m2_adapt_starts_from_the_captured_frozen_heads() -> None:
    source = (ROOT / "src" / "owlv2_detection_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    body = source.split("    def adapt(", 1)[1].split("    def save_artifact(", 1)[0]
    reset = body.index("param.copy_(self._base_heads[name])")
    assert reset < body.index('"note": "frozen model"'), "epoch 0 must be scored after the frozen heads are restored"
    assert "pipe._base_heads = {" in source.split("    def from_pretrained(", 1)[1].split("    def reset_to_base(", 1)[0]


def test_owd_m2_section6_refuses_to_label_an_adapted_pipeline_frozen(nb: dict) -> None:
    source = _cell_with(nb, "frozen_test = pipe.evaluate(")
    assert source.index("pipe.reset_to_base()") < source.index("frozen_test = pipe.evaluate(")
    assert "if frozen_test['adapted']:" in source and "'adapted': frozen_test['adapted']" in source


def test_owd_m2_experiments_name_the_cells_to_rerun(nb: dict) -> None:
    md = _markdown(nb)
    assert "run Sections 6–9 again" in md and "run Sections 5–9 again" in md and "run Sections 4–9 again" in md
    assert "before Section 6" not in md  # THRESHOLD lives in Section 5


# --- OWD-M3: comparisons are verdicts, not asserts ----------------------------------------------------------------


def test_owd_m3_no_ranking_assert_remains(nb: dict) -> None:
    code = "\n".join(_code(nb))
    assert not re.search(r"assert adapted_test\['map50'\]", code)
    assert "'verdicts': {'adapted_map50_at_least_frozen'" in code


# --- OWD-m1: the BYOD minimum is stated and enforced with a named split ---------------------------------------------


def test_owd_m1_min_split_images_matches_split_dataset() -> None:
    need = min_split_images()
    assert need == 12
    splits = split_dataset(_records(need), seed=0)
    assert len(splits["train"]) >= MIN_RECORDS and len(splits["validation"]) >= 1 and len(splits["test"]) >= 1
    for name, part in splits.items():
        validate_dataset(part, min_records=MIN_RECORDS if name == "train" else 1)
    with pytest.raises(ValueError, match=r"the train split has 7 records and needs at least 8.*at least 12 distinct images"):
        split_dataset(_records(need - 1), seed=0)


def test_owd_m1_notebook_validates_each_split_and_names_it(nb: dict) -> None:
    source = _cell_with(nb, "dataset_manifests = {}")
    assert "validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1)" in source
    assert "f'{name} split: {exc}" in source and "min_split_images()" in source
    md = _markdown(nb)
    assert "at least 12 distinct images" in md and "at least eight images" not in md


# --- OWD-m2: no stale candidate / queued-run wording ----------------------------------------------------------------


def test_owd_m2_minor_no_stale_status_text(nb: dict) -> None:
    text = _markdown(nb) + "\n".join(_code(nb))
    for stale in ("queued", "not yet been recorded", "source-only build", "This candidate", "on the queued clean-runtime run will"):
        assert stale not in text, stale
    assert "**Timing (measured):**" in text


# --- OWD-m3: the panel text matches the two panels drawn -----------------------------------------------------------


def test_owd_m3_minor_panel_text_matches_panels(nb: dict) -> None:
    md = _markdown(nb)
    assert "the frozen detections and the adapted detections side by side" not in md
    assert "two-panel sheets" in md
    code = _cell_with(nb, "'panels': ['reference boxes', 'adapted detections']")
    assert code.count("draw_boxes(record['image']") == 2


# --- OWD-m4: BYOD reads a folder without google.colab ----------------------------------------------------------------


def test_owd_m4_byod_folder_path_is_read_in_place(nb: dict, tmp_path: Path) -> None:
    source = _cell_with(nb, "byod_dir = Path(")
    line = next(ln for ln in source.splitlines() if ln.strip().startswith("byod_zip = byod_dir if"))
    folder = tmp_path / "kaggle_input"
    folder.mkdir()
    rows = ["file,prompt,x0,y0,x1,y1"]
    for i, record in enumerate(_records(12)):
        record["image"].save(folder / f"img{i:02d}.png")
        rows.append(f"img{i:02d}.png,a cell,2,2,20,20")
    (folder / "boxes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _no_upload(*args, **kwargs):
        raise AssertionError("a folder path must not reach byod_file / google.colab")

    ns = {"Path": Path, "BYOD_PATH": str(folder), "byod_file": _no_upload}
    exec("byod_dir = Path(str(BYOD_PATH).strip()).expanduser() if str(BYOD_PATH).strip() else None\n" + line.strip(), ns)
    assert ns["byod_zip"] == folder
    records = load_byod_dataset(ns["byod_zip"])
    assert len(split_dataset(records, seed=0)["train"]) == 8


def test_owd_m4_zip_byod_still_reads(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        rows = ["file,prompt,x0,y0,x1,y1"]
        for i, record in enumerate(_records(12)):
            out = io.BytesIO()
            record["image"].save(out, format="PNG")
            archive.writestr(f"img{i:02d}.png", out.getvalue())
            rows.append(f"img{i:02d}.png,a cell,2,2,20,20")
        archive.writestr("boxes.csv", "\n".join(rows) + "\n")
    path = tmp_path / "byod.zip"
    path.write_bytes(buf.getvalue())
    assert len(load_byod_dataset(path)) == 12


# --- OWD-m5 + lock coverage: pyarrow and scipy are pinned runtime dependencies --------------------------------------


def test_owd_m5_pyarrow_and_scipy_are_runtime_pins_and_locked(nb: dict) -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", pyproject, re.M | re.S).group(1)
    assert '"pyarrow==25.0.1"' in deps and '"scipy==1.18.1"' in deps
    lock = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    for name in ("pyarrow==25.0.1", "scipy==1.18.1"):
        assert re.search(rf"^{re.escape(name)} \\$", lock, re.M), name
    runtime = _cell_with(nb, "'pyarrow': pyarrow.__version__")
    assert "'scipy': scipy.__version__" in runtime


# --- OWD-m7: spec 2.2 declared --------------------------------------------------------------------------------------


def test_owd_m7_declares_notebook_spec_2_2(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "DIMER Notebook Specification 2.2" in _markdown(nb)


# --- OWD-m8: the printed weight source is real ---------------------------------------------------------------------


def test_owd_m8_load_cell_prints_the_verified_source(nb: dict) -> None:
    code = "\n".join(_code(nb))
    assert "'local-snapshot'" not in code
    assert "'weights_dir': str(WEIGHTS_DIR), 'weight_sha256': getattr(pipe, 'weight_sha256', None)" in code
