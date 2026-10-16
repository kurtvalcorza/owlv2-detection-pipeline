"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on owlv2_detection_colab (OWD-*).

They need only CI's dependencies; the model-backed acceptance check for OWD-M2 (two adaptations, the same frozen
epoch 0) lives in tests/test_model_backed.py and runs where the snapshot is staged.
"""
# ruff: noqa: E501  -- notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import sys
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


# --- OWD-m7: spec declared (2.2 then; migrated to 2.5 by review OWL-m3, 2026-10-16) ------------------------------


def test_owd_m7_declares_notebook_spec_2_5(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.5"
    assert "DIMER Notebook Specification 2.5" in _markdown(nb)


# --- OWD-m8: the printed weight source is real ---------------------------------------------------------------------


def test_owd_m8_load_cell_prints_the_verified_source(nb: dict) -> None:
    code = "\n".join(_code(nb))
    assert "'local-snapshot'" not in code
    assert "'weights_dir': str(WEIGHTS_DIR), 'weight_sha256': getattr(pipe, 'weight_sha256', None)" in code


# === 2026-10-16 review (OWL-*): DAT9, REL13 labels, spec 2.5, DAT20 refusals, EXE5, float16 cache wording ==========


def test_owl_m1_no_novelty_claim_and_pretraining_overlap_stated(nb: dict) -> None:
    md = _markdown(nb)
    assert not re.search(r"\bnever (?:saw|seen)\b", md, re.I)
    assert "has not been audited and cannot be ruled" in md
    assert "web image-text pairs" in md


def test_owl_m2_worked_answers_credit_the_recorded_run_of_their_blob(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("still needs its own", "25 September 2026", "705 s", "previous notebook version", "this version removes it"):
        assert stale not in md, stale
    assert "8 October 2026" in md and "`a545a73b`" in md and "598.8 s" in md
    assert "counts as verified only once a run of that exact blob is listed there" in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "a545a73b" in record and "598.8 s" in record


def test_owl_m3_spec_2_5_kind_use_notice_and_ieee_references(nb: dict) -> None:
    meta = nb["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    before = "\n".join(_src(c) for c in nb["cells"][:first_code])
    assert "> **⚠️ Notice.** For research, training and evaluation only." in before
    assert "without warranty of any kind" in before and "accept no liability" in before
    assert "> **Not a medical device.**" in before and "diagnosis" in before
    assert before.index("**⚠️ Notice.**") > before.index("\n")
    text = _markdown(nb)
    refs = text[text.index("## References"):]
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", refs, flags=re.M)]
    assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 8
    body = text[: text.index("## References")]
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\], \[\d+)*)\]", body) for n in re.findall(r"\d+", group)}
    assert cited == set(numbers), (sorted(cited), numbers)


def _byod_prefix(nb: dict, path: str) -> str:
    source = _cell_with(nb, "class BYODRefusal(ValueError):")
    source = source[: source.index("disjoint = check_split_disjoint(")]
    assert source.count("USE_BYOD = False") == 1 and source.count("BYOD_PATH = ''") == 1
    return source.replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}")


def _byod_namespace(**extra) -> dict:
    import os

    from owlv2_detection_pipeline import min_split_images as _msi

    ns = {"os": os, "Path": Path, "load_byod_dataset": load_byod_dataset, "split_dataset": split_dataset,
          "validate_dataset": validate_dataset, "MIN_RECORDS": MIN_RECORDS, "min_split_images": _msi,
          "train_records": ["stale"], "splits": {"stale": []}}
    ns.update(extra)
    return ns


def test_owl_m4_missing_byod_path_is_one_refusal_without_traceback(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _byod_namespace()
    with pytest.raises(ValueError) as caught:
        exec(_byod_prefix(nb, str(tmp_path / "missing.zip")), ns)
    exc = caught.value
    assert type(exc).__name__ == "BYODRefusal" and getattr(exc, "dimer_refusal", False)
    assert str(exc).startswith("BYOD input refused: BYOD path") and "does not exist" in str(exc)
    assert str(exc).endswith("run this cell again.")
    assert exc._render_traceback_() == [str(exc)]
    assert "train_records" not in ns and "splits" not in ns  # stale data cannot leak into later cells


def test_owl_m4_too_small_byod_zip_is_one_refusal(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "small.zip"
    rows = ["file,prompt,x0,y0,x1,y1"]
    with zipfile.ZipFile(archive, "w") as zf:
        for i, record in enumerate(_records(min_split_images() - 1)):
            buf = io.BytesIO()
            record["image"].save(buf, format="PNG")
            zf.writestr(f"img{i:02d}.png", buf.getvalue())
            rows.append(f"img{i:02d}.png,a cell,2,2,20,20")
        zf.writestr("boxes.csv", "\n".join(rows) + "\n")
    with pytest.raises(ValueError) as caught:
        exec(_byod_prefix(nb, str(archive)), _byod_namespace())
    assert type(caught.value).__name__ == "BYODRefusal"
    assert "small.zip" in str(caught.value) and f"at least {min_split_images()} distinct images" in str(caught.value)


def test_owl_m4_unexpected_byod_error_keeps_its_traceback(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir()

    def _broken(path):
        raise OSError("disk went away")

    with pytest.raises(OSError, match="disk went away"):
        exec(_byod_prefix(nb, str(folder)), _byod_namespace(load_byod_dataset=_broken))


def _kernel_source(nb: dict) -> str:
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and "# dimer: kernel cell" in _src(c)]
    assert len(cells) == 1
    return cells[0]


def test_owl_m4_router_raises_the_refusal_message_alone(nb: dict) -> None:
    import signal
    import sys

    tree = ast.parse(_kernel_source(nb))
    wanted = {"IsolatedCellError", "IsolatedCellRefusal", "IsolatedRuntime"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns: dict = {"sys": sys, "signal": signal}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<router>", "exec"), ns)

    class _Pipe:
        def __init__(self, messages):
            self.messages = list(messages)

        def recv(self):
            return self.messages.pop(0)

        def send(self, message):
            pass

    runtime = ns["IsolatedRuntime"].__new__(ns["IsolatedRuntime"])
    runtime._recv = _Pipe([("refusal", "BYOD input refused: x.zip: boxes.csv must have the columns ...")])
    runtime._send = _Pipe([])
    runtime._display = lambda *a, **k: None
    with pytest.raises(ns["IsolatedCellRefusal"]) as caught:
        runtime.run("raise_something()")
    assert caught.value._render_traceback_() == ["BYOD input refused: x.zip: boxes.csv must have the columns ..."]
    runtime._recv = _Pipe([("error", "Traceback ...\nOSError: boom\n", "OSError: boom")])
    with pytest.raises(ns["IsolatedCellError"]):
        runtime.run("raise_something()")


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds")
def test_owl_m4_worker_sends_refusals_without_traceback_and_errors_with_one(nb: dict) -> None:
    import os
    import subprocess
    from multiprocessing.connection import Connection

    tree = ast.parse(_kernel_source(nb))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_WORKER_SOURCE")
    worker = ast.literal_eval(node.value)
    to_kernel_r, to_kernel_w = os.pipe()
    to_worker_r, to_worker_w = os.pipe()
    proc = subprocess.Popen(
        [sys.executable, "-c", worker, str(to_kernel_w), str(to_worker_r)],
        pass_fds=(to_kernel_w, to_worker_r),
        env=dict(os.environ, MPLBACKEND="Agg", DIMER_KERNEL_IS_COLAB="0"),
    )
    os.close(to_kernel_w)
    os.close(to_worker_r)
    recv, send = Connection(to_kernel_r, writable=False), Connection(to_worker_w, readable=False)

    def _final(source):
        send.send(("run", source))
        while True:
            message = recv.recv()
            if message[0] in ("refusal", "error", "done"):
                return message

    try:
        assert _final("class R(ValueError):\n    dimer_refusal = True\nraise R('BYOD input refused: x')\n") == ("refusal", "BYOD input refused: x")
        error = _final("raise OSError('boom')\n")
        assert error[0] == "error" and "Traceback" in error[1] and error[2] == "OSError: boom"
        assert _final("x = 1\n") == ("done",)
    finally:
        send.close()
        proc.wait(timeout=30)


def test_owl_m5_isolated_env_override_documented(nb: dict) -> None:
    assert "DIMER_ISOLATED_ENV" in _markdown(nb)


def test_owl_m6_float16_cache_wording_matches_the_code(nb: dict) -> None:
    md = _markdown(nb)
    pipeline = (ROOT / "src" / "owlv2_detection_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert '.to("cpu", torch.float16)' in pipeline
    for text in (md, pipeline):
        assert "logits equal the full model" not in text
        assert "train on float16-quantized frozen features (promoted to the head dtype)" in " ".join(text.split())
        assert "approximates training on freshly computed float32 features" in " ".join(text.split())
    assert "run the same arithmetic" not in md
    assert "a fixed-weight comparison" in md and "0.047" in md  # the logit bound is the fixed-weight comparison
    assert "0.8914" in md and "0.8885" in md and "0.279" in md and "0.333" in md and "2,974" in md  # training comparison
    assert "In the reported runs" in md or "in the reported runs" in md


# --- Codex peer review item 12: non-square padding, target normalisation and prediction inversion ------------------


def test_owl_r12_non_square_box_round_trips_through_the_padded_square() -> None:
    """A 300 x 200 image with one known box: the processor pads it to a square on the bottom/right, so the target
    normalisation divides both axes by max(W, H); the box located in the processor's own 960 x 960 pixel_values
    matches those fractions, and `_postprocess` maps the fractions back to the original pixel box."""
    import numpy as np
    import torch
    from transformers import Owlv2ImageProcessor

    width, height, box = 300, 200, [60.0, 120.0, 180.0, 180.0]
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0, x1, y1 = (int(v) for v in box)
    pixels[y0:y1, x0:x1] = 255
    image = Image.fromarray(pixels)

    classes, cxcywh = Owlv2DetectionPipeline._targets({"image": image, "boxes": [{"prompt": "a cell", "box": box}]}, ["a cell"])
    side = float(max(width, height))
    expected = torch.tensor([[(x0 + x1) / 2 / side, (y0 + y1) / 2 / side, (x1 - x0) / side, (y1 - y0) / side]])
    assert classes.tolist() == [0] and torch.allclose(cxcywh, expected, atol=1e-6)

    values = Owlv2ImageProcessor()(images=[image], return_tensors="pt")["pixel_values"][0].mean(0).numpy()
    size = values.shape[0]
    bright = np.argwhere(values > values.max() - 0.5)
    (ry0, rx0), (ry1, rx1) = bright.min(0), bright.max(0) + 1
    seen = np.array([rx0, ry0, rx1, ry1], dtype=float) / size
    assert np.allclose(seen, np.array(box) / side, atol=2.0 / size), (seen, np.array(box) / side)

    logits = torch.full((1, 1, 1), 10.0)
    dets = Owlv2DetectionPipeline._postprocess(logits, cxcywh.unsqueeze(0), ["a cell"], [(width, height)], 0.1)
    assert len(dets[0]) == 1 and dets[0][0]["label"] == "a cell"
    assert np.allclose(dets[0][0]["box"], box, atol=1e-3), dets[0][0]["box"]


# --- Codex peer review round 2 (items 1, 4, 5, 9, 2, 6, 8, 10, 11, 13) -------------------------------------------------


def _box(x0: float, y0: float, x1: float, y1: float, label: str = "a cell", score: float = 0.9) -> dict:
    return {"box": [x0, y0, x1, y1], "label": label, "score": score}


def test_owl_r4_voc_matching_counts_a_second_detection_of_a_taken_box_as_a_false_positive() -> None:
    """Two overlapping references: the second detection overlaps the taken box most and the free one only enough.
    VOC matching makes it a false positive; the earlier greedy best-unmatched rule would have matched the free box."""
    from owlv2_detection_pipeline.metrics import _match, detection_metrics

    refs = [{"prompt": "a cell", "box": [0, 0, 10, 10]}, {"prompt": "a cell", "box": [0, 3, 10, 13]}]
    first, second = _box(0, 0, 10, 10, score=0.9), _box(0, 1, 10, 11, score=0.8)
    assert box_iou_ok(second, refs)
    assert _match([first, second], refs, 0.5) == [(0.9, "a cell", True), (0.8, "a cell", False)]
    record = {"id": "r", "image": Image.new("RGB", (16, 16)), "boxes": refs}
    m = detection_metrics([[first, second]], [record])
    assert m["n_matched"] == 1 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert "conventional VOC matching" in m["definitions"]["map50"] and "retained at the score threshold" in m["definitions"]["map50"]


def box_iou_ok(pred: dict, refs: list) -> bool:
    from owlv2_detection_pipeline import box_iou

    first_iou, second_iou = (box_iou(pred["box"], r["box"]) for r in refs)
    return first_iou > second_iou >= 0.5


def test_owl_r5_selection_compares_the_unrounded_validation_map() -> None:
    from owlv2_detection_pipeline.metrics import detection_metrics

    refs = [{"prompt": "a cell", "box": [0, 0, 10, 10]}, {"prompt": "a cell", "box": [20, 20, 30, 30]}, {"prompt": "a cell", "box": [40, 40, 50, 50]}]
    record = {"id": "r", "image": Image.new("RGB", (64, 64)), "boxes": refs}
    m = detection_metrics([[_box(0, 0, 10, 10, score=0.9), _box(60, 60, 63, 63, score=0.8), _box(20, 20, 30, 30, score=0.7)]], [record])
    assert round(m["map50_exact"], 4) == m["map50"] and m["map50_exact"] != m["map50"]
    source = (ROOT / "src" / "owlv2_detection_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert 'score = (entry["val"] or {}).get("map50_exact")' in source
    assert '(history[0]["val"] or {}).get("map50_exact", -1.0)' in source


def test_owl_r1_all_candidates_is_a_display_only_diagnostic() -> None:
    def runner(image, queries, threshold):
        dets = [_box(2, 2, 20, 20, score=0.9), _box(2, 2, 20, 20, score=0.05), _box(40, 40, 60, 60, score=0.04)]
        return [d for d in dets if d["score"] >= threshold]

    records = [{"id": f"r{i}", "image": Image.new("RGB", (64, 64), (i, i, i)), "boxes": [{"prompt": "a cell", "box": [2, 2, 20, 20]}, {"prompt": "a cell", "box": [40, 40, 60, 60]}]} for i in range(3)]
    pipe = Owlv2DetectionPipeline(runner, "cpu")
    plain = pipe.evaluate(records, prompts=["a cell"], threshold=0.1)
    both = pipe.evaluate(records, prompts=["a cell"], threshold=0.1, all_candidates=True)
    assert "all_candidates" not in plain
    for key in ("map50", "map50_exact", "precision", "recall", "f1", "n_predicted_boxes", "n_matched", "per_prompt"):
        assert both[key] == plain[key], key
    # all candidates, ranked: three 0.9 hits, three 0.05 duplicates (false positives), three 0.04 hits -> AP 0.5 + 0.5 * 6/9
    assert plain["map50"] == 0.5 and both["all_candidates"]["map50"] == 0.8333
    assert both["all_candidates"]["n_predicted_boxes"] == 9 and "display-only" in both["all_candidates"]["note"]
    with pytest.raises(ValueError):
        pipe.evaluate(records, prompts=["a cell"], threshold=0.1, all_candidates="yes")


def test_owl_r_round2_notebook_wording(nb: dict) -> None:
    md = _markdown(nb)
    code = "\n".join(_code(nb))
    assert "AP50 over detections retained at score ≥ 0.1" in md and "all-candidates AP50" in md
    assert "all_candidates=True" in code and "frozen_all_candidates_ap50" in code and "all_candidates_ap50_display_only" in code
    assert "moved threshold, not a better detector" not in md and "is a moved threshold" not in md
    assert "one precision/recall pair is one operating point" in md
    assert "exploratory" in md and "report them collectively" in md
    assert "without exact decoded-image overlap" in md and "near-duplicates are not established" in md
    assert "Both rose, but precision remained low" in md and "plausible contributor" in md and "most kept boxes are duplicates" not in md
    assert "**detection parity at reported precision**" in md and "'compared_at': 'label, score to 4 dp, box to 1 dp'" in code
    assert "outperforms this training-derived grid prior" in md and "tiling order" in md
    assert "suppressing lower-scoring boxes that overlap a higher-scoring box" in md
    assert "IoU = 1/3" in md and "what intersection-over-union, average precision" not in md
    assert "supervised adaptation to three fixed phrases" in md and "not a cell count" in md
    assert "DETR-style matched objective" in md and "the upstream heads were trained with" not in md and "with the upstream objective" not in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "the current blob" not in record.lower() and "The current blob" not in record and "| Current `E2E` carrier" not in record
