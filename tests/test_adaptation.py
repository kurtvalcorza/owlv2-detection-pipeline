"""Offline checks of the adaptation contract: the (image, phrases, boxes) record contract and its refusals, the
pinned-corpus refusals and the draw, splitting, the BYOD loader, the metrics and baselines, the Hungarian matcher,
`detect_batch` / `evaluate` with an injected runner, the artifact-manifest checks, and the model-free refusals of
`adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import itertools
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from owlv2_detection_pipeline import (
    ARTIFACT_FORMAT,
    BCCD_CLASSES,
    CLASS_PHRASES,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_PHRASES,
    SAMPLE_SPLIT,
    Owlv2DetectionPipeline,
    average_precision,
    boxes_digest,
    build_sample_dataset,
    check_split_disjoint,
    coerce_box,
    dataset_digest,
    detection_metrics,
    empty_baseline,
    fetch_corpus,
    grid_baseline,
    hungarian,
    image_digest,
    load_byod_dataset,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from owlv2_detection_pipeline import pipeline as pl
from owlv2_detection_pipeline import samples as sm

PROMPTS = ["a platelet", "a red blood cell", "a white blood cell"]


def _image(i, size=(64, 48)):
    image = Image.new("RGB", size, ((i * 37) % 256, 120, 90))
    image.putpixel((i % size[0], 0), (255, 0, 0))
    return image


def _boxes(i):
    x = float((i * 7) % 30)
    return [{"prompt": PROMPTS[i % 3], "box": [x, 8.0, x + 16.0, 32.0]}, {"prompt": PROMPTS[(i + 1) % 3], "box": [40.0, 2.0 + i % 5, 60.0, 20.0 + i % 5]}]


def _record(i, boxes=None, **extra):
    return {"id": f"r{i:03d}", "image": _image(i), "boxes": _boxes(i) if boxes is None else boxes, **extra}


def _records(n=12):
    return [_record(i) for i in range(n)]


def _echo_runner(image, queries, threshold):
    """An injected runner that returns the record's own boxes (a side table by pixel digest) at score 0.9."""
    boxes = _echo_runner.table.get(image_digest(image), [])
    return [{"box": list(b["box"]), "label": b["prompt"], "score": 0.9} for b in boxes if b["prompt"] in queries and threshold <= 0.9]


_echo_runner.table = {}


# --- record contract -------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    info = validate_dataset(_records())
    assert info["n_records"] == 12 and info["n_boxes"] == 24 and info["n_prompts"] == 3 and info["prompts"] == sorted(PROMPTS)
    assert info["boxes_per_prompt"] == {p: 8 for p in PROMPTS} and info["boxes_per_record"] == {"min": 2, "max": 2, "mean": 2.0}
    assert info["image_width"] == {"min": 64, "max": 64} and 0.0 < info["box_area_fraction"]["mean"] < 1.0
    assert info["digest"] == dataset_digest(_records()) and info["model_id"] == MODEL_ID
    # prompts are normalised like queries; boxes may hold strings or ints
    item = validate_dataset([_record(0, boxes=[{"prompt": "  A Platelet. ", "box": ["1", 2, "10.5", 20]}])] + _records()[1:])["records"][0]
    assert item["boxes"] == [{"prompt": "a platelet", "box": [1.0, 2.0, 10.5, 20.0]}]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("boxes"), "missing 'boxes'"),
        (lambda r: r.update(id="bad id!"), "id must match"),
        (lambda r: r.update(image=Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.update(boxes=[]), "1..200"),
        (lambda r: r.update(boxes=[{"prompt": "   ", "box": [0, 0, 5, 5]}]), "must not be empty"),
        (lambda r: r.update(boxes=[{"prompt": "x" * 49, "box": [0, 0, 5, 5]}]), "MAX_PROMPT_CHARS"),
        (lambda r: r.update(boxes=[{"prompt": "a platelet", "box": [0, 0, 5]}]), r"\[x0, y0, x1, y1\]"),
        (lambda r: r.update(boxes=[{"prompt": "a platelet", "box": [0, 0, 70, 5]}]), "outside the 64x48 image"),
        (lambda r: r.update(boxes=[{"prompt": "a platelet", "box": [10, 10, 10.5, 20]}]), "at least 1.0 px"),
        (lambda r: r.update(boxes=[{"prompt": "a platelet", "box": ["a", 0, 5, 5]}]), "four numbers"),
        (lambda r: r.update(boxes=[{"prompt": "a platelet"}]), "prompt and box"),
        (lambda r: r.update(boxes="not boxes"), "boxes must be a list"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[3])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_unique_ids_and_the_query_ceiling():
    with pytest.raises(ValueError, match="8..5000 are required"):
        validate_dataset(_records(4))
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    dup = _records()
    dup[1]["id"] = dup[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(dup)
    many = [_record(i, boxes=[{"prompt": f"phrase {i}", "box": [0, 0, 5, 5]}]) for i in range(17)]
    with pytest.raises(ValueError, match="at most 16 queries"):
        validate_dataset(many)
    assert coerce_box([0, 0, 64, 48], (64, 48)) == [0.0, 0.0, 64.0, 48.0]
    with pytest.raises(ValueError, match="finite"):
        coerce_box([0, 0, float("nan"), 5], (64, 48))


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError, match="outside the 64x48 image"):
        validate_dataset([_record(0, boxes=[{"prompt": "a platelet", "box": [0, 0, 65, 5]}])] + _records()[1:])


def test_digests_and_split_disjointness():
    a, b = _record(0), _record(0)
    assert image_digest(a["image"]) == image_digest(b["image"]) and boxes_digest(a["boxes"]) == boxes_digest(b["boxes"])
    assert boxes_digest(list(reversed(a["boxes"]))) == boxes_digest(a["boxes"])
    assert image_digest(_record(1)["image"]) != image_digest(a["image"])
    assert dataset_digest([a, _record(1)]) == dataset_digest([_record(1), a])
    assert dataset_digest([a]) != dataset_digest([dict(a, boxes=a["boxes"][:1])])
    splits = {"train": _records()[:9], "test": _records()[9:]}
    assert check_split_disjoint(splits) == {"train": 9, "test": 3}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": _records()[:9], "test": [_record(0)]})


def test_split_dataset_is_seeded_and_deduplicates():
    records = _records(20)
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert check_split_disjoint(splits) == {"test": 4, "validation": 4, "train": 12}
    assert splits == split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert splits != split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=2)
    dup = records + [dict(records[0], id="dup")]
    assert sum(len(part) for part in split_dataset(dup).values()) == 20
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- pinned corpus ---------------------------------------------------------------------------------------------


def test_pins_classes_and_draw_sizes():
    assert sm.CORPUS_REVISION == "22cf1b9d2367e799ab54a16774a2266e4f8ce9a9" and sorted(sm.CORPUS_FILES) == ["test", "train", "validation"]
    assert all(len(sha) == 64 and n > 400_000 for _p, sha, n, _r in sm.CORPUS_FILES.values())
    assert sum(n for _p, _s, n, _r in sm.CORPUS_FILES.values()) == sm.CORPUS_BYTES == 4_820_309
    assert sum(r for _p, _s, _n, r in sm.CORPUS_FILES.values()) == sm.CORPUS_ROWS == 364
    assert BCCD_CLASSES == ("platelets", "rbc", "wbc") and SAMPLE_PHRASES == ("a platelet", "a red blood cell", "a white blood cell") and CLASS_PHRASES["rbc"] == "a red blood cell"
    assert SAMPLE_SPLIT == {"train": 260, "validation": 40, "test": 64} and len(sm.SAMPLE_DIGEST) == 64


def _jpeg(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_table(n=6):
    rows = []
    for i in range(n):
        image = np.full((32, 32, 3), (i * 30) % 256, dtype=np.uint8)
        rows.append({"image_id": i, "image": {"bytes": _jpeg(image), "path": f"{i}.jpg"}, "width": 32, "height": 32, "objects": {"id": [i, i + 100], "area": [100, 50], "bbox": [[2.0, 3.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]], "category": [1, i % 3]}})
    return pa.Table.from_pylist(rows)


def test_fetch_corpus_refuses_a_file_that_does_not_match_its_pin(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    pq.write_table(_fake_table(), buffer)
    data = buffer.getvalue()
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path / "cache", splits=["test"], opener=lambda url: data)
    assert not (tmp_path / "cache" / "test.parquet").exists()
    monkeypatch.setitem(sm.CORPUS_FILES, "test", ("full/test/0000.parquet", "0" * 64, len(data), 6))
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", splits=["test"], opener=lambda url: data)
    with pytest.raises(ValueError, match="unknown corpus split"):
        fetch_corpus(cache_dir=tmp_path / "cache", splits=["other"], opener=lambda url: data)
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "cache" / "test.parquet").write_bytes(b"stale")  # a stale cache file is refetched and refused the same way
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", splits=["test"], opener=lambda url: data)
    monkeypatch.setitem(sm.CORPUS_FILES, "test", ("full/test/0000.parquet", hashlib.sha256(data).hexdigest(), len(data), 6))
    assert fetch_corpus(cache_dir=tmp_path / "cache", splits=["test"], opener=lambda url: data) == {"test": data}
    assert (tmp_path / "cache" / "test.parquet").read_bytes() == data


def test_read_corpus_converts_coco_boxes_to_phrased_xyxy_records(monkeypatch):
    buffer = io.BytesIO()
    pq.write_table(_fake_table(4), buffer)
    monkeypatch.setitem(sm.CORPUS_FILES, "test", ("full/test/0000.parquet", "x", 1, 4))
    records = read_corpus({"test": buffer.getvalue()})
    assert len(records) == 4 and records[0]["id"] == "bccd-test-0" and records[0]["source_split"] == "test" and records[0]["source_image_id"] == 0
    assert records[0]["boxes"] == [{"prompt": "a red blood cell", "box": [2.0, 3.0, 12.0, 13.0]}, {"prompt": "a platelet", "box": [20.0, 20.0, 32.0, 32.0]}]
    assert records[1]["boxes"][1]["prompt"] == "a red blood cell" and records[2]["boxes"][1]["prompt"] == "a white blood cell"
    with pytest.raises(ValueError, match="rows, pinned"):
        read_corpus({"validation": buffer.getvalue()})


def test_build_sample_dataset_draws_seeded_sizes():
    records = _records(20)
    splits = build_sample_dataset(records, seed=3, sizes={"train": 12, "validation": 3, "test": 5})
    assert {k: len(v) for k, v in splits.items()} == {"train": 12, "validation": 3, "test": 5}
    assert splits == build_sample_dataset(records, seed=3, sizes={"train": 12, "validation": 3, "test": 5})
    with pytest.raises(ValueError, match="distinct images <"):
        build_sample_dataset(records[:8], sizes={"train": 12, "validation": 3, "test": 5})


@pytest.mark.skipif(not all((sm.DEFAULT_CACHE_DIR / f"{s}.parquet").is_file() for s in sm.CORPUS_FILES), reason="pinned parquet files not cached")
def test_default_draw_matches_the_pinned_digest_when_the_files_are_cached():
    splits = build_sample_dataset(read_corpus(fetch_corpus()))
    assert check_split_disjoint(splits) == SAMPLE_SPLIT
    assert dataset_digest(splits["train"] + splits["validation"] + splits["test"]) == sm.SAMPLE_DIGEST


# --- BYOD --------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_and_boxes_from_a_zip_or_directory(tmp_path):
    root = tmp_path / "boxes"
    root.mkdir()
    for i in range(3):
        _image(i).save(root / f"img{i}.png")
    (root / "boxes.csv").write_text("file,prompt,x0,y0,x1,y1\nimg0.png,a platelet,1,2,10,20\nimg0.png,A red blood cell.,20,2,30,20\nimg1.png,a white blood cell,0,0,64,48\nimg2.png,a platelet,5,5,15,15\n", encoding="utf-8")
    records = load_byod_dataset(root)
    assert [r["id"] for r in records] == ["img0", "img1", "img2"] and len(records[0]["boxes"]) == 2
    checked = validate_dataset(records, min_records=1)["records"]
    assert checked[0]["boxes"][1] == {"prompt": "a red blood cell", "box": [20.0, 2.0, 30.0, 20.0]}
    archive = tmp_path / "boxes.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for file in root.iterdir():
            z.write(file, f"nested/{file.name}")
    assert [r["id"] for r in load_byod_dataset(archive)] == ["img0", "img1", "img2"]
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as z:
        z.writestr("a/boxes.csv", "file,prompt,x0,y0,x1,y1\nimg0.png,a platelet,1,2,10,20\n")
        z.writestr("a/img0.png", (root / "img0.png").read_bytes())
        z.writestr("b/img0.png", (root / "img1.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate basename 'img0.png'"):
        load_byod_dataset(duplicate)
    _image(9).save(root / "extra.png")
    with pytest.raises(ValueError, match="no boxes.csv row"):
        load_byod_dataset(root)
    (root / "extra.png").unlink()
    (root / "boxes.csv").write_text("file,prompt,x0,y0,x1,y1\nmissing.png,a platelet,1,2,10,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing file"):
        load_byod_dataset(root)
    (root / "boxes.csv").write_text("file,text\nimg0.png,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns file, prompt, x0, y0, x1, y1"):
        load_byod_dataset(root)
    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        load_byod_dataset(tmp_path / "nope.txt")
    out = write_dataset_csv(checked, tmp_path / "out" / "train.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,file,prompt,x0,y0,x1,y1,width,height,source_split,source_image_id" and len(lines) == 5


# --- metrics, baselines and the matcher ----------------------------------------------------------------------------


def test_detection_metrics_average_precision_and_baselines():
    records = _records()
    perfect = detection_metrics([[{"box": b["box"], "label": b["prompt"], "score": 0.9} for b in r["boxes"]] for r in records], records)
    assert perfect["map50"] == 1.0 and perfect["precision"] == 1.0 and perfect["recall"] == 1.0 and perfect["n_matched"] == 24 and perfect["per_prompt"]["a platelet"]["ap"] == 1.0
    # one wrong label, one shifted box, one extra detection
    preds = [[{"box": b["box"], "label": b["prompt"], "score": 0.9} for b in r["boxes"]] for r in records]
    preds[0][0]["label"] = PROMPTS[(0 + 2) % 3]
    x = preds[1][0]["box"]
    preds[1][0]["box"] = [x[0] + 12, x[1], x[2] + 12, x[3]]  # IoU 4/28 < 0.5
    preds[2].append({"box": [0.0, 0.0, 5.0, 5.0], "label": PROMPTS[0], "score": 0.99})
    m = detection_metrics(preds, records)
    assert m["n_matched"] == 22 and m["n_predicted_boxes"] == 25 and m["recall"] == pytest.approx(22 / 24, abs=1e-4) and m["precision"] == pytest.approx(22 / 25, abs=1e-4) and 0 < m["map50"] < 1
    assert m["rows"][2] == {"id": "r002", "n_reference": 2, "n_predicted": 3, "matched": 2}
    assert average_precision([(0.9, True), (0.8, False), (0.7, True)], 2) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision([], 3) == 0.0 and average_precision([(0.5, True)], 0) == 0.0
    with pytest.raises(ValueError, match="prediction lists for"):
        detection_metrics([[]], records)
    with pytest.raises(ValueError, match="iou_threshold"):
        detection_metrics(preds, records, iou_threshold=0)
    empty = empty_baseline(records)
    assert empty["map50"] == 0.0 and empty["recall"] == 0.0 and empty["baseline"] == "empty"
    grid = grid_baseline(records, records)
    assert sorted(grid["prior_box_sizes"]) == sorted(PROMPTS) and all(0 < w <= 64 and 0 < h <= 48 for w, h in grid["prior_box_sizes"].values())
    assert grid["n_predicted_boxes"] > 0 and grid["baseline"].startswith("grid prior") and 0.0 <= grid["map50"] <= 1.0


def test_hungarian_matches_brute_force_on_rectangular_costs():
    rng = np.random.default_rng(0)
    for n, m in [(1, 4), (3, 3), (4, 7), (5, 9)]:
        cost = rng.random((n, m))
        pairs = hungarian(cost)
        assert len(pairs) == n and len({c for _, c in pairs}) == n and [r for r, _ in pairs] == list(range(n))
        best = min(sum(cost[r, c] for r, c in enumerate(perm)) for perm in itertools.permutations(range(m), n))
        assert sum(cost[r, c] for r, c in pairs) == pytest.approx(best)
    assert hungarian(np.zeros((0, 5))) == []
    assert hungarian(np.array([[5.0, 1.0], [1.0, 5.0]])) == [(0, 1), (1, 0)]
    with pytest.raises(ValueError, match="more rows"):
        hungarian(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        hungarian(np.array([[np.inf, 1.0]]))


# --- detect_batch / evaluate with an injected runner ---------------------------------------------------------------


def test_detect_batch_and_evaluate_score_the_runner_and_flag_small_samples():
    records = _records()
    _echo_runner.table = {image_digest(r["image"]): r["boxes"] for r in records[:6]}
    pipe = Owlv2DetectionPipeline(_echo_runner, "cpu")
    batches = pipe.detect_batch([r["image"] for r in records], PROMPTS, batch_size=5)
    assert len(batches) == 12 and batches[0][0]["label"] in PROMPTS and len(batches[0]) == 2 and batches[7] == []
    report = pipe.evaluate(records)
    assert report["n"] == 12 and report["recall"] == 0.5 and report["precision"] == 1.0 and report["map50"] == 0.5 and report["verdict"] == "measured-small-sample" and report["adapted"] is False and report["threshold"] == 0.1 and report["prompts"] == sorted(PROMPTS)
    assert pipe.evaluate(records, threshold=0.95)["n_predicted_boxes"] == 0  # nothing reaches 0.95
    with pytest.raises(ValueError, match="batch_size"):
        pipe.detect_batch([records[0]["image"]], PROMPTS, batch_size=0)
    with pytest.raises(ValueError, match="threshold"):
        pipe.evaluate(records, threshold=1.5)
    with pytest.raises(ValueError, match="outside the evaluated vocabulary"):
        pipe.evaluate(records, prompts=PROMPTS[:2])


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = Owlv2DetectionPipeline(_echo_runner, "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records(), None)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), None, lr=0.5)
    with pytest.raises(ValueError, match="threshold"):
        pipe.adapt(_records(), None, threshold=2)
    with pytest.raises(ValueError, match="outside the training vocabulary"):
        pipe.adapt(_records(), None, prompts=PROMPTS[:1])
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "tensors": ["class_head.dense0.weight", "class_head.logit_scale.bias", "box_head.dense2.weight"],
        "adapter": {"best_epoch": 1, "threshold": 0.1, "prompts": list(PROMPTS)},
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="exactly adapter.safetensors"):
        pl._check_artifact_manifest(_manifest(tmp_path, files=[]), tmp_path, "base-digest")
    for name in ("owlv2.vision_model.encoder.layers.0.x", "owlv2.text_model.encoder.layers.0.x", "owlv2.visual_projection.weight", "objectness_head.dense0.weight", "layer_norm.weight"):
        with pytest.raises(ValueError, match="class and box heads"):
            pl._check_artifact_manifest(_manifest(tmp_path, tensors=[name]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="score threshold"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"best_epoch": 1, "prompts": PROMPTS}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="phrase vocabulary"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"best_epoch": 1, "threshold": 0.1}), tmp_path, "base-digest")


def test_trainable_names_selects_the_heads():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["owlv2.vision_model.encoder.layers.0.x", "owlv2.visual_projection.weight", "class_head.dense0.weight", "class_head.logit_shift.bias", "box_head.dense1.weight", "objectness_head.dense0.weight", "layer_norm.weight", "owlv2.text_projection.weight"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["class_head.dense0.weight", "class_head.logit_shift.bias", "box_head.dense1.weight"]
    assert pl._TRAINABLE_PREFIXES == ("class_head.", "box_head.") and pl.HEAD_PARAMETERS == 1_579_526 and pl.PARAMETER_COUNT == 154_966_792 and pl.NUM_PATCHES == 3600
    assert pl.LOSS_WEIGHTS == {"class": 2.0, "l1": 5.0, "giou": 2.0} and pl.FOCAL_ALPHA == 0.25 and pl.FOCAL_GAMMA == 2.0


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.3, "val": {"map50": 0.5, "precision": 0.6, "recall": 0.6, "f1": 0.6, "n": 40, "n_predicted_boxes": 300}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["map50"] == 0.5
