"""Model-backed checks that run only where the pinned snapshot is staged: the measured model facts, batched
detection equal to single-image detection and to the pinned processor's post-processing, the heads on cached image
features equal to the full forward, corpus evaluation on drawn shapes, a short adaptation of the heads, the
artifact round trip with reload parity, the loader's scope check, the transactional guarantee and — where CUDA is
visible — the same path on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from owlv2_detection_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    HEAD_PARAMETERS,
    PARAMETER_COUNT,
    WEIGHTS_FILE,
    Owlv2DetectionPipeline,
)
from owlv2_detection_pipeline.pipeline import _TRAINABLE_PREFIXES

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

SHAPES = (("a red square", (220, 30, 30)), ("a blue circle", (30, 60, 220)), ("a green triangle", (30, 160, 60)))
PROMPTS = [name for name, _ in SHAPES]


def _record(i, size=(320, 240)):
    """A white scene with two coloured shapes; each phrase names one and its box is the shape's extent."""
    rng = np.random.default_rng(i)
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    boxes = []
    for slot, kind in enumerate((i + s) % len(SHAPES) for s in (0, 1)):
        x, y = int(rng.integers(8, 120)) + slot * 160, int(rng.integers(8, 120))
        side = int(rng.integers(60, 100))
        if kind == 0:
            draw.rectangle((x, y, x + side, y + side), fill=SHAPES[0][1])
        elif kind == 1:
            draw.ellipse((x, y, x + side, y + side), fill=SHAPES[1][1])
        else:
            draw.polygon([(x, y + side), (x + side // 2, y), (x + side, y + side)], fill=SHAPES[2][1])
        boxes.append({"prompt": SHAPES[kind][0], "box": [float(x), float(y), float(x + side), float(y + side)]})
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return {"id": f"shape{i:02d}", "image": image, "boxes": boxes}


@pytest.fixture(autouse=True)
def _release_memory():
    yield
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return Owlv2DetectionPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR)


def test_model_facts_batched_detection_head_parity_and_frozen_evaluation(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith(_TRAINABLE_PREFIXES)) == HEAD_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    single = [pipe.detect(r["image"], PROMPTS)["detections"] for r in records[:3]]
    batched = pipe.detect_batch([r["image"] for r in records[:3]], PROMPTS, batch_size=3)
    for s, b in zip(single, batched, strict=True):
        assert len(s) == len(b) and [d["label"] for d in s] == [d["label"] for d in b]
        # batch-size-dependent matmul kernels move a sigmoid by O(1e-4); a pairing bug would move it by O(0.5)
        assert max((abs(x["score"] - y["score"]) for x, y in zip(s, b, strict=True)), default=0.0) < 1e-3
    # the heads on cached features equal the full forward, and our post-processing equals the processor's
    image = records[0]["image"]
    inputs = pipe._processor(images=image, text=[PROMPTS], return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        full = pipe._model(**inputs)
        logits, boxes = pipe._heads(pipe._cache(records[:1], 1), pipe._encode_queries(PROMPTS))
    assert float((logits - full.logits).abs().max()) < 0.1 and float((boxes - full.pred_boxes).abs().max()) < 1e-3  # fp16 cache
    hf = pipe._processor.post_process_grounded_object_detection(full, threshold=0.1, target_sizes=[image.size[::-1]], text_labels=[PROMPTS])[0]
    ours = sorted(pipe._postprocess(full.logits, full.pred_boxes, PROMPTS, [image.size], 0.1)[0], key=lambda d: -d["score"])
    theirs = sorted(zip(hf["scores"].tolist(), hf["boxes"].tolist(), hf["text_labels"], strict=True), key=lambda t: -t[0])
    assert len(ours) == len(theirs)
    for mine, (score, box, label) in zip(ours, theirs, strict=True):
        assert abs(mine["score"] - score) < 1e-5 and mine["label"] == label and max(abs(a - b) for a, b in zip(mine["box"], box, strict=True)) < 1e-2
    metrics = pipe.evaluate(records[:8])
    assert metrics["n"] == 8 and metrics["adapted"] is False and metrics["verdict"] == "measured-small-sample" and metrics["threshold"] == 0.1 and metrics["prompts"] == sorted(PROMPTS)
    assert 0.0 <= metrics["map50"] <= 1.0 and len(metrics["rows"]) == 8 and metrics["rows"][0]["n_reference"] == 2


def test_short_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=2, lr=1e-4, batch_size=4)
    assert result["n_trainable"] == HEAD_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["threshold"] == 0.1 and result["prompts"] == sorted(PROMPTS)
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0 and set(result["history"][1]["loss_terms"]) == {"focal", "l1", "giou"}
    assert set(result["history"][1]["val"]) == {"map50", "precision", "recall", "f1", "n", "n_predicted_boxes"} and result["best_epoch"] in (0, 1, 2)
    assert all(n.startswith(("class_head.", "box_head.")) for n in result["trainable_names"]) and len(result["trainable_names"]) == 12
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == 12 and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["metadata"] == {"note": "test"} and manifest["adapter"]["selection"] == "highest validation mAP at the IoU threshold"
    reloaded = Owlv2DetectionPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    a = pipe.detect_batch([r["image"] for r in records[:3]], PROMPTS)
    b = reloaded.detect_batch([r["image"] for r in records[:3]], PROMPTS)
    assert [len(x) for x in a] == [len(y) for y in b]
    assert all(abs(p["score"] - q["score"]) < 1e-5 and p["label"] == q["label"] for x, y in zip(a, b, strict=True) for p, q in zip(x, y, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.adapter["prompts"] == sorted(PROMPTS) and reloaded.evaluate(records[:4])["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = Owlv2DetectionPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])


def test_adapt_refuses_bad_hyperparameters_and_datasets(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:12], None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:12], None, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:12], None, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, epochs=1)
    with pytest.raises(ValueError, match="outside the"):
        pipe.adapt([*records[:11], {**records[11], "boxes": [{"prompt": "a platelet", "box": [0, 0, 400, 5]}]}], None, epochs=1)
    with pytest.raises(ValueError, match="outside the training vocabulary"):
        pipe.adapt(records[:12], None, epochs=1, prompts=PROMPTS[:1])
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        Owlv2DetectionPipeline.from_artifact(fewer, weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["box_head.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        Owlv2DetectionPipeline.from_artifact(extra, weights_dir=DEFAULT_WEIGHTS_DIR)
    tower = tmp_path / "tower"
    shutil.copytree(artifact, tower)
    (tower / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "owlv2.vision_model.encoder.layers.0.x"]}))
    with pytest.raises(ValueError, match="class and box heads"):
        Owlv2DetectionPipeline.from_artifact(tower, weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_the_default_device_is_cuda_when_visible(pipe):
    assert pipe.device == "cuda:0" and next(pipe._model.parameters()).device.type == "cuda"
