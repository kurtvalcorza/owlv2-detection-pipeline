"""Open-vocabulary (text-prompted) object detection with the pinned ``google/owlv2-base-patch16-ensemble``
checkpoint (OWLv2).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the OWLv2 architecture comes from the pinned ``transformers`` release, the
weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "owlv2-base-patch16-ensemble"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Threshold: the value the pinned README's usage example passes to post_process_object_detection
# (threshold=0.1). It gates a sigmoid over the best text query per image patch that is not calibrated;
# the deployment owns tuning it on its own labelled data.
DETECTION_THRESHOLD = 0.1
# The ViT-B/16 image tower sees a 960x960 padded square as 60x60 = 3600 patch tokens, each of which
# is one detection candidate, so no image can yield more than this many boxes.
MAX_DETECTIONS = 3600
# Input ceilings. The processor pads the image to a square with grey (bottom/right) and resizes it to
# 960x960 (preprocessor_config.json), so image cost is bounded; each text query is tokenised by the
# CLIP tokenizer with model_max_length 16 (padded/truncated), so a phrase longer than that is cut.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 48
MAX_TEXT_TOKENS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for any caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def format_prompts(prompts: Sequence[str]) -> list[str]:
    """Validate a list of phrases and normalise them to the OWLv2 query form: stripped, lower-cased, one
    text query per phrase (the upstream example uses "a photo of a cat"-style queries; the pipeline
    passes the caller's phrases through unchanged apart from case and whitespace)."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a list of phrases, not a single string")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"prompt count {len(prompts)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
    cleaned: list[str] = []
    for phrase in prompts:
        if not isinstance(phrase, str):
            raise TypeError(f"prompt must be str, got {type(phrase).__name__}")
        text = " ".join(phrase.split()).strip().rstrip(".").strip().lower()
        if not text:
            raise ValueError("prompt phrases must not be empty")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt {text[:12]!r}... is {len(text)} chars > MAX_PROMPT_CHARS {MAX_PROMPT_CHARS}"
            )
        cleaned.append(text)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("prompt phrases must be distinct after normalisation")
    return cleaned


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image (any mode, converted to RGB) plus 1..MAX_PROMPTS free-text phrases",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "prompts": [1, MAX_PROMPTS],
    "prompt_chars": [1, MAX_PROMPT_CHARS],
    "prompt_tokens_per_query": [1, MAX_TEXT_TOKENS],
    "threshold": [0.0, 1.0],
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB, padded to a square with grey on the bottom/right and resized to 960x960 "
        "(CLIP mean/std); phrases stripped and lower-cased into one CLIP text query each (format_prompts, "
        "16-token limit per query); returned boxes are mapped back to input pixels"
    ),
}


def _check_inputs(image: Any, prompts: Any, threshold: Any) -> tuple[Image.Image, list[str], float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``detect`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    queries = format_prompts(prompts)
    checked = _check_threshold("threshold", threshold)
    return rgb, queries, checked


def validate_inputs(
    image: Image.Image,
    prompts: Sequence[str],
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, queries, checked = _check_inputs(image, prompts, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
                "n_prompts": len(prompts),
            }
        ],
        "queries": queries,
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[float]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (phrase -> xyxy reference box) the report carries one ``box_iou``
    entry per reference as sample-sanity geometry evidence; without them the verdict is
    ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    detections = list(result["detections"])
    base = {
        "task": "zero-shot (open-vocabulary, text-prompted) object detection",
        "decision_rule": (
            "each of the 3600 image patches proposes one box labelled with its best-matching text query; "
            "the box survives when the sigmoid of that best image-text logit reaches the threshold; the "
            "score is an uncalibrated sigmoid, not a probability, and is not exclusive across queries"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes on your own images with a phrase vocabulary matching the prompts, "
                "scored per object with box_iou and aggregated into precision/recall or mean average "
                "precision at a stated IoU threshold; no such labelled set ships with this repository"
            ),
        }
    metrics = []
    for phrase, box in ground_truth_boxes.items():
        ious = [box_iou(det["box"], box) for det in detections]
        best = max(range(len(ious)), key=ious.__getitem__) if ious else None
        metrics.append(
            {
                "id": "box_iou",
                "reference": phrase,
                "value": ious[best] if best is not None else 0.0,
                "matched_label": detections[best]["label"] if best is not None else None,
                "label_matches_reference": (detections[best]["label"] == phrase)
                if best is not None
                else False,
                "estimation": "one reference box per phrase on a single scene, no dispersion estimate",
            }
        )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial sample; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled box set from the deployment domain with a matching phrase vocabulary for any "
            "mean-average-precision or precision/recall claim"
        ),
    }


@dataclass
class Owlv2DetectionPipeline:
    """Text-prompted (open-vocabulary) object detection over the pinned OWLv2 base/16 ensemble checkpoint."""

    _runner: Callable[[Image.Image, list[str], float], list[dict[str, Any]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Owlv2DetectionPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Owlv2Processor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image, queries: list[str], threshold: float) -> list[dict]:
            # One text query per phrase; the CLIP tokenizer pads/truncates each to MAX_TEXT_TOKENS.
            inputs = processor(images=image, text=[queries], return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
            # The pinned processor scales boxes by max(height, width) itself because OWLv2 pads the
            # image to a square before resizing; passing the original size is the documented contract.
            result = processor.post_process_grounded_object_detection(
                outputs, threshold=threshold, target_sizes=[image.size[::-1]], text_labels=[queries]
            )[0]
            return [
                {"box": [float(v) for v in box.tolist()], "label": str(label), "score": float(score)}
                for box, label, score in zip(
                    result["boxes"], result["text_labels"], result["scores"], strict=True
                )
            ]

        return cls(runner, resolved_device)

    def detect(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        *,
        threshold: float = DETECTION_THRESHOLD,
    ) -> dict[str, Any]:
        """Detect the phrases in `prompts`; boxes are xyxy pixel coordinates in the input image."""
        rgb, queries, checked = _check_inputs(image, prompts, threshold)
        detections = self._runner(rgb, queries, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > MAX_DETECTIONS {MAX_DETECTIONS}"
            )
        for det in detections:
            if set(det) != {"box", "label", "score"} or len(det["box"]) != 4 or det["label"] not in queries:
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "queries": queries,
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
