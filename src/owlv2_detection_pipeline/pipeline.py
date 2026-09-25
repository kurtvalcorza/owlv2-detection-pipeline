"""Open-vocabulary (text-prompted) object detection with the pinned ``google/owlv2-base-patch16-ensemble``
checkpoint (OWLv2), plus the adaptation contract for labelled (image, phrases, boxes) records: corpus evaluation
(per-phrase AP at an IoU threshold, precision and recall at a score threshold), bounded fine-tuning of the class
and box heads on cached image features with the DETR-style matched loss, and a verified adapter artifact.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the OWLv2 architecture comes from the pinned ``transformers`` release, the
weights are SafeTensors, and no model-repository code is executed.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

# Adaptation contract. The trainable part is what the upstream authors trained on top of the CLIP towers for
# detection: the text-conditioned class head (dense projection + logit shift and scale) and the box head (a
# three-layer MLP with the per-patch box bias). The image tower, the text tower, the post-merge layer norm and
# the (unused at inference) objectness head stay frozen, so the image features can be cached once per record.
WEIGHTS_FILE = "model.safetensors"
PARAMETER_COUNT = 154_966_792
HEAD_PARAMETERS = 1_579_526  # class_head 395,266 + box_head 1,184,260 (12 tensors)
NUM_PATCHES = 3600  # 60 x 60 patch tokens = one box candidate each
_TRAINABLE_PREFIXES = ("class_head.", "box_head.")
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = 1
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 4
GRAD_CLIP = 1.0
IOU_THRESHOLD = 0.5
# DETR-style matched loss (the OWL-ViT training objective): sigmoid focal classification over every
# (patch, query) logit, L1 and generalised IoU on the matched boxes; the same weights build the matching cost.
LOSS_WEIGHTS = {"class": 2.0, "l1": 5.0, "giou": 2.0}
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHTS_FILE), None)


def _trainable_names(model: Any) -> list[str]:
    """The class and box head tensors; the CLIP towers, the post-merge layer norm and the objectness head stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError("artifact tensors must all belong to the OWLv2 class and box heads")
    adapter = manifest.get("adapter") or {}
    threshold = adapter.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0.0 <= threshold <= 1.0:
        raise ValueError("artifact manifest must record the score threshold the adapter was selected at")
    prompts = adapter.get("prompts")
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError("artifact manifest must record the phrase vocabulary the adapter was trained on")


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment of every row of a rectangular cost matrix (rows <= columns) to a distinct column —
    the shortest-augmenting-path Hungarian algorithm with numpy over the column dimension. Returns (row, column)
    pairs. Used to match each reference box to one patch candidate before the loss is computed."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be a 2-D array")
    n, m = cost.shape
    if n == 0:
        return []
    if n > m:
        raise ValueError(f"cost has more rows ({n}) than columns ({m})")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost must be finite")
    inf = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=np.int64)
    padded = np.empty((n + 1, m + 1))
    padded[1:, 1:] = cost
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = padded[i0] - u[i0] - v
            better = (~used) & (cur < minv)
            minv[better] = cur[better]
            way[better] = j0
            candidates = np.where(~used, minv, inf)
            j1 = int(np.argmin(candidates))
            delta = candidates[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def _cxcywh_to_xyxy(boxes: Any) -> Any:
    cx, cy, w, h = boxes.unbind(-1)
    import torch

    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _generalized_iou(a: Any, b: Any) -> Any:
    """Pairwise GIoU between xyxy boxes a (N, 4) and b (M, 4) -> (N, M)."""
    import torch

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-9)
    lt2 = torch.min(a[:, None, :2], b[None, :, :2])
    rb2 = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    enclosing = (wh2[..., 0] * wh2[..., 1]).clamp(min=1e-9)
    return iou - (enclosing - union) / enclosing


def _focal_terms(logits: Any) -> tuple[Any, Any]:
    """Per-logit sigmoid focal cost of a positive and of a negative target."""
    import torch

    prob = torch.sigmoid(logits)
    positive = FOCAL_ALPHA * (1 - prob) ** FOCAL_GAMMA * torch.nn.functional.softplus(-logits)
    negative = (1 - FOCAL_ALPHA) * prob**FOCAL_GAMMA * torch.nn.functional.softplus(logits)
    return positive, negative


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
    _model: Any = None
    _processor: Any = None
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
            source, revision=MODEL_REVISION, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        weight_sha256 = _weight_digest(root) if (root / MANIFEST_NAME).is_file() else None
        pipe = cls(None, resolved_device, model, processor, weight_sha256, None)  # type: ignore[arg-type]

        def runner(image: Image.Image, queries: list[str], threshold: float) -> list[dict]:
            return pipe.detect_batch([image], queries, threshold=threshold, batch_size=1)[0]

        pipe._runner = runner
        return pipe

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained")
        return self._model, self._processor

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

    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract: batched detection, corpus evaluation, bounded fine-tuning of the heads, artifacts
    # ------------------------------------------------------------------------------------------------------

    def _encode_images(self, images: Sequence[Image.Image]) -> Any:
        """The frozen image tower on a batch: the post-merge feature map (B, 60, 60, 768) in float32 on the device."""
        model, processor = self._require_model()
        import torch

        pixel_values = processor.image_processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            feature_map, _vision = model.image_embedder(pixel_values=pixel_values)
        return feature_map

    def _encode_queries(self, queries: Sequence[str]) -> Any:
        """The frozen text tower on the phrase vocabulary: (Q, 512) query embeddings on the device."""
        model, processor = self._require_model()
        import torch

        tokens = processor.tokenizer(list(queries), padding="max_length", max_length=MAX_TEXT_TOKENS, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return model.owlv2.get_text_features(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])

    def _heads(self, feature_map: Any, query_embeds: Any) -> tuple[Any, Any]:
        """The class and box heads on (cached) image features: exactly what `Owlv2ForObjectDetection.forward`
        computes after its frozen steps (parity asserted by the model-backed tests). Returns the (B, 3600, Q)
        logits and the (B, 3600, 4) boxes as cx, cy, w, h fractions of the padded square."""
        model, _ = self._require_model()
        import torch

        feature_map = feature_map.to(self.device, torch.float32)
        b, h, w, d = feature_map.shape
        image_feats = feature_map.reshape(b, h * w, d)
        queries = query_embeds.to(self.device, torch.float32).unsqueeze(0).expand(b, -1, -1)
        mask = torch.ones(b, queries.shape[1], dtype=torch.bool, device=self.device)
        logits, _class_embeds = model.class_predictor(image_feats, queries, mask)
        boxes = model.box_predictor(image_feats, feature_map)
        return logits, boxes

    @staticmethod
    def _postprocess(logits: Any, boxes: Any, queries: Sequence[str], sizes: Sequence[tuple[int, int]], threshold: float) -> list[list[dict[str, Any]]]:
        """The pinned processor's `post_process_grounded_object_detection` for the text path: the best query per
        patch, its sigmoid as the score, boxes scaled by max(width, height) because the image was padded to a
        square (parity with the processor asserted by the model-backed tests)."""
        import torch

        scores, labels = torch.sigmoid(logits).max(-1)
        xyxy = _cxcywh_to_xyxy(boxes)
        out = []
        for k, (width, height) in enumerate(sizes):
            keep = scores[k] >= threshold
            scale = float(max(width, height))
            boxes_k = (xyxy[k][keep] * scale).cpu().tolist()
            out.append([
                {"box": [float(v) for v in box], "label": str(queries[int(label)]), "score": float(score)}
                for box, label, score in zip(boxes_k, labels[k][keep].tolist(), scores[k][keep].tolist(), strict=True)
            ])
        return out

    def detect_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        *,
        threshold: float = DETECTION_THRESHOLD,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Detect the same phrases in many images, `batch_size` images per forward; one list of ``{box, label,
        score}`` (score-descending) per image, in order. With an injected runner the images go one by one through it."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        checked = [_check_inputs(image, prompts, threshold) for image in images]
        cut = _check_threshold("threshold", threshold)
        queries = checked[0][1] if checked else format_prompts(prompts)
        out: list[list[dict[str, Any]]] = []
        if self._model is None:
            for rgb, q, _ in checked:
                out.append(sorted(self._runner(rgb, q, cut), key=lambda d: -d["score"]))
                if progress is not None:
                    progress(len(out), len(checked))
            return out
        query_embeds = self._encode_queries(queries)
        for start in range(0, len(checked), batch_size):
            batch = [rgb for rgb, _, _ in checked[start : start + batch_size]]
            logits, boxes = self._heads(self._encode_images(batch), query_embeds)
            for dets in self._postprocess(logits.detach(), boxes.detach(), queries, [im.size for im in batch], cut):
                if len(dets) > MAX_DETECTIONS:
                    raise RuntimeError(f"backend returned {len(dets)} detections > MAX_DETECTIONS {MAX_DETECTIONS}")
                out.append(sorted(dets, key=lambda d: -d["score"]))
            if progress is not None:
                progress(len(out), len(checked))
        return out

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        prompts: Sequence[str] | None = None,
        threshold: float = DETECTION_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Detect the phrase vocabulary (`prompts`, default: every phrase the records use) in every validated record
        and score the detections above `threshold` against the record boxes with ``metrics.detection_metrics``
        (per-phrase AP at `iou_threshold`, their mean, precision, recall and F1). Works with an injected runner too."""
        from .metrics import detection_metrics
        from .samples import validate_dataset

        manifest = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)
        checked = manifest["records"]
        vocabulary = format_prompts(list(prompts) if prompts is not None else manifest["prompts"])
        missing = sorted(set(manifest["prompts"]) - set(vocabulary))
        if missing:
            raise ValueError(f"records use phrases outside the evaluated vocabulary: {missing}")
        started = time.perf_counter()
        predictions = self.detect_batch([r["image"] for r in checked], vocabulary, threshold=threshold, batch_size=batch_size, progress=progress)
        metrics = detection_metrics(predictions, checked, iou_threshold=iou_threshold)
        metrics.update(
            {
                "threshold": float(threshold),
                "prompts": vocabulary,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _cache(self, records: Sequence[Mapping[str, Any]], batch_size: int, progress: Callable[[int, int], None] | None = None) -> Any:
        """Run the frozen image tower once per record and keep the feature maps in half precision on the host."""
        import torch

        maps = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            maps.append(self._encode_images([r["image"] for r in batch]).to("cpu", torch.float16))
            if progress is not None:
                progress(min(start + batch_size, len(records)), len(records))
        return torch.cat(maps)

    @staticmethod
    def _targets(record: Mapping[str, Any], vocabulary: Sequence[str]) -> tuple[Any, Any]:
        """Reference boxes as (class index, cx cy w h fractions of the padded square) — the box parametrisation the
        head predicts (the processor pads the image to a square on the bottom/right, so both axes divide by the
        longer side)."""
        import torch

        scale = float(max(record["image"].size))
        index = {p: i for i, p in enumerate(vocabulary)}
        classes = torch.tensor([index[b["prompt"]] for b in record["boxes"]], dtype=torch.long)
        xyxy = torch.tensor([b["box"] for b in record["boxes"]], dtype=torch.float32) / scale
        cxcywh = torch.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], -1)
        return classes, cxcywh

    def _matched_loss(self, logits: Any, boxes: Any, targets: Sequence[tuple[Any, Any]]) -> tuple[Any, dict[str, float]]:
        """DETR-style loss on one batch: Hungarian matching of each reference box to one patch under the class,
        L1 and GIoU costs, then sigmoid focal loss over every (patch, query) logit (matched pairs positive), L1 and
        GIoU on the matched boxes, all normalised by the number of reference boxes in the batch."""
        import torch

        device = logits.device
        n_boxes = max(1, sum(len(c) for c, _ in targets))
        class_target = torch.zeros_like(logits)
        l1_total = torch.zeros((), device=device)
        giou_total = torch.zeros((), device=device)
        for k, (classes, gt) in enumerate(targets):
            if len(classes) == 0:
                continue
            classes, gt = classes.to(device), gt.to(device)
            with torch.no_grad():
                positive, negative = _focal_terms(logits[k].detach())
                class_cost = (positive - negative)[:, classes].transpose(0, 1)  # (n_gt, patches)
                l1_cost = torch.cdist(gt, boxes[k].detach(), p=1)
                giou_cost = -_generalized_iou(_cxcywh_to_xyxy(gt), _cxcywh_to_xyxy(boxes[k].detach()))
                cost = LOSS_WEIGHTS["class"] * class_cost + LOSS_WEIGHTS["l1"] * l1_cost + LOSS_WEIGHTS["giou"] * giou_cost
            pairs = hungarian(cost.cpu().numpy())
            rows = torch.tensor([r for r, _ in pairs], device=device)
            cols = torch.tensor([c for _, c in pairs], device=device)
            class_target[k, cols, classes[rows]] = 1.0
            matched = boxes[k][cols]
            l1_total = l1_total + (matched - gt[rows]).abs().sum()
            giou_total = giou_total + (1 - torch.diagonal(_generalized_iou(_cxcywh_to_xyxy(matched), _cxcywh_to_xyxy(gt[rows])))).sum()
        positive, negative = _focal_terms(logits)
        focal = (class_target * positive + (1 - class_target) * negative).sum() / n_boxes
        l1 = l1_total / n_boxes
        giou = giou_total / n_boxes
        total = LOSS_WEIGHTS["class"] * focal + LOSS_WEIGHTS["l1"] * l1 + LOSS_WEIGHTS["giou"] * giou
        return total, {"focal": float(focal), "l1": float(l1), "giou": float(giou)}

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        *,
        prompts: Sequence[str] | None = None,
        epochs: int = 8,
        lr: float = 1e-4,
        batch_size: int = 8,
        seed: int = 0,
        threshold: float = DETECTION_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the OWLv2 class and box heads on labelled (image, phrases, boxes) records with the
        DETR-style matched loss the upstream detection heads were trained with (Hungarian matching; sigmoid focal
        classification, L1 and GIoU box terms). The frozen image tower is run once per record under no gradient and
        its feature maps cached (half precision on the host), the frozen text tower once per phrase, so each step
        runs only the heads; the logits equal the full model's. AdamW (no weight decay), gradient clipping at
        `GRAD_CLIP`, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen model's validation
        rates at `threshold`; the epoch with the highest validation mAP@`iou_threshold` (the earliest on ties) is
        kept. On any exception the frozen heads are restored."""
        from .metrics import detection_metrics
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise ValueError("epochs must be an int in 1..100")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be an int in 1..128")
        if not isinstance(lr, int | float) or isinstance(lr, bool) or not 0 < lr <= 1e-2:
            raise ValueError("lr must be a number in (0, 1e-2]")
        cut = _check_threshold("threshold", threshold)
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        train_manifest = validate_dataset(train)
        train_checked = train_manifest["records"]
        val_manifest = validate_dataset(val, min_records=1) if val is not None else None
        val_checked = val_manifest["records"] if val_manifest is not None else None
        vocabulary = format_prompts(list(prompts) if prompts is not None else train_manifest["prompts"])
        used = set(train_manifest["prompts"]) | (set(val_manifest["prompts"]) if val_manifest is not None else set())
        missing = sorted(used - set(vocabulary))
        if missing:
            raise ValueError(f"records use phrases outside the training vocabulary: {missing}")
        model, _processor = self._require_model()
        import torch

        started = time.perf_counter()
        names = _trainable_names(model)
        params = {name: param for name, param in model.named_parameters() if name in set(names)}
        n_trainable = sum(p.numel() for p in params.values())
        backup = {name: param.detach().clone() for name, param in params.items()}
        previous_adapter = self.adapter
        cudnn_flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        try:
            model.eval()
            query_embeds = self._encode_queries(vocabulary)
            cache = self._cache(train_checked, EVAL_BATCH_SIZE)
            cached_val = self._cache(val_checked, EVAL_BATCH_SIZE) if val_checked is not None else None
            cache_seconds = round(time.perf_counter() - started, 3)
            targets = [self._targets(r, vocabulary) for r in train_checked]

            def score_val() -> dict[str, Any] | None:
                if val_checked is None or cached_val is None:
                    return None
                predictions = []
                with torch.no_grad():
                    for start in range(0, len(val_checked), EVAL_BATCH_SIZE):
                        logits, boxes = self._heads(cached_val[start : start + EVAL_BATCH_SIZE], query_embeds)
                        predictions.extend(self._postprocess(logits, boxes, vocabulary, [r["image"].size for r in val_checked[start : start + EVAL_BATCH_SIZE]], cut))
                m = detection_metrics(predictions, val_checked, iou_threshold=iou_threshold)
                return {k: m[k] for k in ("map50", "precision", "recall", "f1", "n", "n_predicted_boxes")}

            for name, param in model.named_parameters():
                param.requires_grad_(name in params)
            history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}]
            if progress is not None:
                progress(history[-1])
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("map50", -1.0)
            best_state = {name: param.detach().clone() for name, param in params.items()}
            optimizer = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            order = list(range(len(train_checked)))
            for epoch in range(1, epochs + 1):
                rng.shuffle(order)
                model.class_head.train()
                model.box_head.train()
                total, steps, parts = 0.0, 0, {"focal": 0.0, "l1": 0.0, "giou": 0.0}
                for start in range(0, len(order), batch_size):
                    idx = order[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits, boxes = self._heads(cache[idx], query_embeds)
                    loss, terms = self._matched_loss(logits, boxes, [targets[i] for i in idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()), GRAD_CLIP)
                    optimizer.step()
                    total += float(loss.detach())
                    for key in parts:
                        parts[key] += terms[key]
                    steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": round(total / max(steps, 1), 5), "loss_terms": {k: round(v / max(steps, 1), 5) for k, v in parts.items()}, "val": score_val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                score = (entry["val"] or {}).get("map50")
                if val_checked is None or (score is not None and score > best_score):
                    best_epoch, best_score = epoch, score if score is not None else best_score
                    best_state = {name: param.detach().clone() for name, param in params.items()}
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(best_state[name])
        except BaseException:
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(backup[name])
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "threshold": cut,
            "iou_threshold": float(iou_threshold),
            "prompts": list(vocabulary),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "highest validation mAP at the IoU threshold" if val_checked is not None else "final epoch (no validation split)",
            "loss": f"Hungarian-matched sigmoid focal (alpha {FOCAL_ALPHA}, gamma {FOCAL_GAMMA}) + L1 + GIoU with weights {LOSS_WEIGHTS}, normalised by the reference-box count; computed on cached image features",
            "lr": float(lr),
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests, the threshold, the
        phrase vocabulary and the training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHTS_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the class and box heads."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Owlv2DetectionPipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
