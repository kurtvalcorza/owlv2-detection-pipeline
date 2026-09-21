"""Open-vocabulary detection metrics for labelled (image, phrases, boxes) records: per-phrase average precision
at an IoU threshold (VOC all-points interpolation), their mean, and precision / recall / F1 of the detections a
score threshold lets through, plus two non-learned baselines (no boxes; a spatial prior tiled over the image).
"""
# ruff: noqa: E501  -- metric rows are kept on single lines

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import box_iou

IOU_THRESHOLD = 0.5

METRIC_DEFINITIONS = {
    "map50": "mean over the phrases with at least one reference box of the per-phrase average precision at IoU >= iou_threshold (VOC all-points interpolation over the detections ranked by score across the whole set; a reference box is matched at most once, by the highest-scoring detection of its phrase that overlaps it enough)",
    "ap": "per-phrase average precision at IoU >= iou_threshold",
    "precision": "matched detections / all detections the score threshold let through (all phrases pooled)",
    "recall": "matched reference boxes / all reference boxes (all phrases pooled)",
    "f1": "harmonic mean of precision and recall",
}


def _match(predictions: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]], iou_threshold: float) -> list[tuple[float, str, bool]]:
    """Greedy matching in score order within one record: (score, phrase, matched) per prediction; a reference
    box is consumed by the first (highest-scoring) prediction of its phrase that reaches the IoU threshold."""
    ordered = sorted(predictions, key=lambda p: -float(p["score"]))
    taken = [False] * len(references)
    out = []
    for pred in ordered:
        best, best_iou = None, iou_threshold
        for k, ref in enumerate(references):
            if taken[k] or ref["prompt"] != pred["label"]:
                continue
            iou = box_iou(pred["box"], ref["box"])
            if iou >= best_iou:
                best, best_iou = k, iou
        if best is not None:
            taken[best] = True
        out.append((float(pred["score"]), str(pred["label"]), best is not None))
    return out


def average_precision(matches: Sequence[tuple[float, bool]], n_reference: int) -> float:
    """VOC all-points AP from (score, matched) pairs of one phrase over the whole set."""
    if n_reference == 0:
        return 0.0
    ordered = sorted(matches, key=lambda m: -m[0])
    tp = fp = 0
    recalls, precisions = [], []
    for _score, matched in ordered:
        tp += int(matched)
        fp += int(not matched)
        recalls.append(tp / n_reference)
        precisions.append(tp / (tp + fp))
    # make precision monotone from the right, then sum the area under the step curve
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    area, previous = 0.0, 0.0
    for recall, precision in zip(recalls, precisions, strict=True):
        area += (recall - previous) * precision
        previous = recall
    return float(area)


def detection_metrics(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    """Score one list of ``{box, label, score}`` detections per record against the record's ``boxes``."""
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} prediction lists for {len(records)} records")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    per_phrase: dict[str, list[tuple[float, bool]]] = {}
    n_reference: dict[str, int] = {}
    rows = []
    total_pred = total_tp = total_ref = 0
    for preds, record in zip(predictions, records, strict=True):
        refs = list(record["boxes"])
        for ref in refs:
            n_reference[ref["prompt"]] = n_reference.get(ref["prompt"], 0) + 1
        matched = _match(preds, refs, iou_threshold)
        tp = 0
        for score, phrase, hit in matched:
            per_phrase.setdefault(phrase, []).append((score, hit))
            tp += int(hit)
        rows.append({"id": record["id"], "n_reference": len(refs), "n_predicted": len(matched), "matched": tp})
        total_pred += len(matched)
        total_tp += tp
        total_ref += len(refs)
    phrases = sorted(set(n_reference) | set(per_phrase))
    per_prompt = {}
    for phrase in phrases:
        n_ref = n_reference.get(phrase, 0)
        matches = per_phrase.get(phrase, [])
        hits = sum(1 for _s, h in matches if h)
        per_prompt[phrase] = {
            "ap": round(average_precision(matches, n_ref), 4) if n_ref else None,
            "n_reference": n_ref,
            "n_predicted": len(matches),
            "precision": round(hits / len(matches), 4) if matches else 0.0,
            "recall": round(hits / n_ref, 4) if n_ref else None,
        }
    aps = [v["ap"] for v in per_prompt.values() if v["ap"] is not None]
    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(records),
        "n_reference_boxes": total_ref,
        "n_predicted_boxes": total_pred,
        "n_matched": total_tp,
        "iou_threshold": float(iou_threshold),
        "map50": round(statistics.fmean(aps), 4) if aps else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_prompt": per_prompt,
        "rows": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def empty_baseline(records: Sequence[Mapping[str, Any]], *, iou_threshold: float = IOU_THRESHOLD) -> dict[str, Any]:
    """No detections at all: every rate is 0 by construction (the floor)."""
    out = detection_metrics([[] for _ in records], records, iou_threshold=iou_threshold)
    out["baseline"] = "empty"
    return out


def grid_baseline(
    records: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]], *, iou_threshold: float = IOU_THRESHOLD
) -> dict[str, Any]:
    """A spatial prior with no image content: for each phrase the median box size in `reference` (the training
    records) tiled edge to edge over every image, all with the same score. Beating it shows the detector reads the
    image rather than the label statistics."""
    sizes: dict[str, list[tuple[float, float]]] = {}
    for record in reference:
        for b in record["boxes"]:
            x0, y0, x1, y1 = b["box"]
            sizes.setdefault(b["prompt"], []).append((x1 - x0, y1 - y0))
    prior = {p: (statistics.median(w for w, _ in v), statistics.median(h for _, h in v)) for p, v in sizes.items()}
    predictions = []
    for record in records:
        width, height = record["image"].size
        preds = []
        for phrase, (bw, bh) in prior.items():
            x = 0.0
            while x < width:
                y = 0.0
                while y < height:
                    preds.append({"box": [x, y, min(x + bw, width), min(y + bh, height)], "label": phrase, "score": 1.0})
                    y += bh
                x += bw
        predictions.append(preds)
    out = detection_metrics(predictions, records, iou_threshold=iou_threshold)
    out["baseline"] = "grid prior (median training box per phrase tiled over the image)"
    out["prior_box_sizes"] = {p: [round(w, 1), round(h, 1)] for p, (w, h) in prior.items()}
    return out
