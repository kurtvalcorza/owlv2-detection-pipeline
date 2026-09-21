"""Labelled (image, phrases, boxes) datasets for the adaptation contract: the digest-pinned BCCD sample, the
record contract and its structural validation, image-disjoint splitting, and the BYOD loader.

A record is ``{id, image, boxes}`` where ``image`` is a PIL image (sides within the pipeline's ceilings) and
``boxes`` a non-empty list of ``{prompt, box}``: the phrase that names the object (normalised like
``format_prompts`` normalises a query) and its ``[x0, y0, x1, y1]`` pixel box inside the image with positive
width and height. One record may carry many objects and many phrases; the phrase vocabulary of a dataset is the
sorted set of its prompts.

The default sample is BCCD (the Blood Cell Count and Detection dataset; **Public Domain**; 364 blood-smear
microscopy photographs at 416×416 with 4,886 boxes over three cell types) as exported by Roboflow to the Hugging
Face Hub (``keremberke/blood-cell-object-detection``) and converted to parquet by the Hub at an immutable revision:
the three parquet files (4.8 MB together) are fetched whole, each refused unless its SHA-256 and byte count match
the pin, and pooled into one corpus that the pipeline splits itself. The class names become the phrases
(``a platelet``, ``a red blood cell``, ``a white blood cell``). The domain gap is the point of the sample: the
queued clean-runtime run measures the frozen detector before asking what the class and box heads can learn from a
few hundred labelled microscopy images; this source-only candidate does not assume the result.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MODEL_ID, format_prompts, validate_image

CORPUS_NAME = "BCCD blood-cell detection (Roboflow export), all three splits pooled"
CORPUS_REPO = "keremberke/blood-cell-object-detection"
CORPUS_REVISION = "22cf1b9d2367e799ab54a16774a2266e4f8ce9a9"  # refs/convert/parquet commit on the Hub
CORPUS_CONFIG = "full"
CORPUS_LICENSE = "Public Domain (BCCD; Roboflow Universe 'blood-cell-detection-1ekwu' export of 2022-11-04)"
# path in the conversion tree -> (sha256, bytes, rows); the whole file is fetched and refused on any mismatch
CORPUS_FILES: dict[str, tuple[str, str, int, int]] = {
    "train": (f"{CORPUS_CONFIG}/train/0000.parquet", "4079046b96b0ba20d5b1f3f6b45293a8d463ff1efa8c0c60650f5233f04ca832", 3_378_661, 255),
    "validation": (f"{CORPUS_CONFIG}/validation/0000.parquet", "0bdced43e91fe6087d3ab381d04181c57d02137d1df90ea3ede8d8eb2b4bb737", 964_704, 73),
    "test": (f"{CORPUS_CONFIG}/test/0000.parquet", "dfbd812e2c5a18592c2cc56d3c610d9cd31e29271876b2b9dd51943cc6308d08", 476_944, 36),
}
CORPUS_ROWS = 364
CORPUS_BYTES = 4_820_309
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/"
DEFAULT_CACHE_DIR = Path("weights") / "bccd"

BCCD_CLASSES = ("platelets", "rbc", "wbc")  # the dataset's label ids 0, 1, 2
CLASS_PHRASES = {"platelets": "a platelet", "rbc": "a red blood cell", "wbc": "a white blood cell"}
SAMPLE_PHRASES = tuple(CLASS_PHRASES[c] for c in BCCD_CLASSES)

SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 260, "validation": 40, "test": 64}  # of the 364 pooled images
SAMPLE_DIGEST = "42d3b5df2df90c79e711f3d302dc46948bdb104f79d4cf81575221efa88d5fb2"
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MAX_BOXES_PER_RECORD = 200
MIN_BOX_SIDE = 1.0
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(
    *,
    cache_dir: str | Path | None = None,
    splits: Sequence[str] | None = None,
    opener: Any = None,
) -> dict[str, bytes]:
    """Fetch the pinned parquet files (cached under `cache_dir`); every file is refused unless its SHA-256 and byte
    count match `CORPUS_FILES`. Returns the parquet bytes per source split. `opener(url) -> bytes` can replace the
    HTTPS fetch (tests inject it)."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    wanted = list(splits) if splits is not None else list(CORPUS_FILES)
    out: dict[str, bytes] = {}
    for split in wanted:
        if split not in CORPUS_FILES:
            raise ValueError(f"unknown corpus split {split!r}; choose from {list(CORPUS_FILES)}")
        path, sha, size, _rows = CORPUS_FILES[split]
        cached = root / f"{split}.parquet"
        data = cached.read_bytes() if cached.is_file() else None
        if data is None or len(data) != size or _sha256_bytes(data) != sha:
            if opener is not None:
                data = opener(CORPUS_URL + path)
            else:
                request = urllib.request.Request(CORPUS_URL + path, headers={"User-Agent": "owlv2-detection-pipeline (DIMER sample fetch)"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
            if len(data) != size:
                raise ValueError(f"{path}: {len(data)} bytes, pinned {size}")
            digest = _sha256_bytes(data)
            if digest != sha:
                raise ValueError(f"{path}: sha256 {digest} != pinned {sha}")
            cached.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the parquet files into records ``{id, image, boxes, source_split, source_image_id}``; COCO
    ``[x, y, w, h]`` boxes become ``[x0, y0, x1, y1]`` clipped to the image, class ids become the phrases."""
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    for split, data in files.items():
        if split not in CORPUS_FILES:
            raise ValueError(f"unknown corpus split {split!r}")
        table = pq.read_table(io.BytesIO(data))
        expected_rows = CORPUS_FILES[split][3]
        if table.num_rows != expected_rows:
            raise ValueError(f"{split}: {table.num_rows} rows, pinned {expected_rows}")
        for i in range(table.num_rows):
            image_id = int(table.column("image_id")[i].as_py())
            image = Image.open(io.BytesIO(table.column("image")[i].as_py()["bytes"]))
            image.load()
            image = image.convert("RGB")
            objects = table.column("objects")[i].as_py()
            boxes = []
            for category, (x, y, w, h) in zip(objects["category"], objects["bbox"], strict=True):
                x0, y0 = max(0.0, float(x)), max(0.0, float(y))
                x1, y1 = min(float(image.width), float(x) + float(w)), min(float(image.height), float(y) + float(h))
                if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
                    continue
                boxes.append({"prompt": CLASS_PHRASES[BCCD_CLASSES[int(category)]], "box": [x0, y0, x1, y1]})
            if not boxes:
                continue
            records.append({"id": f"bccd-{split}-{image_id}", "image": image, "boxes": boxes, "source_split": split, "source_image_id": image_id})
    return records


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] = SAMPLE_SPLIT
) -> dict[str, list[dict[str, Any]]]:
    """Seeded, image-disjoint draw of `sizes` records per split from the pooled corpus (the Roboflow split
    membership is kept only as provenance)."""
    checked = validate_dataset(records, max_records=MAX_RECORDS)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    if len(unique) < sum(sizes.values()):
        raise ValueError(f"{len(unique)} distinct images < {sum(sizes.values())} requested")
    random.Random(seed).shuffle(unique)
    out: dict[str, list[dict[str, Any]]] = {}
    start = 0
    for name in ("train", "validation", "test"):
        out[name] = unique[start : start + sizes[name]]
        start += sizes[name]
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    """The default sample: fetch (or reuse) the pinned parquet files and draw the seeded splits."""
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str | Path):
        try:
            loaded = Image.open(image)
            loaded.load()
            return loaded
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{where}: cannot decode image {image!r}") from exc
    raise ValueError(f"{where}: image must be a PIL image or a path")


def coerce_box(box: Any, size: tuple[int, int], where: str = "box") -> list[float]:
    """A ``[x0, y0, x1, y1]`` pixel box inside an image of `size` (width, height) with positive extent."""
    if isinstance(box, str | bytes) or not isinstance(box, Sequence) or len(box) != 4:
        raise ValueError(f"{where} must be [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must hold four numbers") from exc
    if any(v != v for v in (x0, y0, x1, y1)):  # NaN
        raise ValueError(f"{where} must hold four finite numbers")
    width, height = size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(f"{where} [{x0}, {y0}, {x1}, {y1}] lies outside the {width}x{height} image")
    if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
        raise ValueError(f"{where} must be at least {MIN_BOX_SIDE} px wide and tall")
    return [x0, y0, x1, y1]


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/image/boxes")
    for key in ("id", "image", "boxes"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    try:
        image = validate_image(_open(record["image"], f"{where}.image"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    boxes = record["boxes"]
    if isinstance(boxes, Mapping | str | bytes) or not isinstance(boxes, Sequence) or not 1 <= len(boxes) <= MAX_BOXES_PER_RECORD:
        raise ValueError(f"{where}: boxes must be a list of 1..{MAX_BOXES_PER_RECORD} {{prompt, box}} entries")
    checked_boxes = []
    for k, entry in enumerate(boxes):
        if not isinstance(entry, Mapping) or "prompt" not in entry or "box" not in entry:
            raise ValueError(f"{where}.boxes[{k}] must be a mapping with prompt and box")
        if not isinstance(entry["prompt"], str):
            raise ValueError(f"{where}.boxes[{k}]: prompt must be a str")
        try:
            prompt = format_prompts([entry["prompt"]])[0]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.boxes[{k}]: {exc}") from exc
        checked_boxes.append({"prompt": prompt, "box": coerce_box(entry["box"], image.size, f"{where}.boxes[{k}].box")})
    item = {"id": rid, "image": image, "boxes": checked_boxes}
    for key in ("source_split", "source_image_id"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a labelled-box dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, boxes} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids = [], set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    counts: dict[str, int] = {}
    areas = []
    per_record = []
    for r in checked:
        per_record.append(len(r["boxes"]))
        for b in r["boxes"]:
            counts[b["prompt"]] = counts.get(b["prompt"], 0) + 1
            x0, y0, x1, y1 = b["box"]
            areas.append((x1 - x0) * (y1 - y0) / (r["image"].width * r["image"].height))
    widths = [r["image"].width for r in checked]
    heights = [r["image"].height for r in checked]
    prompts = sorted(counts)
    if len(prompts) > 16:
        raise ValueError(f"{len(prompts)} distinct prompts; the detector takes at most 16 queries per image")
    return {
        "records": checked,
        "n_records": len(checked),
        "n_boxes": sum(per_record),
        "n_prompts": len(prompts),
        "prompts": prompts,
        "boxes_per_prompt": counts,
        "boxes_per_record": {"min": min(per_record), "max": max(per_record), "mean": sum(per_record) / len(per_record)},
        "box_area_fraction": {"min": min(areas), "max": max(areas), "mean": sum(areas) / len(areas)},
        "image_width": {"min": min(widths), "max": max(widths)},
        "image_height": {"min": min(heights), "max": max(heights)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed) — the identity a split is made disjoint on."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def boxes_digest(boxes: Sequence[Mapping[str, Any]]) -> str:
    parts = sorted(f"{b['prompt']}:" + ",".join(f"{float(v):.2f}" for v in b["box"]) for b in boxes)
    return _sha256_bytes("|".join(parts).encode("utf-8"))


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, image digest, boxes digest)."""
    parts = sorted(f"{r['id']}:{image_digest(r['image'])}:{boxes_digest(r['boxes'])}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} distinct images are too few to split into train/validation/test")
    return {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Records from a directory or zip holding images and a `boxes.csv` with one row per object: the columns
    `file`, `prompt`, `x0`, `y0`, `x1`, `y1` (and optionally `id`); every listed file must exist and every image
    file must be listed at least once. Rows of one file become that record's boxes."""
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                if file.name in members:
                    raise ValueError(f"BYOD data has duplicate basename {file.name!r}; use unique flat names")
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    name = Path(info.filename).name
                    if name in members:
                        raise ValueError(f"BYOD data has duplicate basename {name!r}; use unique flat names")
                    members[name] = archive.read(info)  # flattened; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    if "boxes.csv" not in members:
        raise ValueError("BYOD data must include boxes.csv with the columns file, prompt, x0, y0, x1, y1")
    rows = list(csv.DictReader(io.StringIO(members["boxes.csv"].decode("utf-8-sig"))))
    columns = ("file", "prompt", "x0", "y0", "x1", "y1")
    if not rows or any(column not in rows[0] for column in columns):
        raise ValueError("boxes.csv must have the columns file, prompt, x0, y0, x1, y1")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = Path(str(row.get("file", "")).strip()).name
        if name not in members:
            raise ValueError(f"boxes.csv names a missing file: {name}")
        if name not in grouped:
            try:
                image = Image.open(io.BytesIO(members[name]))
                image.load()
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"BYOD file is not a decodable image: {name}") from exc
            rid = str(row.get("id", "") or "").strip()
            grouped[name] = {"id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(name).stem)[:64], "image": image.convert("RGB"), "boxes": []}
        grouped[name]["boxes"].append({"prompt": str(row.get("prompt", "")), "box": [row.get(k, "") for k in ("x0", "y0", "x1", "y1")]})
    unlisted = [n for n in members if n != "boxes.csv" and n not in grouped]
    if unlisted:
        raise ValueError(f"{len(unlisted)} file(s) have no boxes.csv row, e.g. {unlisted[0]}")
    return list(grouped.values())


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One row per object in the BYOD `boxes.csv` column layout plus extras (`file` names the id; the images
    themselves are not written)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file", "prompt", "x0", "y0", "x1", "y1", "width", "height", "source_split", "source_image_id"])
        for r in records:
            for b in r["boxes"]:
                writer.writerow([r["id"], f"{r['id']}.jpg", b["prompt"], *[round(float(v), 2) for v in b["box"]], r["image"].width, r["image"].height, r.get("source_split", ""), r.get("source_image_id", "")])
    return out
