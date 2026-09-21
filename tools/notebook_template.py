"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E detection-adaptation workflow: the pinned google/owlv2-base-patch16-ensemble
snapshot is digest-verified and loaded, the 364 Public-Domain BCCD blood-smear photographs with their 4,886 cell
boxes are fetched as three digest-pinned parquet files and turned into (image, phrases, boxes) records, the records
are validated and split by image, a synthetic scene of drawn shapes is detected through the inference contract, the
frozen model is scored over the held-out records (per-phrase AP at IoU 0.5, mAP, precision and recall at the score
threshold) beside an empty and a grid-prior baseline, a bounded fine-tuning of the class and box heads runs on cached
image features with validation-mAP epoch selection, the held-out split is scored again, six held-out records and the
drawn scene are re-run with the adapted model, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "owlv2_detection_pipeline",
    "repo_name": "owlv2-detection-pipeline",
    "stem": "owlv2_detection",
    "notebook_name": "owlv2_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "Owlv2DetectionPipeline",
    "weights_key": "owlv2-base-patch16-ensemble",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "OWLv2 base/16 ensemble — DIMER E2E open-vocabulary detection adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/owlv2-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/owlv2-detection-pipeline/blob/main/tutorials/owlv2_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlv2--base--patch16--ensemble-ffcc4d?style=flat",
            "https://huggingface.co/google/owlv2-base-patch16-ensemble",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2306.09683-b31b1b.svg", "https://arxiv.org/abs/2306.09683"),
    ],
    "capability": "zero-shot (open-vocabulary, text-prompted) object detection — one image plus 1–16 free-text phrases → score-ordered boxes labelled with the phrase they matched — and bounded supervised fine-tuning of the OWLv2 class and box heads on labelled (image, phrases, boxes) records, using the pinned `google/owlv2-base-patch16-ensemble` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `google/owlv2-base-patch16-ensemble` snapshot (a 620 MB `model.safetensors`; no pickle is opened anywhere), "
        "fetches the three BCCD parquet files from the Hugging Face Hub at an immutable revision (4.8 MB together; each "
        "refused on any SHA-256 or byte-count mismatch), turns the 364 blood-smear photographs into (image, phrases, boxes) "
        "records and splits them by image into 260 / 40 / 64, detects three drawn shapes through the inference contract with "
        "an input manifest and a rejection probe, scores the frozen model over the 64 held-out records (per-phrase average "
        "precision at IoU 0.5, their mean, precision and recall at the score threshold) beside an empty and a grid-prior "
        "baseline, runs a bounded fine-tuning of the class and box heads on cached image features with validation-mAP epoch "
        "selection, scores the held-out records again, re-runs six held-out records and the drawn scene with the adapted "
        "model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify "
        "detection parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload "
        "dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). A CUDA runtime is used automatically when present. "
        "Supported-runtime timing and model-backed results have not yet been recorded for this candidate; the queued "
        "Kaggle Tesla T4 clean-runtime run is the execution gate."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of images plus a `boxes.csv` (`file`, `prompt`, `x0`, `y0`, `x1`, `y1`, optional `id`; one row per object, at least "
        "eight images, at most 16 distinct phrases). The records pass through the same validation, image-disjoint split, "
        "baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the BCCD sample. Uploaded "
        "files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "OWLv2 is a CLIP ViT-B/16 image tower and a CLIP text tower joined by two small heads: the image is padded to a "
        "square, resized to 960 × 960 and split into 60 × 60 = 3,600 patches, each of which the **box head** (a three-layer "
        "MLP with a per-patch position bias) turns into one box and the **class head** (a projection plus a learned logit "
        "shift and scale) scores against every phrase embedding; a patch's box survives when the **sigmoid** of its best "
        "image–text logit reaches a caller-owned threshold, and is labelled with that phrase (154,966,792 parameters in all, "
        "published under the **Apache-2.0** licence). The score is **an uncalibrated sigmoid**, there is **no non-maximum "
        "suppression**, and **the threshold is a caller-owned request parameter**.\n\n"
        "What this notebook adds to inference is **adaptation of the class and box heads on labelled (image, phrases, "
        "boxes) records**. The records are blood-smear microscopy photographs from BCCD, each with every platelet, red "
        "blood cell and white blood cell boxed by hand — an image family and a vocabulary the detector never saw, and on "
        "the queued clean-runtime run will measure the frozen model before interpreting adaptation. The question is "
        "narrow and honest: does a bounded fine-tuning of "
        "the 1,579,526-parameter heads on 260 records — the towers frozen, the DETR-style matched loss the upstream heads "
        "were trained with — move the held-out **mAP**, **per-phrase AP**, **precision** and **recall** on an image-disjoint "
        "test split past the frozen model and two **non-adapted baselines**, and what does it do to the drawn shapes the "
        "same heads detect? Nothing here is a claim about your images or your phrases: it is one seeded split of one "
        "small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 9-file manifest with the tokenizer and "
        "processor files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files "
        "before the processor or the model is constructed. The pipeline runs in **float32 on every device**: the adapter is "
        "trained in float32 and overlays without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled box set, turn it into (image, phrases, boxes) records, validate "
        "it and split it by image without leakage; detect in a drawn scene through the public API and read the output "
        "contract correctly (an uncalibrated sigmoid, a caller-owned threshold, no non-maximum suppression, a "
        "`sample-sanity` report only against boxes you drew yourself); measure the frozen model's held-out per-phrase AP, "
        "mAP, precision and recall beside two non-adapted baselines; run a bounded fine-tuning of the heads with the "
        "Hungarian-matched focal + L1 + GIoU loss, explicit hyperparameters and validation-based epoch selection; evaluate "
        "on an image-disjoint test split; look at the adapted boxes next to the references and at what the drawn scene does "
        "after the shared heads were tuned; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "instance or semantic segmentation, tracking, OCR, captioning, image-guided (one-shot) detection, non-maximum "
        "suppression, threshold tuning (the score threshold is fixed at `DETECTION_THRESHOLD` and the IoU threshold at "
        "`IOU_THRESHOLD` for every measurement here), fine-tuning of the CLIP image or text towers, COCO-style AP averaged "
        "over IoU thresholds, evaluation on LVIS or a detection benchmark proper (only one seeded 364-record sample is scored "
        "here), and any claim that blood-cell boxes stand in for your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Kaggle, Python 3.12; CPU or CUDA). The default path uses CUDA automatically when present. The image tower runs `EVAL_BATCH_SIZE` records per forward at 960 × 960. Supported-runtime timing has not yet been recorded for this candidate; the queued Kaggle Tesla T4 run is the execution gate. The pinned `torch==2.14.0` install and the 620 MB checkpoint are the large downloads; the parquet files are 4.8 MB. The feature cache holds about 1.7 GB of half-precision tensors on the host for 300 records (5.5 MB each).",
        "- **Knowledge:** basic Python, NumPy and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union, average precision and a precision/recall pair at one threshold measure and why 64 records from one draw give no dispersion; why a self-drawn scene is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, boxes}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4,096 px and `boxes` 1..200 entries `{prompt, box}`: a phrase of at most 48 characters (normalised like a query; at most 16 distinct phrases per dataset) and an `[x0, y0, x1, y1]` pixel box inside the image at least 1 px wide and tall. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `boxes.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded and every phrase and box checked, but nothing checks that a box outlines what its phrase names — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads `full/train/0000.parquet`, `full/validation/0000.parquet` and `full/test/0000.parquet` from `https://huggingface.co/datasets/keremberke/blood-cell-object-detection/resolve/<revision>/` at the immutable parquet-conversion revision `22cf1b9d…` (4.8 MB together), each pinned by SHA-256 and byte count in the carried `samples.py` and refused on any mismatch. BCCD is Public Domain (the Roboflow Universe export of 2022-11-04); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. BCCD records, the phrases and the split\n\n"
                "`fetch_corpus` returns the three pinned parquet files from the cache under `weights/bccd/` or the Hub at the "
                "pinned parquet-conversion revision — every cached file is re-hashed and every fetched file refused on any "
                "SHA-256 or byte-count mismatch — and `read_corpus` turns each row into a record: the photograph and one "
                "`{{prompt, box}}` per annotated cell, the COCO `[x, y, w, h]` boxes converted to `[x0, y0, x1, y1]` and the "
                "three class ids to the phrases `a platelet`, `a red blood cell`, `a white blood cell`. The Roboflow split "
                "membership is kept only as provenance: `build_sample_dataset` pools the 364 images and draws a seeded "
                "image-level split (260 / 40 / 64). `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training split's box "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 364 records at 416 × 416 with about thirteen boxes each (red blood cells dominate: roughly eleven "
                "per image, one white blood cell, one platelet), three digests, and four refusal probes — a duplicate id, a "
                "box outside its image, a record without boxes, and a dataset too small to use — each rejected before the "
                "model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/bccd')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'files': sorted(corpus_files), 'bytes': sum(len(v) for v in corpus_files.values()), 'records': len(corpus), 'boxes': sum(len(r['boxes']) for r in corpus), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "PROMPTS = sorted(set(dataset_manifests['train']['prompts']) | set(dataset_manifests['validation']['prompts']) | set(dataset_manifests['test']['prompts']))\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'prompts': PROMPTS}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'boxes': manifest['n_boxes'], 'boxes_per_prompt': manifest['boxes_per_prompt'], 'boxes_per_record': {{k: round(v, 1) for k, v in manifest['boxes_per_record'].items()}}, 'box_area': {{k: round(v, 3) for k, v in manifest['box_area_fraction'].items()}}, 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n\n\n"
                "PALETTE = {{p: c for p, c in zip(PROMPTS, [(220, 40, 40), (40, 70, 200), (60, 179, 75), (250, 200, 30), (160, 60, 200), (0, 170, 170)] * 3, strict=False)}}\n\n\n"
                "def draw_boxes(image, boxes, width=2, key='prompt', dashed=False):\n"
                "    \"\"\"Outline each {{prompt|label, box}} on a copy of the image in the phrase's colour.\"\"\"\n"
                "    out = image.convert('RGB').copy()\n"
                "    d = ImageDraw.Draw(out)\n"
                "    for b in boxes:\n"
                "        colour = PALETTE.get(b[key], (20, 20, 20))\n"
                "        x0, y0, x1, y1 = b['box']\n"
                "        if dashed:\n"
                "            for x in np.arange(x0, x1, 6):\n"
                "                d.line([(x, y0), (min(x + 3, x1), y0)], fill=colour, width=width)\n"
                "                d.line([(x, y1), (min(x + 3, x1), y1)], fill=colour, width=width)\n"
                "            for y in np.arange(y0, y1, 6):\n"
                "                d.line([(x0, y), (x0, min(y + 3, y1))], fill=colour, width=width)\n"
                "                d.line([(x1, y), (x1, min(y + 3, y1))], fill=colour, width=width)\n"
                "        else:\n"
                "            d.rectangle([x0, y0, x1, y1], outline=colour, width=width)\n"
                "    return out\n\n\n"
                "example = train_records[0]\n"
                "draw_boxes(example['image'], example['boxes']).save('outputs/{stem}_example_record.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'boxes': len(example['boxes']), 'phrases': sorted({{b['prompt'] for b in example['boxes']}})}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'box outside the image': [{{**train_records[0], 'boxes': [{{'prompt': 'a platelet', 'box': [0, 0, train_records[0]['image'].width + 10, 20]}}]}}, *train_records[1:8]],\n"
                "    'record without boxes': [{{**train_records[0], 'boxes': []}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Detect in a drawn scene through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a deterministic 640 × 480 "
                "scene drawn in code — a black rectangle at `[80, 120, 280, 360]`, a red disc with bounding box "
                "`[380, 140, 560, 320]` and a blue triangle with bounding box `[200, 380, 360, 460]` — with the prompts naming "
                "them and one absent phrase (`a green star`) on purpose; a different image family from the blood smears, "
                "and a scene the adapted model will detect in again in Section 9. `validate_inputs` applies exactly the "
                "checks `detect` applies (one image with sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, 1..`MAX_PROMPTS` distinct "
                "phrases of at most `MAX_PROMPT_CHARS` characters, a threshold in [0, 1]) and returns an input manifest; an "
                "over-long phrase is validated too and its rejection recorded as a finding. `detect` returns "
                "`{{box, label, score}}` detections **ordered by descending score** — **the scores are an uncalibrated "
                "sigmoid** of the best image–text logit, the threshold is a **caller-owned request parameter**, and there "
                "is **no non-maximum suppression**, so one object can surface as several boxes at a low threshold. "
                "`evaluation_report` with the drawn boxes is `sample-sanity`: one `box_iou` per drawn phrase, plumbing "
                "evidence for one drawing — a detection benchmark needs labelled boxes, which Section 6 supplies. The "
                "inference-only card recorded IoUs of 0.95–0.97 on the three drawn shapes."
            ),
            "code": (
                "def synthetic_scene(width=640, height=480):\n"
                "    \"\"\"Three coloured shapes drawn with Pillow (no text); returns image + {{phrase: xyxy reference box}}.\"\"\"\n"
                "    image = Image.new('RGB', (width, height), (128, 128, 128))\n"
                "    d = ImageDraw.Draw(image)\n"
                "    boxes = {{'a black rectangle': [80.0, 120.0, 280.0, 360.0], 'a red circle': [380.0, 140.0, 560.0, 320.0], 'a blue triangle': [200.0, 380.0, 360.0, 460.0]}}\n"
                "    d.rectangle(boxes['a black rectangle'], fill=(30, 30, 30))\n"
                "    d.ellipse(boxes['a red circle'], fill=(220, 30, 30))\n"
                "    d.polygon([(280, 380), (200, 460), (360, 460)], fill=(30, 60, 220))\n"
                "    return image, boxes\n\n\n"
                "scene, scene_boxes = synthetic_scene()\n"
                "scene_prompts = list(scene_boxes) + ['a green star']  # one absent phrase on purpose\n"
                "scene_name = 'synthetic_scene_640x480'\n"
                "scene_sha256 = hashlib.sha256(np.asarray(scene).tobytes()).hexdigest()\n"
                "THRESHOLD = DETECTION_THRESHOLD\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_DETECTIONS': MAX_DETECTIONS, 'NUM_PATCHES': NUM_PATCHES, 'DETECTION_THRESHOLD': DETECTION_THRESHOLD, 'IOU_THRESHOLD': IOU_THRESHOLD, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'EVAL_BATCH_SIZE': EVAL_BATCH_SIZE, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene, scene_prompts, threshold=THRESHOLD, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(scene, ['x' * (MAX_PROMPT_CHARS + 1)])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': scene_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def detect_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.detect(scene, scene_prompts, threshold=THRESHOLD)\n"
                "    seconds = round(time.perf_counter() - started, 3)\n"
                "    checks = {{\n"
                "        'score_ordered': all(a['score'] >= b['score'] for a, b in zip(result['detections'], result['detections'][1:], strict=False)),\n"
                "        'labels_are_queries': all(d['label'] in result['queries'] for d in result['detections']) and result['queries'] == [q.lower() for q in scene_prompts],\n"
                "        'boxes_inside_the_image': all(0 <= d['box'][0] <= d['box'][2] <= scene.width + 1 and 0 <= d['box'][1] <= d['box'][3] <= scene.height + 1 for d in result['detections']),\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'detect output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, scene_boxes, sample_kind='synthetic (drawn in this notebook)')\n"
                "    ious = {{m['reference']: {{'iou': round(m['value'], 3), 'label_matches': m['label_matches_reference']}} for m in report['metrics']}}\n"
                "    per_label = {{q: sum(1 for d in result['detections'] if d['label'] == q) for q in result['queries']}}\n"
                "    top = [(d['label'], round(d['score'], 3), [round(v, 1) for v in d['box']]) for d in result['detections'][:5]]\n"
                "    with open(f'outputs/{stem}_scene_{{label}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'detections': result['detections'], 'report': report}}, handle, indent=2, ensure_ascii=False)\n"
                "    annotated = scene.copy()\n"
                "    marker = ImageDraw.Draw(annotated)\n"
                "    for d in result['detections']:\n"
                "        marker.rectangle(d['box'], outline=(0, 255, 0), width=2)\n"
                "        marker.text((d['box'][0] + 2, d['box'][1] + 2), f\"{{d['label']}} {{d['score']:.2f}}\", fill=(0, 255, 0))\n"
                "    annotated.save(f'outputs/{stem}_scene_{{label}}.png')\n"
                "    summary = {{'n_detections': len(result['detections']), 'per_label': per_label, 'top': top, 'box_iou': ious}}\n"
                "    print({{label: {{'seconds': seconds, 'checks': checks, **summary, 'verdict': report['verdict']}}}})\n"
                "    return summary, seconds, checks, report\n\n\n"
                "frozen_scene, frozen_scene_seconds, frozen_scene_checks, frozen_scene_report = detect_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test records\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `detection_metrics` (carried in "
                "`metrics.py`): per phrase, the **average precision at IoU ≥ 0.5** — the detections of that phrase ranked "
                "by score across the whole split, each reference box matched at most once, the area under the "
                "precision–recall step curve (VOC all-points) — and their mean, **mAP@0.5**, the measure the epoch is "
                "selected on; and, over all phrases pooled, the **precision** and **recall** of the detections the score "
                "threshold let through and their F1. The **empty** baseline predicts no box and scores 0 by construction — "
                "the floor. The **grid-prior** baseline tiles each phrase's median training box edge to edge over every "
                "image with one constant score, what the label statistics buy without looking at the image. The **frozen "
                "model** is scored by `pipe.evaluate`, which detects the three phrases in every record in batches of "
                "`EVAL_BATCH_SIZE`, keeps the boxes whose sigmoid reaches `THRESHOLD`, and scores them. Do not assume the "
                "frozen detector's behaviour in advance: read its mAP, per-phrase APs, precision, recall, box counts and "
                "per-record rows from this run."
            ),
            "code": (
                "METRICS = ('map50', 'precision', 'recall', 'f1')\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_grid = grid_baseline(test_records, train_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'grid_baseline': {{k: round(baseline_grid[k], 3) for k in METRICS}}, 'prior_box_sizes': baseline_grid['prior_box_sizes'], 'n_predicted_boxes': baseline_grid['n_predicted_boxes'], 'note': baseline_grid['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'reference_boxes': frozen_test['n_reference_boxes'], 'predicted_boxes': frozen_test['n_predicted_boxes'], 'matched': frozen_test['n_matched'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_per_prompt': frozen_test['per_prompt']}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for row in frozen_test['rows'][:4]:\n"
                "    print(row)"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the heads\n\n"
                "`pipe.adapt` trains only the two detection heads — the class head (a 768 → 512 projection with a learned "
                "logit shift and scale) and the box head (a three-layer MLP): 1,579,526 of 154,966,792 parameters — while "
                "the CLIP image and text towers, the post-merge layer norm and the objectness head stay frozen, the split the "
                "upstream authors trained with. The loss is the **DETR-style matched loss** the OWL-ViT heads were trained "
                "with: each reference box is assigned to one of the 3,600 patch candidates by **Hungarian matching** under a "
                "cost of focal classification, L1 and generalised-IoU terms, then the **sigmoid focal loss** is taken over "
                "every (patch, phrase) logit with the matched pairs as positives, and **L1 and GIoU** over the matched boxes, "
                "all normalised by the number of reference boxes. Because the towers are frozen, their outputs — the "
                "60 × 60 × 768 feature map per image and the three phrase embeddings — are computed once under no gradient "
                "and cached in half precision on the host (the **frozen-tower cache**), and each step runs only the heads on "
                "those cached features: the logits equal the full model's. AdamW without weight decay at a fixed learning "
                "rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen "
                "model's validation rates; every epoch is scored on the 40 validation records at `THRESHOLD`, and the epoch "
                "with the **highest validation mAP** (the earliest on ties) is kept.\n\n"
                "Read the emitted epoch history rather than assuming improvement: it records the validation mAP and loss "
                "for every epoch, then restores the earliest epoch with the highest validation mAP."
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('loss_terms'):\n"
                "        row['terms'] = {{k: round(v, 3) for k, v in entry['loss_terms'].items()}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "        row['val_boxes'] = entry['val']['n_predicted_boxes']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, prompts=PROMPTS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, threshold=THRESHOLD, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'threshold': adapt_result['threshold'], 'iou_threshold': adapt_result['iou_threshold'], 'prompts': adapt_result['prompts'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test records were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **mAP@0.5** first (the measure the epoch was selected on), then the "
                "**per-phrase APs**, then **precision** and **recall** together — a gain in one at the cost of the "
                "other is a moved threshold, not a better detector), then the number of predicted boxes against the "
                "reference count. The cell asserts the adapted mAP is at least the frozen one and above the grid-prior "
                "baseline. Sixty-four records from one seeded split give **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a result on one "
                "blood-smear set's three cell types says nothing about other phrases, other images or your data until you "
                "measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'grid': round(baseline_grid[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['per_prompt_ap'] = {{p: {{'frozen': frozen_test['per_prompt'][p]['ap'], 'adapted': adapted_test['per_prompt'][p]['ap'], 'n_reference': adapted_test['per_prompt'][p]['n_reference']}} for p in PROMPTS}}\n"
                "comparison['boxes'] = {{'reference': adapted_test['n_reference_boxes'], 'frozen_predicted': frozen_test['n_predicted_boxes'], 'adapted_predicted': adapted_test['n_predicted_boxes'], 'frozen_matched': frozen_test['n_matched'], 'adapted_matched': adapted_test['n_matched']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'threshold': THRESHOLD,\n"
                "    'iou_threshold': IOU_THRESHOLD,\n"
                "    'prompts': PROMPTS,\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'grid': {{k: v for k, v in baseline_grid.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_record': [{{**frozen_row, 'adapted_n_predicted': adapted_row['n_predicted'], 'adapted_matched': adapted_row['matched']}} for frozen_row, adapted_row in zip(frozen_test['rows'], adapted_test['rows'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['map50'] >= frozen_test['map50']\n"
                "assert adapted_test['map50'] > baseline_grid['map50']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['map50'] > max(baseline_empty['map50'], baseline_grid['map50'])}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the boxes, detect in the scene again, export the adapter and reload it\n\n"
                "Six held-out records are written as panels (`outputs/{stem}_examples/`: the photograph with the reference "
                "boxes, the frozen detections and the adapted detections side by side, coloured by phrase, the counts "
                "beneath) so the numbers can be checked by eye. The drawn scene from Section 5 is then detected in again by "
                "the adapted model — the heads that were tuned serve every phrase, so this is a small look at what the "
                "adaptation did *outside* its phrase vocabulary and its corpus. Treat that one drawing as qualitative "
                "evidence, not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the two heads, about 6.3 MB in float32 — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the score and IoU "
                "thresholds the epoch was selected at, the phrase vocabulary, the training configuration and the epoch "
                "history (OUT8). `Owlv2DetectionPipeline.from_artifact` re-verifies the base snapshot, checks the artifact "
                "manifest, its digest and its exact tensor set **before** deserialising, refuses any tensor outside the two "
                "heads, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical detections on eight test records (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "adapted_items =pipe.detect_batch([r['image'] for r in test_records[:6]], PROMPTS, threshold=THRESHOLD)\n"
                "for record, frozen_row, adapted_row, adapted_dets in zip(test_records[:6], frozen_test['rows'][:6], adapted_test['rows'][:6], adapted_items, strict=True):\n"
                "    panels = [draw_boxes(record['image'], record['boxes']), draw_boxes(record['image'], adapted_dets, key='label', dashed=True)]\n"
                "    sheet = Image.new('RGB', (sum(p.width for p in panels) + 8, panels[0].height + 56), (255, 255, 255))\n"
                "    x = 0\n"
                "    for panel in panels:\n"
                "        sheet.paste(panel, (x, 0))\n"
                "        x += panel.width + 8\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    marker.text((8, panels[0].height + 6), f\"reference {{frozen_row['n_reference']}} boxes (solid) | adapted {{adapted_row['n_predicted']}} boxes, {{adapted_row['matched']}} matched (dashed); frozen matched {{frozen_row['matched']}} of {{frozen_row['n_predicted']}}\", fill=(20, 20, 20), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panels': ['reference boxes', 'adapted detections'], 'palette': PALETTE}})\n\n"
                "adapted_scene, adapted_scene_seconds, adapted_scene_checks, adapted_scene_report = detect_scene(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'threshold': artifact_manifest['adapter']['threshold'], 'prompts': artifact_manifest['adapter']['prompts'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = Owlv2DetectionPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.detect_batch([r['image'] for r in test_records[:8]], PROMPTS, threshold=THRESHOLD)\n"
                "after = reloaded.detect_batch([r['image'] for r in test_records[:8]], PROMPTS, threshold=THRESHOLD)\n"
                "\n\ndef same(dets):\n"
                "    return [(d['label'], round(d['score'], 4), tuple(round(v, 1) for v in d['box'])) for d in dets]\n\n\n"
                "parity ={{'identical_detections': sum(bool(same(a) == same(b)) for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_detections'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'threshold': THRESHOLD,\n"
                "    'iou_threshold': IOU_THRESHOLD,\n"
                "    'prompts': PROMPTS,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'files': {{k: v[0] for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE, 'bytes': CORPUS_BYTES, 'rows': CORPUS_ROWS, 'classes': list(BCCD_CLASSES), 'class_phrases': CLASS_PHRASES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256, 'prompts': scene_prompts, 'drawn_boxes': scene_boxes}}, 'frozen': {{'summary': frozen_scene, 'seconds': frozen_scene_seconds, 'checks': frozen_scene_checks, 'report': frozen_scene_report}}, 'adapted': {{'summary': adapted_scene, 'seconds': adapted_scene_seconds, 'checks': adapted_scene_checks, 'report': adapted_scene_report}}, 'output_files': ['outputs/{stem}_scene_frozen.json', 'outputs/{stem}_scene_adapted.json', 'outputs/{stem}_scene_frozen.png', 'outputs/{stem}_scene_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "This candidate asks whether bounded fine-tuning of an open-vocabulary detector's class and box heads on 260 "
        "labelled blood-smear records improves held-out detection relative to two non-adapted baselines and the frozen "
        "model. The clean-runtime run must answer that question; this source-only build makes no numerical claim. Read the "
        "emitted per-phrase AP at one IoU threshold, precision and recall at one score threshold, and predicted-box counts "
        "together rather than in isolation. The exported adapter is accepted only after fresh-base reload parity passes.\n\n"
        "The test split is 64 records from one seeded draw of one 364-record sample, the validation split that picks the "
        "epoch is 40, and every rate is at the one score threshold `DETECTION_THRESHOLD` and the one IoU threshold "
        "`IOU_THRESHOLD` — not a benchmark, not a threshold sweep, not COCO-style AP averaged over IoU thresholds, not a "
        "measure of phrases the sample never asks. So a result here says the contract works on three blood-cell phrases, "
        "not that the adapted model handles other phrases, other image families or your boxes. The heads that were tuned "
        "serve every phrase: the drawn scene re-detected in Section 9 is one drawing of evidence about what the tuning did "
        "outside its vocabulary, not a measurement, and a deployment that detects other phrases must "
        "measure them after adapting. The towers were not adapted: what the image encoder cannot see stays undetected, "
        "**the scores remain an uncalibrated sigmoid**, and there is still no non-maximum suppression.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and grid-prior rates on *your* boxes, and the "
        "frozen model's box count, are the numbers to read before any adapted one. **Precision and recall together:** a "
        "gain in mAP that comes with a collapse of one of them is a moved threshold, and the threshold is yours to set on "
        "a validation split, not the test split. **Leakage:** keep every image in one split (the contract de-duplicates by "
        "decoded pixels) and split by source, session or slide when your images come from few sources.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled box set, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract for a drawn scene and a bounded "
        "fine-tuning of the heads with the upstream objective, evaluate against two non-adapted baselines and the frozen "
        "model on an image-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, detection quality on any other phrase vocabulary or "
        "image family, calibration of the sigmoid, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `EPOCHS` and watch the validation mAP pick "
        "the epoch; change `LEARNING_RATE` by a factor of ten in either direction and read the curve; set `THRESHOLD` to "
        "`0.05` or `0.3` before Section 6 and read how precision and recall trade against each other for both the frozen "
        "and the adapted model; change `SPLIT_SEED` and read how much 64 records move; or bring your own boxes through "
        "BYOD and read the two baselines before the adapted number.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/owlv2-base-patch16-ensemble/` and rerun Section 3. A `sha256` "
        "`ValueError` naming a parquet file in Section 4: a cached `weights/bccd/*.parquet` is incomplete — delete it and "
        "rerun Section 4.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code (Scenic, OWL-ViT project): https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit\n"
        "- Scaling Open-Vocabulary Object Detection (Minderer, Gritsenko, Houlsby, 2023): https://arxiv.org/abs/2306.09683\n"
        "- Simple Open-Vocabulary Object Detection with Vision Transformers (Minderer et al., 2022): https://arxiv.org/abs/2205.06230\n"
        "- BCCD (Public Domain): https://huggingface.co/datasets/keremberke/blood-cell-object-detection — the Roboflow Universe export of the Blood Cell Count and Detection dataset\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
