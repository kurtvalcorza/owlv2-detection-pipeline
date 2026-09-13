"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "owlv2_detection_pipeline",
    "repo_name": "owlv2-detection-pipeline",
    "stem": "owlv2_detection",
    "notebook_name": "owlv2_detection_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "Owlv2DetectionPipeline",
    "weights_key": "owlv2-base-patch16-ensemble",
    "runtime_imports": ["torch", "transformers"],
    "title": "OWLv2 base/16 ensemble — DIMER open-vocabulary object detection tutorial (standalone)",
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
    "capability": "zero-shot (open-vocabulary, text-prompted) object detection using the pinned `google/owlv2-base-patch16-ensemble` weights",
    "intro": (
        "At inference the image is padded to a square with grey on the bottom and right, resized to 960×960 and split into "
        "60×60 = 3,600 patches by a CLIP ViT-B/16 image tower; each of the caller's phrases is encoded by the CLIP text tower "
        "into one query embedding; every image patch proposes one box and is scored against every query by a **sigmoid** of "
        "the image–text logit; the box survives when its best query score reaches the threshold and is labelled with that "
        "query. The pipeline returns each surviving box in xyxy pixel coordinates of the input image, the phrase it matched, "
        "and its score. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing "
        "fitting happens in this notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the "
        "carried pipeline module adds snapshot verification, prompt and image validation with named ceilings, a fixed output "
        "contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic "
        "scene drawn in code; the IoU values reported for it are sanity evidence against the boxes you drew, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with known object boxes, validate the image and the "
        "prompts into an input manifest through the pipeline's own validation stage, run text-prompted detection through "
        "the public API with an explicit caller-owned threshold, read sigmoid scores and the absence of non-maximum "
        "suppression correctly, produce an evaluation report that is `sample-sanity` with `box_iou` only when reference "
        "boxes exist and `not-measurable` otherwise, exercise an optional BYOD path, and export machine-readable detections "
        "plus an annotated image and provenance."
    ),
    "exclusions": (
        "instance or semantic segmentation (see the sibling SAM 2 pipeline), tracking, OCR, captioning, closed-set "
        "detection with a fixed class list (see the sibling RT-DETR pipeline), image-guided (one-shot) detection, mAP or "
        "precision/recall evaluation (which needs a labelled box set), or any training. Prompts are free text limited to "
        "16 CLIP tokens each, so a long phrase is silently truncated; and there is no non-maximum suppression, so one "
        "object can surface as several overlapping boxes at a low threshold — the score, not the label, is your only signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate for one image: the repository's model card records 5.3 s to load and 2.9–3.4 s per `detect` on the 640×480 synthetic scene in the Windows venv (Intel Core Ultra 9 275HX) — the cost is the fixed 960×960 ViT-B/16 pass, not the image size. The pinned `torch==2.14.0` install and the 620 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; what a sigmoid score is and why it is not a probability.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code (grey background, a black rectangle, a red disc, a blue triangle) with three prompts naming them, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own comma-separated prompt phrases (1–16 distinct phrases, at most 48 characters each; the upstream convention is `a photo of a <thing>`). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: a deterministic 640×480 RGB scene is "
                "drawn in code — grey background, a black filled rectangle at `[80, 120, 280, 360]`, a red filled disc whose "
                "bounding box is `[380, 140, 560, 320]` and a blue filled triangle whose bounding box is `[200, 380, 360, 460]` — "
                "and the prompts name them (`a black rectangle`, `a red circle`, `a blue triangle`). This is the same scene and "
                "prompt set the repository's smoke run used. The drawn boxes are the reference for the `box_iou` sanity check "
                "later; they are not a labelled dataset, so nothing here is a precision/recall measurement. The image digest is "
                "printed for the record. BYOD is optional and disabled by default; when enabled, upload one image and set "
                "`BYOD_PROMPTS` to the phrases you want found — no reference boxes exist for it, so the evaluation report will be "
                "`not-measurable`.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant: a patch's box survives "
                "when the sigmoid of its best image–text logit reaches it. The package default (`DETECTION_THRESHOLD = 0.1`) "
                "follows the pinned README's usage example, not a calibration; it is exposed here as a form parameter and passed "
                "explicitly on every call. Nothing is validated in this cell — the next section hands the image, the prompts and "
                "the threshold to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming "
                "the sample kind, the image size and digest, the prompts, the threshold, and the drawn reference boxes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PROMPTS = 'a photo of a cat, a photo of a remote control'  # @param {{type:\"string\"}}\n"
                "threshold = 0.1  # @param {{type:\"number\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    prompts = [phrase.strip() for phrase in BYOD_PROMPTS.split(',') if phrase.strip()]\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable.\n"
                "    image = Image.new('RGB', (640, 480), (128, 128, 128))\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    drawn_boxes = {{'a black rectangle': [80.0, 120.0, 280.0, 360.0], 'a red circle': [380.0, 140.0, 560.0, 320.0], 'a blue triangle': [200.0, 380.0, 360.0, 460.0]}}\n"
                "    draw.rectangle(drawn_boxes['a black rectangle'], fill=(30, 30, 30))\n"
                "    draw.ellipse(drawn_boxes['a red circle'], fill=(220, 30, 30))\n"
                "    draw.polygon([(280, 380), (200, 460), (360, 460)], fill=(30, 60, 220))\n"
                "    prompts = list(drawn_boxes)\n"
                "    image_name = 'synthetic_scene_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'prompts': prompts, 'threshold': threshold, 'drawn_boxes': drawn_boxes}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, 1..`MAX_PROMPTS` distinct phrases of at most "
                "`MAX_PROMPT_CHARS` characters each, and a threshold in `[0, 1]` — canonicalises the phrases through the "
                "package's `format_prompts` (whitespace-collapsed, lower-cased, trailing period removed, one text query each) and "
                "returns an **input manifest** naming the schema and ceilings (including the 3,600-patch ceiling on detections and "
                "the 16-token CLIP limit per query), the input's observed mode and size, the queries actually sent to the "
                "tokenizer, the threshold, and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To "
                "show what rejection looks like, the cell also validates a deliberately over-long phrase and records the "
                "pipeline's own error message as a finding. Inside the pipeline the image is converted to RGB, padded to a square "
                "and resized to 960×960 by the processor; boxes are mapped back to input pixels, and nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_DETECTIONS': MAX_DETECTIONS, 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, prompts, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['x' * (MAX_PROMPT_CHARS + 1)])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Detect and read the scores correctly\n\n"
                "`detect` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` the matched query text — plus `queries`, the threshold "
                "used, `width`, `height` and the model identity. At most 3,600 boxes can ever be returned (one per image patch). "
                "Each `score` is a **sigmoid of the best image–text logit, not a calibrated probability**: it was never fitted "
                "to the frequency with which a box is correct, so 0.9 does not mean \"90 % likely\", and the scores of different "
                "queries for one patch are independent. The threshold you passed is the only decision rule; the pipeline ships "
                "0.1 as a default (the README example's value), not as a calibration, and the caller owns it per deployment — "
                "raise it when false boxes cost more than missed ones, lower it for recall. **There is no non-maximum "
                "suppression**: neighbouring patches can propose overlapping boxes for the same object, and at a low threshold "
                "the same object appears more than once. Inference is deterministic on a fixed device and dtype (no sampling, "
                "`torch.inference_mode`); CUDA kernel selection can move scores in the third or fourth decimal place and reorder "
                "near-ties. As recorded in the model card, the repository's CPU smoke on this same scene at the default threshold "
                "returned exactly three boxes — `a red circle` 0.917, `a blue triangle` 0.810, `a black rectangle` 0.426 — within "
                "about 4 px of the drawn ones, and five boxes at threshold 0.05; that is one observation, not a calibration point."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.detect(image, prompts, threshold=threshold)\n"
                "elapsed = time.time() - t0\n"
                "print({{'n_detections': len(result['detections']), 'queries': result['queries'], 'threshold': result['threshold'], 'device': pipe.device, 'seconds': round(elapsed, 2)}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>2}}. score {{det['score']:.4f}}  label {{det['label']!r}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No detection metric "
                "is reported by default: mean average precision needs a labelled box set with a matching vocabulary, and this "
                "repository ships none. The repository's only metric helper is `box_iou(a, b)` (intersection-over-union of two "
                "xyxy boxes), the building block a caller would use to compute mAP on their own labelled data; when reference "
                "boxes are supplied the report carries one `box_iou` entry per reference — its value, which detection matched it "
                "best and whether that detection's label agrees — with the verdict `sample-sanity`. On the synthetic path those "
                "references are shapes **you drew yourself**, so a high IoU proves only that the input contract, query "
                "formatting, forward pass and coordinate mapping (including the square padding) round-trip. On BYOD no reference "
                "exists, the verdict is `not-measurable`, and the report states what would make the task measurable: labelled "
                "boxes on your own images with a phrase vocabulary matching the prompts. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered detections with boxes and labels, the queries, "
                "the threshold), the evaluation report, the input manifest, the sample identity, digest and drawn boxes, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable "
                "model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The "
                "detections are also written as CSV with explicit `image`, `rank`, `label`, `score`, `x0`, `y0`, `x1`, `y1` "
                "columns so score ordering survives downstream use, and an annotated PNG draws every returned box for visual "
                "inspection (a supplement to, not a replacement for, the machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=(0, 255, 0), width=2)\n"
                "    draw.text((det['box'][0] + 2, det['box'][1] + 2), f\"{{det['label']}} {{det['score']:.2f}}\", fill=(0, 255, 0))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'prompts': prompts, 'drawn_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes are matched to free-text phrases: the label tells you which query the box scored best against, not that the "
        "object is really there, and the sigmoid score is uncalibrated. The threshold is a request parameter you own; the "
        "default is the README usage example, not a tuned operating point. On the synthetic scene the `box_iou` values in the "
        "evaluation report compare detections to shapes you drew yourself and the verdict is `sample-sanity`, which proves only "
        "that the input contract, query formatting, forward pass and coordinate mapping work; they say nothing about "
        "photographs, small or occluded objects, crowded scenes, or vocabulary the model has never seen, and a BYOD result is a "
        "single-image observation with the verdict `not-measurable`. Prompts longer than 16 CLIP tokens are truncated silently, "
        "the image is padded to a square so a wide or tall image is encoded at a lower effective resolution, and without "
        "non-maximum suppression one object can surface as several boxes at a low threshold — the smoke run's icon scene showed "
        "duplicate `clock` and `stop sign` boxes at 0.11 and 0.106 alongside the real ones. Empty input is handled better than a "
        "closed-set detector's: a blank 4096×4096 image returned no boxes at the default threshold. The pipeline provides no "
        "segmentation, tracking, OCR, captioning, mAP evaluation, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** lower `threshold` to 0.05 and count the duplicate boxes (the smoke run found five for three "
        "shapes); rephrase the prompts in the upstream form (`a photo of a red circle`) and compare scores; add a prompt for "
        "something that is not in the scene (`a green star`) and see whether anything surfaces; enable `USE_BYOD` with a "
        "photograph, hand-label a few objects and pass them to `evaluation_report` to see the verdict switch to `sample-sanity` — "
        "the first step towards a real precision/recall number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code (Scenic, OWL-ViT project): https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit\n"
        "- Scaling Open-Vocabulary Object Detection (Minderer, Gritsenko, Houlsby, 2023): https://arxiv.org/abs/2306.09683\n"
        "- Simple Open-Vocabulary Object Detection with Vision Transformers (Minderer et al., 2022): https://arxiv.org/abs/2205.06230"
    ),
}
