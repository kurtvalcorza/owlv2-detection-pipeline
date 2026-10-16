"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

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
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "Owlv2DetectionPipeline",
    "weights_key": "owlv2-base-patch16-ensemble",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL", "pyarrow", "scipy"],
    "title": "OWLv2 base/16 ensemble — DIMER E2E open-vocabulary detection adaptation tutorial (standalone)",
    # NOTEBOOK_SPEC 2.5 KND1 and DOC17-DOC21 (review OWL-m3, 2026-10-16): the kind and the canonical use notice,
    # with the DOC19 clause for the medical-imaging-adjacent domain (blood-smear microscopy).
    "notebook_kind": "TUTORIAL",
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use.\n"
        ">\n"
        "> **Not a medical device.** The blood-smear photographs are teaching data. Neither the model nor an\n"
        "> adapter trained here is a medical device, and neither is for diagnosis, screening or any clinical decision."
    ),
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
        "Selecting **Run all** in a fresh supported runtime builds an isolated hash-locked environment from the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the "
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
        "dialog and no configuration edit (NOTEBOOK_SPEC 2.5 §5). A CUDA runtime is used automatically when present. "
        "Hosted clean-runtime runs are recorded, by notebook blob, in `docs/release-verification.md`; a revision of this "
        "notebook counts as verified only once a run of that exact blob is listed there."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` and `BYOD_PATH` (a zip or a folder already in the runtime, such as an attached Kaggle dataset; on Colab an empty path opens an upload dialog) in Section 4 and re-run from that cell to supply one zip "
        "or folder of images plus a `boxes.csv` (`file`, `prompt`, `x0`, `y0`, `x1`, `y1`, optional `id`; one row per object, at least "
        "12 distinct images — at the default 15 % validation / 20 % test fractions fewer leave the training split under eight records — and at most 16 distinct phrases). The records pass through the same validation, image-disjoint split, "
        "baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the BCCD sample. Uploaded "
        "files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "OWLv2 [1], [2] is a CLIP ViT-B/16 image tower and a CLIP text tower joined by two small heads: the image is padded to a "
        "square, resized to 960 × 960 and split into 60 × 60 = 3,600 patches, each of which the **box head** (a three-layer "
        "MLP with a per-patch position bias) turns into one box and the **class head** (a projection plus a learned logit "
        "shift and scale) scores against every phrase embedding; a patch's box survives when the **sigmoid** of its best "
        "image–text logit reaches a caller-owned threshold, and is labelled with that phrase (154,966,792 parameters in all, "
        "published under the **Apache-2.0** licence). The score is **an uncalibrated sigmoid**, there is **no non-maximum "
        "suppression**, and **the threshold is a caller-owned request parameter**.\n\n"
        "What this notebook adds to inference is **adaptation of the class and box heads on labelled (image, phrases, "
        "boxes) records**. The records are blood-smear microscopy photographs from BCCD, each with every platelet, red "
        "blood cell and white blood cell boxed by hand [5]. BCCD is not among the detection datasets the model card names (COCO and OpenImages) [3], "
        "but the CLIP backbone was trained on image-caption data crawled from the web [3] and OWLv2 was self-trained on "
        "web image-text pairs with machine-generated boxes (over 1B examples) [1], and these blood-smear photographs have "
        "been public since 2017; overlap between that pretraining data and BCCD has not been audited and cannot be ruled "
        "out, and the three phrases are ordinary English. So the notebook measures the frozen model first and treats its "
        "score as the starting point, not as proof that the domain is new to the detector. The question is "
        "narrow and honest: does a bounded fine-tuning of "
        "the 1,579,526-parameter heads on 260 records — the towers frozen, a DETR-style matched objective — move the "
        "held-out **mAP** (the mean of the per-phrase AP50 over detections retained at score ≥ 0.1), **per-phrase AP**, **precision** and **recall** on an image-disjoint "
        "test split past the frozen model and two **non-adapted baselines**, and what does it do to the drawn shapes the "
        "same heads detect? Nothing here is a claim about your images or your phrases: it is one seeded split of one "
        "small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 9-file manifest with the tokenizer and "
        "processor files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files "
        "before the processor or the model is constructed. Model and adapter weights are held in **float32 on every device**: "
        "the adapter is trained in float32 and overlays without a cast. The one exception is the training-time "
        "frozen-tower cache (Section 7), which stores image features in float16 to halve host memory. CPUs and different "
        "GPUs do not run bit-identical arithmetic. In the reported runs, precision and kept-box counts at the score "
        "threshold differed between devices and numeric paths more than AP50 did (Section 8 gives the numbers); this "
        "notebook does not promise the same counts on any other device."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met bounding boxes, and wants to see how an open-vocabulary detector finds objects from free-text phrases, how to score detections honestly against baselines, and how to adapt its heads to a new image family without fooling themselves. The audience is students and practitioners preparing their own labelled boxes; no prior experience with OWLv2, CLIP or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A T4 GPU runtime is strongly recommended (the image tower runs at 960 × 960); CPU works but is slow.\n\n**Input → Model → Output.**\n\n| | Open-vocabulary detection | Bounded head fine-tuning |\n|---|---|---|\n| Input | one image and 1–16 free-text phrases | labelled records `{{id, image, boxes}}`: 364 public-domain BCCD blood-smear photographs with 4,886 cell boxes (260 train, 40 validation, 64 test) or your own zip |\n| Model | OWLv2: CLIP image and text towers, a box head and a class head over 3,600 patches; a box survives when its sigmoid score reaches the threshold (no non-maximum suppression) | the same model with only the class and box heads trained (1,579,526 parameters) on cached frozen-tower features, Hungarian-matched focal + L1 + GIoU loss |\n| Output | score-ordered boxes, each labelled with the phrase it matched | held-out mAP@0.5, per-phrase AP, precision and recall beside an empty and a grid-prior baseline, and a 6.3 MB safetensors adapter that reloads with identical detections |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults are the recorded run's settings. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from **the recorded run**: the Colab Tesla T4 run of 8 October 2026 of notebook blob `a545a73b` (commit `cbb9f81`), listed in `docs/release-verification.md` [6]. That run used the same default-path computation as this revision; on a CPU or another GPU, expect precision and box counts to differ (Section 8). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 BCCD records, phrases and an image-disjoint split *(evaluation practice)* → 5 the inference contract on drawn shapes *(core concept: sigmoid scores, caller-owned threshold)* → 6 two baselines and the frozen model *(evaluation practice)* → 7 bounded fine-tuning of the heads *(core concept: what is trained)* → 8 held-out evaluation → 9 the boxes by eye, the drawn scene again, export and reload *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled box set, turn it into (image, phrases, boxes) records, validate "
        "it and split it by image without exact decoded-image overlap; compute the intersection-over-union of two boxes "
        "(for example `[0, 0, 10, 10]` and `[5, 0, 15, 10]` share 50 px² of a 150 px² union, IoU = 1/3, below the 0.5 "
        "needed to match) and read AP50 as the area under one phrase's precision–recall curve; detect in a drawn scene through the public API and read the output "
        "contract correctly (an uncalibrated sigmoid, a caller-owned threshold, no non-maximum suppression, a "
        "`sample-sanity` report only against boxes you drew yourself); measure the frozen model's held-out per-phrase AP, "
        "mAP, precision and recall beside two non-adapted baselines; run a bounded fine-tuning of the heads with a "
        "Hungarian-matched focal + L1 + GIoU objective, explicit hyperparameters and validation-based epoch selection; evaluate "
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
        '- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with OWLv2, CLIP or fine-tuning. Sigmoid scores, thresholds, IoU, AP and mAP, baselines, matched losses and adapters are explained where they are first used and again in the Glossary.',
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter; CPU or CUDA). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path uses CUDA automatically when present. The image tower runs `EVAL_BATCH_SIZE` records per forward at 960 × 960. **Timing (measured):** the recorded run (Colab Tesla T4, 8 October 2026, blob `a545a73b`) took 598.8 s end to end in one pass, including the one-time build of the isolated environment. **CPU** is supported but slow: about 500 image-tower passes at 960 × 960 dominate, so expect roughly two hours on a two-core hosted CPU runtime (an estimate from a local two-thread measurement of about 13 s per pass, not a recorded run); use a GPU runtime. The pinned `torch==2.14.0` install and the 620 MB checkpoint are the large downloads; the parquet files are 4.8 MB. The feature cache holds about 1.7 GB of half-precision tensors on the host for 300 records (5.5 MB each).",
        "- **Knowledge:** basic Python, NumPy and PIL; what a bounding box in xyxy pixel coordinates is. Intersection-over-union, AP50 and a precision/recall pair are taught here (see the learning objectives), as is why 64 records from one draw give no dispersion and why a self-drawn scene is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, boxes}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4,096 px and `boxes` 1..200 entries `{prompt, box}`: a phrase of at most 48 characters (normalised like a query; at most 16 distinct phrases per dataset) and an `[x0, y0, x1, y1]` pixel box inside the image at least 1 px wide and tall. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits, and `split_dataset` needs at least 12 distinct images at its default fractions (training at least 8, validation and test at least 1 each; `min_split_images()` gives the number for other fractions). BYOD accepts one zip (or folder) of images plus a `boxes.csv` in the layout named above.",
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
                '\n\n**Predict before running:** why split by image rather than by box? About how many boxes of each cell type will a typical image hold?'
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                '\n'
                'class BYODRefusal(ValueError):\n'
                '    """A refused BYOD input: shown as this one message, without a traceback (it names the file and the rule)."""\n'
                '\n'
                '    dimer_refusal = True  # the isolated-environment router prints the message alone\n'
                '\n'
                '    def _render_traceback_(self):  # the same when the cell runs in the kernel itself\n'
                '        return [str(self)]\n'
                '\n'
                '\n'
                'def refuse_byod(reason):\n'
                '    """Forget the previous data so no later cell reuses it, and return one actionable refusal (DAT20)."""\n'
                "    for stale in ('splits', 'dataset_manifests', 'train_records', 'val_records', 'test_records', 'PROMPTS'):\n"
                '        globals().pop(stale, None)\n'
                "    return BYODRefusal(f'BYOD input refused: {{reason}} Fix the upload or BYOD_PATH and run this cell again.')\n"
                '\n'
                '\n'
                "if USE_BYOD:\n"
                "    # A refused input (missing path, wrong file, bad boxes.csv, too few images) prints one message;\n"
                "    # any other error keeps its traceback.\n"
                "    try:\n"
                "        byod_dir = Path(str(BYOD_PATH).strip()).expanduser() if str(BYOD_PATH).strip() else None\n"
                "        # A folder (e.g. a Kaggle dataset under /kaggle/input) is read in place; otherwise one zip.\n"
                "        byod_zip = byod_dir if byod_dir is not None and byod_dir.is_dir() else byod_file(BYOD_PATH, 'zip of images plus boxes.csv (or set BYOD_PATH to a folder holding them)', ('.zip',))\n"
                "    except (FileNotFoundError, RuntimeError, ValueError) as exc:\n"
                "        raise refuse_byod(exc) from None\n"
                "    try:\n"
                "        records = load_byod_dataset(byod_zip)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    except ValueError as exc:\n"
                "        raise refuse_byod(f'{{byod_zip.name}}: {{exc}}.') from None\n"                "    data_source = 'BYOD (' + byod_zip.name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/bccd')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'files': sorted(corpus_files), 'bytes': sum(len(v) for v in corpus_files.values()), 'records': len(corpus), 'boxes': sum(len(r['boxes']) for r in corpus), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{}}\n"
                "for name, part in splits.items():\n"
                "    # adapt() needs MIN_RECORDS training records; validation and test need one each (split_dataset checks the same).\n"
                "    try:\n"
                "        dataset_manifests[name] = validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1)\n"
                "    except ValueError as exc:\n"
                "        message = f'{{name}} split: {{exc}} (a BYOD set needs at least {{min_split_images()}} distinct images at the default split fractions)'\n"
                "        if USE_BYOD:\n"
                "            raise refuse_byod(message) from None\n"
                "        raise ValueError(message) from None\n"
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
                '<details><summary>Check your reasoning</summary>Boxes in one photograph share its staining, focus and crowding, so a box-level split would put near-copies of a test cell in training. The recorded run split 260 / 40 / 64 images without exact decoded-image overlap (by decoded-pixel digest); whether images from one slide, patient or source, or near-duplicates, sit in two splits is not established for BCCD; a typical image holds about eleven red blood cells, one white blood cell and one platelet, so the test split has 846 reference boxes, 731 of them red blood cells. All four refusal probes were rejected before the model ran.</details>'
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
                "`metrics.py`): per phrase, the **AP50 over detections retained at score ≥ 0.1** — only the detections of that phrase that pass `THRESHOLD`, "
                "ranked by score across the whole split; with conventional VOC matching, each detection is compared with the "
                "reference box of its phrase it overlaps most and is a true positive only when that IoU is at least 0.5 and "
                "the box is still free (a second detection of the same box is a false positive); the area under that "
                "truncated precision–recall step curve (VOC all-points interpolation) — and their mean, printed as "
                "**mAP@0.5**, the measure the epoch is selected on (compared unrounded, printed to 4 decimal places). Boxes "
                "below the threshold never enter that curve, so the cells also print an **all-candidates AP50** over every "
                "patch box as a display-only diagnostic; it is never used for selection; and, over all phrases pooled, the **precision** and **recall** of the detections the score "
                "threshold let through and their F1. The **empty** baseline predicts no box and scores 0 by construction — "
                "the floor. The **grid-prior** baseline tiles each phrase's median training box edge to edge over every "
                "image with one constant score, what the label statistics buy without looking at the image; with every score "
                "tied, its boxes rank in tiling order (phrase by phrase, columns left to right, top to bottom within a "
                "column), and its AP depends on that order. The **frozen "
                "model** is scored by `pipe.evaluate`, which detects the three phrases in every record in batches of "
                "`EVAL_BATCH_SIZE`, keeps the boxes whose sigmoid reaches `THRESHOLD`, and scores them. Do not assume the "
                "frozen detector's behaviour in advance: read its mAP, per-phrase APs, precision, recall, box counts and "
                "per-record rows from this run."
                '\n\n**Predict before running:** the model card names no blood-smear set among its detection data, and the three phrases are ordinary English. Will its frozen mAP@0.5 be closer to the empty baseline or to 1? Which phrase do you expect it to find best?'
            ),
            "code": (
                '# A re-run after Section 7 (an experiment or BYOD): put the frozen heads captured at load back first, so the\n'
                '# "frozen" scores below really are the frozen model (adapt() itself always starts from the frozen heads too).\n'
                'if pipe.adapter is not None:\n'
                "    print({{'reset_to_frozen_heads': pipe.reset_to_base()}})\n"
                '\n'
                "METRICS = ('map50', 'precision', 'recall', 'f1')\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_grid = grid_baseline(test_records, train_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'grid_baseline': {{k: round(baseline_grid[k], 3) for k in METRICS}}, 'prior_box_sizes': baseline_grid['prior_box_sizes'], 'n_predicted_boxes': baseline_grid['n_predicted_boxes'], 'note': baseline_grid['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE, all_candidates=True)\n"
                "if frozen_test['adapted']:\n"
                "    raise RuntimeError('the pipeline still carries an adapter, so this would not be the frozen model: run Section 6 again')\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'adapted': frozen_test['adapted'], 'n': frozen_test['n'], 'reference_boxes': frozen_test['n_reference_boxes'], 'predicted_boxes': frozen_test['n_predicted_boxes'], 'matched': frozen_test['n_matched'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_per_prompt': frozen_test['per_prompt']}})\n"
                "print({{'frozen_all_candidates_ap50 (display only, not used for selection)': frozen_test['all_candidates']}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for row in frozen_test['rows'][:4]:\n"
                "    print(row)"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Close to the floor. In the recorded run the empty baseline scored mAP@0.5 0.0, the grid prior 0.0147 and the frozen model 0.0723, with precision 0.1295 and recall 0.0591: it kept 386 boxes for 846 references and matched 50. Per phrase, only `a white blood cell` (the largest, most distinctive cell) had a non-zero AP, 0.2168; platelets and red blood cells scored 0.0. These are AP50 over detections retained at score ≥ 0.1; the recorded run predates the all-candidates diagnostic, which in a local CPU measurement (16 October 2026, not a hosted run) was 0.0845 for the frozen model against 0.0723 at the cut-off — still near the floor.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the heads\n\n"
                "`pipe.adapt` trains only the two detection heads — the class head (a 768 → 512 projection with a learned "
                "logit shift and scale) and the box head (a three-layer MLP): 1,579,526 of 154,966,792 parameters — while "
                "the CLIP image and text towers, the post-merge layer norm and the objectness head stay frozen, the split the "
                "tutorial keeps fixed. The loss is a **DETR-style matched objective** of the kind OWL-ViT's heads were "
                "trained with (this notebook does not verify it against the upstream training code): each reference box is assigned to one of the 3,600 patch candidates by **Hungarian matching** under a "
                "cost of focal classification, L1 and generalised-IoU terms, then the **sigmoid focal loss** is taken over "
                "every (patch, phrase) logit with the matched pairs as positives, and **L1 and GIoU** over the matched boxes, "
                "all normalised by the number of reference boxes. Because the towers are frozen, their outputs — the "
                "60 × 60 × 768 feature map per image and the three phrase embeddings — are computed once under no gradient "
                "and cached in half precision (float16) on the host (the **frozen-tower cache**), and each step runs only the "
                "heads on those cached features: the heads train on float16-quantized frozen features (promoted to the "
                "head dtype), which approximates training on freshly computed float32 features. With the frozen heads held "
                "fixed, quantizing the features moved the logits by at most 0.047 (a fixed-weight comparison, local CPU "
                "measurement on this split, 16 October 2026); the training comparison is in Section 8. AdamW without weight decay at a fixed learning "
                "rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen "
                "model's validation rates; every epoch is scored on the 40 validation records at `THRESHOLD`, and the epoch "
                "with the **highest validation mAP** (the earliest on ties) is kept.\n\n"
                "Read the emitted epoch history rather than assuming improvement: it records the validation mAP and loss "
                "for every epoch, then restores the earliest epoch with the highest validation mAP. `adapt` always starts from the "
                "frozen heads captured when the snapshot was loaded — a re-run of this cell trains from scratch rather than continuing "
                "an earlier adaptation — so epoch 0 really is the frozen model (Section 6 resets `pipe` the same way before scoring it)."
                '\n\n**Predict before running:** only 1.6 M of 155 M parameters are trained. Will the validation mAP keep rising for all eight epochs, and which epoch will be kept?'
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
                '# A re-run after Section 7 (an experiment or BYOD): put the frozen heads captured at load back first, so the\n'
                '# "frozen" scores below really are the frozen model (adapt() itself always starts from the frozen heads too).\n'
                'if pipe.adapter is not None:\n'
                "    print({{'reset_to_frozen_heads': pipe.reset_to_base()}})\n"
                '\n'
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, prompts=PROMPTS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, threshold=THRESHOLD, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'threshold': adapt_result['threshold'], 'iou_threshold': adapt_result['iou_threshold'], 'prompts': adapt_result['prompts'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded run the best validation mAP@0.5 was 0.8734 at **epoch 8**, the last one, so selection kept the final epoch this time; with more epochs or a higher learning rate an earlier epoch can win. Training only the heads is enough because the frozen towers already describe the image; the heads learn which patches are cells and how to box them.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test records were never used for training or epoch selection, and no decoded image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **mAP@0.5** first (the measure the epoch was selected on), then the "
                "**per-phrase APs**, then **precision** and **recall** together — one precision/recall pair is one operating point, and the same "
                "numeric threshold need not be a comparable operating point before and after training, because training "
                "moves the scores — then the number of predicted boxes against the reference count, and last the "
                "display-only all-candidates AP50. The cell reports (in `verdicts`, without stopping a BYOD run before export) whether the adapted mAP "
                "is at least the frozen one and outperforms this training-derived grid prior. Sixty-four records from one seeded split give **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a result on one "
                "blood-smear set's three cell types says nothing about other phrases, other images or your data until you "
                "measure them."
                "\n\n**Predict before running:** will the adapted model's precision and recall rise together? Will it predict more or fewer boxes than there are reference boxes?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE, all_candidates=True)\n"
                "adapted_val = pipe.evaluate(val_records, prompts=PROMPTS, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'grid': round(baseline_grid[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['per_prompt_ap'] = {{p: {{'frozen': frozen_test['per_prompt'][p]['ap'], 'adapted': adapted_test['per_prompt'][p]['ap'], 'n_reference': adapted_test['per_prompt'][p]['n_reference']}} for p in PROMPTS}}\n"
                "comparison['all_candidates_ap50_display_only'] = {{'frozen': frozen_test['all_candidates']['map50'], 'adapted': adapted_test['all_candidates']['map50']}}\n"
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
                "    'verdicts': {{'adapted_map50_at_least_frozen': adapted_test['map50'] >= frozen_test['map50'], 'adapted_above_grid_prior': adapted_test['map50'] > baseline_grid['map50']}},\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "if not all(evaluation_report_payload['verdicts'].values()):\n"
                "    print('The adapted model did not reach the frozen mAP or did not beat the grid prior; the report, export and reload still run. Read the baselines and the epoch history before trusting it.')\n"

                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['map50'] > max(baseline_empty['map50'], baseline_grid['map50'])}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Both rose, but precision remained low. In the recorded run mAP@0.5 rose from 0.0723 to 0.8918, recall from 0.0591 to 0.9894, and precision from 0.1295 to only 0.2236: the adapted model kept 3,744 boxes for 846 references (837 matched), so 2,907 kept boxes matched no reference. Duplicates of cells already found are a plausible contributor — there is no non-maximum suppression and the threshold stayed at 0.1 — but this run does not show what share they are. What low-scoring false positives cost AP depends on where they rank: below the last true positive they cost nothing, interleaved with true positives they lower the curve; precision at one threshold counts them all. Per phrase: platelet 0.0 → 0.8805, red blood cell 0.0 → 0.8476, white blood cell 0.2168 → 0.9473. The verdict line records the comparison instead of stopping the notebook. Your counts may differ: in the reported runs, precision and kept-box counts at the 0.1 threshold moved with the device and numeric path more than AP50 did. Local CPU measurements of the same code (16 October 2026, not hosted runs): trained on the float16 cache, the adapted model kept 2,974 boxes at precision 0.279 with test AP50 0.8914; trained on a float32 cache instead, precision 0.333 and test AP50 0.8885; both kept epoch 8. Those runs and the recorded run scored with the earlier greedy best-unmatched matcher; on CPU, switching to the conventional VOC matching used now moved the adapted test AP50 from 0.8914 to 0.8909 and left the frozen 0.0723 unchanged. The all-candidates diagnostic for the adapted model was 0.8935 on CPU.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the boxes, detect in the scene again, export the adapter and reload it\n\n"
                "Six held-out records are written as two-panel sheets (`outputs/{stem}_examples/`: the photograph with the "
                "reference boxes, solid, beside the same photograph with the adapted detections, dashed, coloured by phrase; "
                "beneath them the reference count, the adapted boxes and matches, and the frozen model's matches for comparison) so the numbers can be checked by eye. The drawn scene from Section 5 is then detected in again by "
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
                "model (VER2). The cell asserts **detection parity at reported precision** on eight test records (VER4): the "
                "same labels, scores equal to 4 decimal places and box coordinates to 1 decimal place."
                '\n\n**Predict before running:** the reloaded pipeline is rebuilt from the adapter file and the pinned base. Will its detections on eight test records match at that precision?'
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
                "parity ={{'compared_at': 'label, score to 4 dp, box to 1 dp', 'identical_detections': sum(bool(same(a) == same(b)) for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
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
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Yes: the recorded run reported parity on 8 of 8 records at that precision (scores to 4 decimal places, coordinates to 1), and the adapter was 6,319,224 bytes. The drawn scene after adaptation is qualitative only — the heads were tuned on cells, so expect different scores on shapes they were never shown.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "This notebook asks whether bounded fine-tuning of an open-vocabulary detector's class and box heads on 260 "
        "labelled blood-smear records improves held-out detection relative to two non-adapted baselines and the frozen "
        "model. Your run's Section 8 answers that question for this one split; the worked answers quote the recorded run. Read the "
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
        "**the scores remain an uncalibrated sigmoid**, and there is still no non-maximum suppression. What was "
        "measured is **supervised adaptation to three fixed phrases**, not retained open-vocabulary ability, and AP50 "
        "is a box-matching score: it is not a cell count, a differential count or evidence of clinical validity.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and grid-prior rates on *your* boxes, and the "
        "frozen model's box count, are the numbers to read before any adapted one. **Precision and recall together:** "
        "one pair is one operating point, and the same numeric threshold need not be comparable before and after "
        "training; choose the operating point on a validation split, not the test split. **Leakage:** the contract only "
        "guarantees splits without exact decoded-image overlap (it de-duplicates by decoded pixels); slide, patient and "
        "source independence and near-duplicates are not established for BCCD, so split by source, session or slide "
        "when you know them.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled box set, validate "
        "the demonstrated dataset contract with splits free of exact decoded-image overlap, execute the inference contract for a drawn scene and a bounded "
        "fine-tuning of the heads with a DETR-style matched objective, evaluate against two non-adapted baselines and the frozen "
        "model on an image-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, detection quality on any other phrase vocabulary or "
        "image family, calibration of the sigmoid, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `EPOCHS` and watch the validation mAP pick "
        "the epoch; change `LEARNING_RATE` by a factor of ten in either direction and read the curve (for both, edit the "
        "field in Section 7 and run Sections 6–9 again); change `THRESHOLD` (it is set in the Section 5 cell) to `0.05` or "
        "`0.3` and run Sections 5–9 again to read how precision and recall trade against each other for both the frozen and "
        "the adapted model; change `SPLIT_SEED` in Section 4 and run Sections 4–9 again to read how much 64 records move; "
        "or bring your own boxes through BYOD (Sections 4–9) and read the two baselines before the adapted number. Compare "
        "settings on the validation numbers (`adapted_val` and the epoch history): the test split was scored once on the "
        "default path, and every further look at it is exploratory, not a new held-out result. If you try several "
        "`SPLIT_SEED` values, report them collectively (all of them, or their range), not the best one. Each "
        "re-run starts from the frozen heads: Section 6 resets `pipe` and prints `reset_to_frozen_heads`, and the frozen "
        "row it prints carries `'adapted': False`. To return to the defaults, restore the field and run the same range again.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory (the frozen-tower cache and two pipelines in Section 9); restart the session and choose **Run all** on a GPU runtime.\n'
        '- **`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3** — a staged file is incomplete or altered; delete it from `weights/owlv2-base-patch16-ensemble/` and run Section 3 again.\n'
        '- **A `sha256` `ValueError` naming a parquet file in Section 4** — a cached `weights/bccd/*.parquet` is incomplete; delete it and run Section 4 again.\n'
        '- **CUDA out of memory in Section 6 or 7** — lower `EVAL_BATCH_SIZE` or `BATCH_SIZE` and run Sections 6–9 again (numbers will differ slightly from the recorded run).\n'
        '- **Section 8 says the adapted model did not beat the frozen model or the grid prior** — on the default path that is a finding worth recording; on your own data read the baselines and the epoch history first.\n'
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to the zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **BYOD: a `load_byod_dataset`, `split_dataset` or `validate_dataset` refusal** — it names the `boxes.csv` row, the file or the split and the rule (box outside the image, too many phrases, too few images: at least 12 distinct images at the default split).\n'
        '- **The session crashed or the worker died during Section 7 on CPU or a small GPU** — the frozen-tower cache holds about 5.5 MB per record on the host; lower the number of records (BYOD) or use a runtime with more memory.\n\n'
        '## Glossary\n\n'
        '- **Open-vocabulary detection** — detecting objects named by free-text phrases rather than a fixed class list.\n'
        '- **Patch / box head / class head** — the image is cut into 3,600 patches; the box head predicts one box per patch and the class head scores it against each phrase.\n'
        '- **Sigmoid score / threshold** — each (patch, phrase) logit squashed into 0..1 on its own; boxes whose score reaches the caller-owned `THRESHOLD` are kept. Not a calibrated probability.\n'
        '- **Non-maximum suppression** — suppressing lower-scoring boxes that overlap a higher-scoring box (usually of the same label); OWLv2 here applies none, so duplicates are possible.\n'
        '- **IoU** — intersection over union of two boxes; a detection matches a reference box at IoU ≥ 0.5.\n'
        "- **AP50 / mAP@0.5** — the area under one phrase's precision–recall curve at IoU ≥ 0.5, here over the detections retained at score ≥ 0.1 (VOC matching); their mean over phrases. The all-candidates AP50 uses every patch box and is display-only.\n"
        '- **Precision / recall / F1** — share of kept boxes that match; share of reference boxes found; their harmonic mean.\n'
        "- **Empty / grid-prior baseline** — predicting no box; tiling each phrase's median training box over the image.\n"
        '- **Hungarian matching / focal, L1, GIoU loss** — the one-to-one assignment of reference boxes to patches, and the classification and box terms trained on it.\n'
        '- **Frozen-tower cache** — the frozen image and text tower outputs computed once, so training runs only the heads.\n'
        '- **Epoch / validation selection** — one pass over the training records; keeping the epoch with the highest unrounded validation mAP.\n'
        '- **Adapter / reload parity** — the two trained heads only, overlaid on the pinned base; the reloaded model gives the same detections at reported precision (scores to 4 dp, boxes to 1 dp).\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your images and boxes through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- Empty ___, grid prior ___, frozen ___, adapted ___ (test mAP@0.5); the verdict was ___.\n'
        '- Precision moved from ___ to ___ and recall from ___ to ___; the adapted model predicted ___ boxes against ___ references, which tells me ___.\n'
        '- The phrase that changed most was ___, because ___.\n'
        '- One reason not to trust this gain on my own images yet: ___.\n\n'
        "The upstream code is the Scenic OWL-ViT project [4]. The package, its contract and the weight provenance are "
        "documented in the repository README [7], model card [8] and weight notes [9]; the notebook follows the DIMER "
        "specifications [10].\n\n"
        "## References\n\n"
        "[1] M. Minderer, A. Gritsenko, and N. Houlsby, \u201cScaling open-vocabulary object detection,\u201d in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023. [Online]. Available: https://arxiv.org/abs/2306.09683\n\n"
        "[2] M. Minderer *et al.*, \u201cSimple open-vocabulary object detection with vision transformers,\u201d in *Proc. European Conf. Computer Vision (ECCV)*, 2022. [Online]. Available: https://arxiv.org/abs/2205.06230\n\n"
        "[3] Google, \u201cowlv2-base-patch16-ensemble,\u201d Hugging Face model card, revision `{MODEL_REVISION}`. [Online]. Available: https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/README.md\n\n"
        "[4] Google Research, \u201cScenic: OWL-ViT project,\u201d GitHub repository. [Online]. Available: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit\n\n"
        "[5] keremberke, \u201cblood-cell-object-detection (BCCD, Roboflow Universe export, Public Domain),\u201d Hugging Face dataset. [Online]. Available: https://huggingface.co/datasets/keremberke/blood-cell-object-detection\n\n"
        "[6] K. Valcorza, \u201cowlv2-detection-pipeline: release verification,\u201d GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/docs/release-verification.md\n\n"
        "[7] K. Valcorza, \u201cowlv2-detection-pipeline: README,\u201d GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/README.md\n\n"
        "[8] K. Valcorza, \u201cowlv2-detection-pipeline: model card,\u201d GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/MODEL_CARD.md\n\n"
        "[9] K. Valcorza, \u201cowlv2-detection-pipeline: weight provenance,\u201d GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/docs/WEIGHTS.md\n\n"
        "[10] DIMER, \u201cDIMER Notebook Specification 2.5 and Model Card Specification 1.1,\u201d fleet specifications in the ml-worker repository."
    ),
}
