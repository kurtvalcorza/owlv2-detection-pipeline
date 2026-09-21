---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: zero-shot-object-detection
task: "Others - Open-Vocabulary Object Detection"
base_model: google/owlv2-base-patch16-ensemble
date_published: "2023-10-13"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2023-10-13T09:27:09Z, https://huggingface.co/api/models/google/owlv2-base-patch16-ensemble — the Transformers-format conversion); the OWLv2 paper is arXiv:2306.09683 (2023-06) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# OWLv2 base/16 ensemble — Open-Vocabulary Object Detection (Inference and Adaptation)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlv2--base--patch16--ensemble-ffcc4d?style=flat)](https://huggingface.co/google/owlv2-base-patch16-ensemble)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2306.09683-b31b1b.svg)](https://arxiv.org/abs/2306.09683)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a standalone end-to-end notebook that stages and verifies the pinned model and sample data, validates labelled boxes, measures baselines, fine-tunes the detection heads, evaluates a held-out split, exports an adapter, and verifies fresh-base reload:

- **E2E Fine-tuning Tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/owlv2-detection-pipeline/blob/main/tutorials/owlv2_detection_colab.ipynb) [`owlv2_detection_colab.ipynb`](https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/tutorials/owlv2_detection_colab.ipynb)  
  *Text-prompted detection plus bounded class-head and box-head fine-tuning on a digest-pinned BCCD sample, with image-disjoint splits, per-phrase AP/mAP/precision/recall, two non-adapted baselines, SafeTensors adapter export, and fresh-base reload parity. The current blob remains Candidate until its queued Kaggle Tesla T4 run passes.*

---

#### Description

`google/owlv2-base-patch16-ensemble` is the Transformers-format release of OWLv2 — "Scaling Open-Vocabulary Object Detection" (Minderer, Gritsenko and Houlsby, arXiv:2306.09683; Google Research) — the base/16 variant whose detection heads were trained with self-training on web-scale pseudo-labels and then ensembled by weight-space averaging with a fine-tuned checkpoint (the "ensemble" suffix), converted by the Hugging Face team and pinned here to revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `Owlv2ForObjectDetection` (`model_type` owlv2): a CLIP ViT-B/16 image tower over a 960×960 input (60×60 = 3,600 patch tokens, `projection_dim` 512) and a CLIP text tower (BPE tokenizer, `model_max_length` 16 per query); the float32 `model.safetensors` is 620 MB. At inference the processor pads the image to a square, resizes and normalises it, then every patch predicts one box and one image-text logit per phrase; the retained box carries the sigmoid of its best phrase logit, with no non-maximum suppression. This repository adds digest-verified staging and local-only loading, validated single- and batched-detection APIs, a labelled-box dataset contract, per-phrase AP and pooled precision/recall metrics, empty and grid-prior baselines, and bounded adaptation of the 1,579,526 class-head and box-head parameters under Hungarian-matched focal + L1 + GIoU loss while both CLIP towers remain frozen. `save_artifact` writes only those trained head tensors as SafeTensors plus a manifest bound to the base-weight digest; `from_artifact` verifies and reloads them into a fresh base.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is open-vocabulary object detection: input one image (`PIL.Image.Image`, any mode, converted to RGB), 1–16 free-text phrases and a threshold; output a list of at most 3,600 detections, each an xyxy pixel box, a `label` equal to the best-matching phrase, and the model's sigmoid `score`, sorted by score. Envisioned applications are detection of objects that closed-set detectors do not know — a product, a part, a sign, a species named in words — as a labelling assistant, a first pass for cropping or counting, a retrieval filter over image collections, and a comparison baseline against the sibling closed-set RT-DETR pipeline. The adaptation surface accepts validated labelled `(image, phrase, box)` records and fine-tunes only the class and box heads for a bounded vocabulary while retaining text-conditioned inference. The pipeline is an inference and bounded-adaptation component and a zero-configuration baseline for text-conditioned detection, not a certified detector for any specific vocabulary, camera or scene.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers, and data analysts integrating text-prompted detection into research prototypes or in-house tooling. A user is expected to understand that the label is the phrase a patch scored best against, not a verified identity — every patch proposes a box, and a phrase the image does not contain can still surface just above a low threshold — that the score is a sigmoid of an image–text logit, a ranking signal within one image that is not calibrated on their data and not exclusive across queries, that there is no non-maximum suppression so one object can appear as several boxes at a low threshold, that phrases are truncated at 16 CLIP tokens and work best in the upstream `a photo of a <thing>` form, that the image is padded to a square so a wide or tall image is encoded at a lower effective resolution, that the training data is web image–text pairs and detection datasets so drawn graphics, documents, medical and aerial imagery are distribution shifts, and that precision, recall or mAP can only be measured on a labelled image set with a phrase vocabulary they supply. Users who adapt the heads are also expected to understand box annotation, image-disjoint train/validation/test splits, AP at a stated IoU threshold, validation-based epoch selection, and that fine-tuning shared heads for one vocabulary can change detections for other phrases. Users who need segmentation masks, tracking, image-guided (one-shot) queries, calibrated probabilities or real-time throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no instance or semantic segmentation, no tracking, no image-conditioned (one-shot) detection although the upstream model supports it, no video, no non-maximum suppression, no OCR or captioning, and no calibrated confidence. Batch helpers exist for evaluation but real-time serving is not claimed. Adaptation covers only the class and box heads; the image and text towers are never fine-tuned.
2. **Input boundary:** `detect` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a single string instead of a list of phrases, more than `MAX_PROMPTS = 16` phrases, empty or duplicate phrases, phrases longer than `MAX_PROMPT_CHARS = 48`, and thresholds outside `[0, 1]` (`ValueError`). Each phrase is tokenised to at most `MAX_TEXT_TOKENS = 16` CLIP tokens and silently truncated beyond that. One image per call; at most `MAX_DETECTIONS = 3600` boxes (one per patch). The square padding plus 960×960 resize means a 4:1 panorama is encoded at a quarter of its width's resolution.
3. **Input boundary:** the CLIP backbone was trained on web image–caption pairs and the detection heads on COCO, OpenImages, Objects365 and web-scale pseudo-labels (per the paper and the snapshot README, which notes its data section is "to be updated for v2"). Drawn graphics (the tutorial's shapes and icons), documents and screenshots, medical, aerial, thermal or underwater imagery, tiny or heavily occluded objects, and phrases describing attributes, actions or relations rather than objects fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded.
4. **Decision boundary:** not for autonomous decisions that act on detections — vehicle control, security or surveillance alerts, safety interlocks, medical or industrial inspection, content moderation — without a human reviewing the detections and a locally measured precision/recall on the deployment's own labelled images with the deployment's own phrase vocabulary at the chosen threshold.

#### Factors

###### Groups

This pipeline is human-centric whenever a prompt names a person or a personal attribute: a free-text query such as `a photo of a person`, `a woman`, `a child`, or a query naming clothing, religious dress, skin, age or disability makes the model localise people by that description, and the CLIP text tower is known from the CLIP literature to encode social biases in how such phrases match images. The upstream README warns that the web-crawled training data "is more representative of people and societies most connected to the internet"; neither the upstream authors nor this repository audited per-group detection performance, so the fairness of any person-related deployment is unknown, not known to be equal. Objects vary by region as well: the vocabulary is open, but the recall for a named object depends on how often that object appeared with that name in web data, which favours Western, English-language contexts. The operator who detects people or culturally specific objects is responsible for a per-group audit on their own imagery and vocabulary before relying on the output.

###### Instrumentation

The upstream training instruments are web image–text pairs (any camera, any quality, with captions as weak labels) and detection datasets with human-drawn boxes (COCO, OpenImages, Objects365) plus the model's own pseudo-labels on web images. Inference images arrive from whatever produced them, and the square padding plus 960×960 resize is applied regardless of source, discarding resolution unevenly for non-square images. The text side is an instrument too: the phrasing (`a photo of a red circle` vs `red circle`), capitalisation, and truncation at 16 tokens all change the query embedding and therefore the scores. The pipeline validates image type and size and phrase form only; it cannot detect a rendered scene, a night frame, a truncated phrase's changed meaning, or a phrase the image does not contain. The synthetic tutorial scene (Pillow shapes on grey) is a rendering instrument far from any camera: the three drawn shapes are found with IoU 0.95–0.97 and, on the sibling RT-DETR's drawn icons, OWLv2 finds all four including the sports ball the closed-set detector misses — evidence about drawings, not photographs.

###### Environment

Operating environment: Python 3.12 with the exact pins in `pyproject.toml`, float32 model and adapter tensors, and CUDA selected automatically when visible. The inference-only carrier previously ran on CPU, but the E2E adaptation candidate is targeted at a hosted Tesla T4 and has no supported-runtime timing or result yet; its model-backed evidence is deferred to the serial Kaggle gate. The frozen-tower cache stores about 1.7 GB of half-precision host tensors for the 300 train/validation records. Data environment: the default adaptation sample is BCCD blood-smear microscopy, which is a severe shift from the web photographs and detection corpora used upstream. Any measured result applies only to the three sample phrases and one seeded image-disjoint split; real cameras, other microscopes, other cell preparations, other phrase vocabularies, documents, aerial imagery, and attribute or relation phrases require separate measurement.

#### Metrics

###### Performance Measures

Each detection carries `score`, the uncalibrated sigmoid of the patch's best image–text logit. For labelled datasets, `detection_metrics` reports `map50` (the mean of per-phrase VOC all-points AP at `IOU_THRESHOLD = 0.5`), pooled `precision`, `recall`, and `f1` at the caller's score threshold, plus reference, predicted, and matched box counts. AP captures ranked localisation/classification across a phrase; precision and recall expose the operating point that a single ranking measure hides, and box counts reveal duplicate-heavy outputs. `evaluation_report` remains a per-image plumbing/sanity report for drawn references, not corpus evidence. The notebook compares the adapted model with an empty baseline, a grid-prior baseline, and the same frozen model on the identical held-out records. The upstream paper's benchmark numbers are not reproduced or claimed.

###### Decision thresholds

`DETECTION_THRESHOLD = 0.1` keeps a patch box only when its best phrase sigmoid reaches 0.1; below it the box is discarded, and there is no non-maximum suppression. `IOU_THRESHOLD = 0.5` defines matching for AP, precision, and recall. Neither value is a deployment acceptance threshold. The adaptation loop chooses the earliest epoch with the highest validation `map50`; the test split is not used for selection. A deployment owns tuning its score threshold on separate validation data under the asymmetric cost of false boxes versus missed objects, and must re-tune when the camera, scene, phrase wording, vocabulary, or adapter changes.

###### Approaches to uncertainty and variability

The tutorial uses one deterministic draw (`SAMPLE_SEED = 42`) from 364 BCCD images: 260 train, 40 validation, and 64 test records, de-duplicated by decoded-pixel digest. It reports one run and no dispersion estimate. Seeded shuffling controls record order, but GPU kernels and optimiser arithmetic may still vary, so the clean-runtime result is not a confidence interval. The sigmoid is not calibrated; callers who need calibrated probabilities must fit and validate a calibration map on their own held-out images and vocabulary. To estimate metric uncertainty, repeat across seeds or bootstrap at an appropriate source/session grouping rather than treating 64 correlated images as independent deployment evidence.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states that the CLIP backbone was trained on publicly available image–caption data assembled by crawling websites and from datasets such as YFCC100M, that the detection heads were fine-tuned on COCO and OpenImages, and that the data section is "to be updated for v2"; the OWLv2 paper adds Objects365 and self-training on WebLI-scale web images with pseudo-labels. Web-crawled image–text data contains identifiable people, personal environments, text, licence plates, names, demographic descriptors, and copyrighted imagery; it was not audited here. The E2E tutorial fetches the Public-Domain BCCD Roboflow export from `keremberke/blood-cell-object-detection` at immutable revision `22cf1b9d2367e799ab54a16774a2266e4f8ce9a9`: 364 blood-smear images with 4,886 boxes over platelet, red-blood-cell, and white-blood-cell classes. Each of the three parquet files is pinned by byte count and SHA-256 and is not redistributed by this repository. The pipeline validates structure and digests, not label truth, consent, or representativeness; operators remain responsible for the images and phrases they submit.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — describing and locating people by attribute for surveillance or profiling, weapon or contraband phrases for security screening, hazard phrases for safety interlocks, lesion or defect phrases for medical or industrial inspection — would be admissible only with human review of the detections, a locally measured precision/recall on the deployment's own labelled imagery and vocabulary stratified by the groups named above, a documented threshold, phrasing and NMS policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 9 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and detected with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, prompts, *, threshold)` stage applies exactly the checks `detect` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the normalised queries and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `format_prompts` rejects a bare string, empty, duplicate, over-long or too many phrases; thresholds outside `[0, 1]` (and booleans) are rejected; `detect` raises on a malformed backend object, a label that is not one of the queries, or more than 3,600 detections.
- **Dataset and split integrity:** sample files are digest-pinned; every record, phrase, and box is bounded and validated; decoded-pixel digests prevent one image from crossing splits; validation selects the epoch and the test split is reserved for final reporting. BYOD zip members are read without `extractall`.
- **Artifact integrity:** only the class-head and box-head tensors may be exported; the manifest binds model id, immutable revision, base-weight digest, tensor names, adapter digest, thresholds, vocabulary, and training history. Reload refuses foreign bases, altered files, unexpected tensors, or malformed thresholds before applying weights.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; the processor is used as the snapshot declares it; evaluation and adaptation records carry model identity, thresholds, seed, split sizes, and metrics. One seeded run has no dispersion and is labelled accordingly.
- **Refusals:** no download without the explicit flag, no hidden threshold, no NMS or de-duplication that would hide raw model behaviour, no pickle deserialisation, no unverified model or corpus file, and no attempt to infer whether a phrase is semantically valid for an image.

###### Risks and harms

- **Phrase-driven false positives:** every patch proposes a box for every phrase; a phrase the image does not contain can still surface just above a low threshold and be labelled with that phrase, so a system that trusts labels finds what it asks for. The only guard is the uncalibrated score.
- **Duplicates without NMS:** one object can be returned as several overlapping boxes at a low threshold (observed at 0.05 and, on icons, at 0.1); counting or cropping by box count over-counts.
- **Person and attribute queries:** free text makes the pipeline a describe-and-locate tool for people with no consent or purpose check, and CLIP's documented social biases shape which descriptions match whom.
- **Phrase truncation and phrasing sensitivity:** a phrase over 16 CLIP tokens is cut silently; scores shift with wording, so a pipeline that changes prompts changes behaviour without any other signal.
- **Automation bias:** tight boxes with high scores invite trust that an uncalibrated sigmoid has not earned; the shapes scene's best-localised object had the lowest score.
- **Aspect and resolution loss:** square padding plus the 960×960 resize degrades wide or tall images and small objects silently.
- **Bias amplification:** any object, scene or population under-represented in web image–text data is reproduced as uneven recall for its name, undetected because no per-group evaluation exists.
- **Adaptation shift:** the same shared heads serve every phrase; fitting three blood-cell phrases can degrade or otherwise change boxes for unrelated prompts. Re-evaluate all deployment vocabularies after loading an adapter.
- **Resource use:** the 620 MB base plus a frozen-tower cache of about 1.7 GB for the default train/validation records; the E2E path requires a hosted GPU budget and has no recorded exact-blob timing yet.

###### Use cases

Prohibited even where the model would work: describing and locating people by attribute in order to surveil, track, profile, or score them, or to enable unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement; processing imagery the operator has no right to process, or in breach of consent, privacy or data-protection obligations; deceptive uses that present detections as verified facts or as evidence; autonomous physical control or safety interlocks based on unreviewed detections; and any use that violates the upstream Apache-2.0 licence terms or the terms of the deployment that runs the pipeline. Autonomous high-consequence actions triggered by unreviewed detections are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `google/owlv2-base-patch16-ensemble`
- Revision: `cfd3195ba4ea9592eec887ded089f4c08eff231d`
- Snapshot manifest: `weights/owlv2-base-patch16-ensemble/dimer-base-manifest.json`, 9 files, `totalBytes` 621510370
- `model.safetensors` SHA-256: `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` (619,918,824 bytes, float32)
- `config.json` SHA-256: `ba9df8c25a4b8461887dd0a93d9252c9cd84697fe8d49a9d8794ce409af9acb2` (414 bytes; `Owlv2ForObjectDetection`, image size 960, patch 16, projection 512)
- `preprocessor_config.json` SHA-256: `cf3e396635b797ee1a464e1b2836e98748f8edac19e89aaa2c93b55ac15b0064` (425 bytes; `Owlv2ImageProcessor`, pad to square, 960×960, CLIP mean/std)
- Weight format: SafeTensors; loader `Owlv2ForObjectDetection.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)` with `Owlv2Processor` from the same directory. The upstream `pytorch_model.bin` is not part of the manifest and is never loaded.

## Input/output contract

- `Owlv2DetectionPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `detect(image, prompts, *, threshold=0.1) -> dict` with keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": <one of the normalised queries>, "score": float}` in input-pixel coordinates, sorted by descending score, at most 3,600 entries), `queries`, `threshold`, `width`, `height`, `model_id`, `model_revision`.
- `detect_batch(images, prompts, *, threshold=0.1, batch_size=4) -> list[dict]`; `evaluate(records, prompts, *, threshold, batch_size) -> dict`; `adapt(train_records, validation_records, prompts, *, epochs, learning_rate, batch_size, seed, threshold) -> dict` with the towers frozen and the earliest best-validation-mAP epoch restored.
- `fetch_corpus` / `read_corpus` / `build_sample_dataset` acquire and construct the pinned BCCD sample; `validate_dataset`, `check_split_disjoint`, `split_dataset`, and `load_byod_dataset` enforce the labelled-box and split contract. `detection_metrics`, `empty_baseline`, and `grid_baseline` share the same IoU and score-threshold semantics as model evaluation.
- `save_artifact(path, training_summary=...)` writes `adapter.safetensors` plus `manifest.json`; `from_artifact(path, weights_dir=...)` verifies the base binding, artifact digest, and exact head tensor set before fresh-base reload.
- `format_prompts(prompts) -> list[str]` (whitespace-collapsed, lower-cased, trailing period removed, distinct); ceilings `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 16`, `MAX_DETECTIONS = 3600`, `DETECTION_THRESHOLD = 0.1`, `INPUT_SCHEMA`.
- `box_iou(a, b) -> float` on xyxy boxes; `validate_inputs(image, prompts, *, threshold, names) -> dict`; `evaluation_report(result, ground_truth_boxes=None, *, sample_kind) -> dict` where `ground_truth_boxes` maps a phrase to its xyxy reference box; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing pad to square (grey, bottom/right), resize to 960×960, CLIP mean/std (`Owlv2ImageProcessor`, snapshot defaults, slow processor as declared); one CLIP text query per phrase, 16 tokens.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.35 s (9 files, 620 MB); load 5.32 s; `detect` on a synthetic 640×480 scene (grey background, a black rectangle at [80, 120, 280, 360], a red disc in [380, 140, 560, 320], a blue triangle in [200, 380, 360, 460]) with prompts `["a black rectangle", "a red circle", "a blue triangle"]` at the default threshold 0.1 → 3 detections in 3.39 s (2.91 s on the second call): `a red circle` [379.3, 138.4, 560.5, 322.4] 0.917, `a blue triangle` [199.7, 380.9, 361.7, 462.2] 0.810, `a black rectangle` [79.7, 119.4, 283.9, 360.6] 0.426; `box_iou` against the drawn boxes 0.974 / 0.971 / 0.950 with agreeing labels; the same scene gave 5 boxes at 0.05 and 3 at 0.2 and 0.3. The sibling RT-DETR pipeline's drawn icon scene (stop sign, traffic light, clock, sports ball) with prompts `a photo of a <thing>` → 6 detections in 3.17 s: all four icons found (`stop sign` 0.835 / IoU 0.896, `clock` 0.681 / 0.942, `traffic light` 0.614 / 0.931, `sports ball` 0.349 / 0.945 — the ball the closed-set detector misses) plus two duplicate boxes (`clock` 0.111, `stop sign` 0.106). 4096×4096 blank image with `["a photo of a cat"]` → 0 detections in 4.43 s.
- Historical inference-only execution: the superseded `TASK-INFERENCE` notebook passed Kaggle CPU on 2026-09-14 (8/8 code cells, 243.2 s); that blob is not evidence for the current E2E carrier.
- Current E2E verification state: source/static and offline unit checks only. The exact notebook blob still requires the queued Kaggle Tesla T4 model-backed run; no E2E metric, timing, adaptation, or reload result is claimed before that run.
- Not executed for this candidate: the E2E default path, CUDA model-backed adaptation, held-out corpus metrics, or fresh-base artifact parity. Image-guided detection, the fast image processor, and phrases near the 16-token limit remain outside the demonstrated path.

## References

- Minderer, Gritsenko, Houlsby. Scaling Open-Vocabulary Object Detection. NeurIPS 2023. https://arxiv.org/abs/2306.09683
- Minderer et al. Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT). ECCV 2022. https://arxiv.org/abs/2205.06230
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
- Upstream code (Scenic, OWL-ViT project): https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
- Upstream card: https://huggingface.co/google/owlv2-base-patch16-ensemble
- Transformers `OWLv2` documentation: https://huggingface.co/docs/transformers/model_doc/owlv2
