---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: zero-shot-object-detection
task: "Others - Open-Vocabulary Object Detection"
base_model: google/owlv2-base-patch16-ensemble
date_published: "2023-10-13"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2023-10-13T09:27:09Z, https://huggingface.co/api/models/google/owlv2-base-patch16-ensemble — the Transformers-format conversion); the OWLv2 paper is arXiv:2306.09683 (2023-06) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# OWLv2 base/16 ensemble — Open-Vocabulary Object Detection (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlv2--base--patch16--ensemble-ffcc4d?style=flat)](https://huggingface.co/google/owlv2-base-patch16-ensemble)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2306.09683-b31b1b.svg)](https://arxiv.org/abs/2306.09683)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/owlv2-detection-pipeline/blob/main/tutorials/owlv2_detection_colab.ipynb) [`owlv2_detection_colab.ipynb`](https://github.com/kurtvalcorza/owlv2-detection-pipeline/blob/main/tutorials/owlv2_detection_colab.ipynb)  
  *Text-prompted detection on a scene drawn in code with the pinned `google/owlv2-base-patch16-ensemble` weights: score-ordered boxes labelled with the matched phrase under a caller-owned `threshold`, and per-object `box_iou` against the drawn references as sanity evidence only — no mAP.*

---

#### Description

`google/owlv2-base-patch16-ensemble` is the Transformers-format release of OWLv2 — "Scaling Open-Vocabulary Object Detection" (Minderer, Gritsenko and Houlsby, arXiv:2306.09683; Google Research) — the base/16 variant whose detection heads were trained with self-training on web-scale pseudo-labels and then ensembled by weight-space averaging with a fine-tuned checkpoint (the "ensemble" suffix), converted by the Hugging Face team and pinned here to revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `Owlv2ForObjectDetection` (`model_type` owlv2): a CLIP ViT-B/16 image tower over a 960×960 input (60×60 = 3,600 patch tokens, `projection_dim` 512) and a CLIP text tower (BPE tokenizer, `model_max_length` 16 per query); the float32 `model.safetensors` is 620 MB. At inference the processor (`Owlv2ImageProcessor`, `preprocessor_config.json`) pads the image to a square with grey on the bottom and right, resizes it to 960×960 and normalises with CLIP mean/std; each of the caller's phrases becomes one text-query embedding; every image patch predicts one box and one image-text logit per query, the box's score is the **sigmoid of its best query logit** and its label that query; `post_process_grounded_object_detection` applies the threshold and maps boxes back to the input's pixel frame (the pinned transformers release scales by `max(height, width)` to undo the padding). There is no non-maximum suppression. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `Owlv2DetectionPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`), `format_prompts` (whitespace-collapsed, lower-cased, distinct queries within named ceilings), `detect` (input validation, threshold checks, sorted pixel-space output, 3,600-candidate ceiling), the `validate_inputs` and `evaluation_report` stage helpers, and `box_iou`.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is open-vocabulary object detection: input one image (`PIL.Image.Image`, any mode, converted to RGB), 1–16 free-text phrases and a threshold; output a list of at most 3,600 detections, each an xyxy pixel box, a `label` equal to the best-matching phrase, and the model's sigmoid `score`, sorted by score. Envisioned applications are detection of objects that closed-set detectors do not know — a product, a part, a sign, a species named in words — as a labelling assistant, a first pass for cropping or counting, a retrieval filter over image collections, and a comparison baseline against the sibling closed-set RT-DETR pipeline. The pipeline is an inference component and a zero-configuration baseline for text-conditioned detection, not a certified detector for any specific vocabulary, camera or scene.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers, and data analysts integrating text-prompted detection into research prototypes or in-house tooling. A user is expected to understand that the label is the phrase a patch scored best against, not a verified identity — every patch proposes a box, and a phrase the image does not contain can still surface just above a low threshold — that the score is a sigmoid of an image–text logit, a ranking signal within one image that is not calibrated on their data and not exclusive across queries, that there is no non-maximum suppression so one object can appear as several boxes at a low threshold, that phrases are truncated at 16 CLIP tokens and work best in the upstream `a photo of a <thing>` form, that the image is padded to a square so a wide or tall image is encoded at a lower effective resolution, that the training data is web image–text pairs and detection datasets so drawn graphics, documents, medical and aerial imagery are distribution shifts, and that precision, recall or mAP can only be measured on a labelled image set with a phrase vocabulary they supply. Users who need segmentation masks, tracking, image-guided (one-shot) queries or real-time throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no instance or semantic segmentation, no tracking, no image-conditioned (one-shot) detection although the upstream model supports it, no batching or video, no non-maximum suppression, no OCR or captioning, and no calibrated confidence. Real-time use is not claimed: the base/16 tower at 960×960 takes about 3 s per image on the reference CPU.
2. **Input boundary:** `detect` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a single string instead of a list of phrases, more than `MAX_PROMPTS = 16` phrases, empty or duplicate phrases, phrases longer than `MAX_PROMPT_CHARS = 48`, and thresholds outside `[0, 1]` (`ValueError`). Each phrase is tokenised to at most `MAX_TEXT_TOKENS = 16` CLIP tokens and silently truncated beyond that. One image per call; at most `MAX_DETECTIONS = 3600` boxes (one per patch). The square padding plus 960×960 resize means a 4:1 panorama is encoded at a quarter of its width's resolution.
3. **Input boundary:** the CLIP backbone was trained on web image–caption pairs and the detection heads on COCO, OpenImages, Objects365 and web-scale pseudo-labels (per the paper and the snapshot README, which notes its data section is "to be updated for v2"). Drawn graphics (the tutorial's shapes and icons), documents and screenshots, medical, aerial, thermal or underwater imagery, tiny or heavily occluded objects, and phrases describing attributes, actions or relations rather than objects fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded.
4. **Decision boundary:** not for autonomous decisions that act on detections — vehicle control, security or surveillance alerts, safety interlocks, medical or industrial inspection, content moderation — without a human reviewing the detections and a locally measured precision/recall on the deployment's own labelled images with the deployment's own phrase vocabulary at the chosen threshold.

#### Factors

###### Groups

This pipeline is human-centric whenever a prompt names a person or a personal attribute: a free-text query such as `a photo of a person`, `a woman`, `a child`, or a query naming clothing, religious dress, skin, age or disability makes the model localise people by that description, and the CLIP text tower is known from the CLIP literature to encode social biases in how such phrases match images. The upstream README warns that the web-crawled training data "is more representative of people and societies most connected to the internet"; neither the upstream authors nor this repository audited per-group detection performance, so the fairness of any person-related deployment is unknown, not known to be equal. Objects vary by region as well: the vocabulary is open, but the recall for a named object depends on how often that object appeared with that name in web data, which favours Western, English-language contexts. The operator who detects people or culturally specific objects is responsible for a per-group audit on their own imagery and vocabulary before relying on the output.

###### Instrumentation

The upstream training instruments are web image–text pairs (any camera, any quality, with captions as weak labels) and detection datasets with human-drawn boxes (COCO, OpenImages, Objects365) plus the model's own pseudo-labels on web images. Inference images arrive from whatever produced them, and the square padding plus 960×960 resize is applied regardless of source, discarding resolution unevenly for non-square images. The text side is an instrument too: the phrasing (`a photo of a red circle` vs `red circle`), capitalisation, and truncation at 16 tokens all change the query embedding and therefore the scores. The pipeline validates image type and size and phrase form only; it cannot detect a rendered scene, a night frame, a truncated phrase's changed meaning, or a phrase the image does not contain. The synthetic tutorial scene (Pillow shapes on grey) is a rendering instrument far from any camera: the three drawn shapes are found with IoU 0.95–0.97 and, on the sibling RT-DETR's drawn icons, OWLv2 finds all four including the sports ball the closed-set detector misses — evidence about drawings, not photographs.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. The snapshot declares the slow `Owlv2ImageProcessor` (transformers prints a `use_fast` notice); it is used as declared. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 9-file, 620 MB snapshot 0.35 s, load 5.32 s, a 640×480 drawn scene with three prompts 3.39 s (first call) / 2.91 s, the same scene size with four prompts 3.17 s, a 4096×4096 blank image with one prompt 4.43 s — cost is dominated by the fixed 960×960 ViT-B/16 pass and grows only mildly with the prompt count. Data environment: the model assumes a photograph in which the named things are visible and the phrases are object nouns; the synthetic scene satisfies the second condition and not the first, and is where the measured behaviour holds. Real cameras, attribute or relation phrases, tiny objects and non-photographic imagery violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — beyond the sigmoid score, which is not calibrated.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each detection carries `score`, the sigmoid of the patch's best image–text logit — a ranking signal within one image, not a probability that the box is a real instance of the phrase, not exclusive across phrases for the same patch, and not a measure of correctness. The repository ships `box_iou(a, b)`, the intersection-over-union of two xyxy boxes, because it is the primitive every detection metric is built from; mean average precision, per-phrase precision/recall and the LVIS/COCO zero-shot AP the upstream paper reports are not implemented, since they need a labelled image set with a phrase vocabulary that the caller must choose. To evaluate, the caller supplies ground-truth boxes per phrase and computes precision/recall at their chosen IoU with `box_iou`, or runs a COCO/LVIS-style evaluator. The public `evaluation_report(result, ground_truth_boxes=None)` stage returns that report in machine-readable form: one `box_iou` entry per supplied reference box (its best-overlapping detection, that detection's label, and whether the label agrees with the reference phrase) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no reference is supplied. The upstream paper's zero-shot numbers (LVIS rare-class AP for the base/16 ensemble, among others) are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

One threshold is applied and exposed as a module constant: `DETECTION_THRESHOLD = 0.1` keeps a patch's box only if the sigmoid of its best query logit is at least 0.1; below it the box is discarded, and there is no non-maximum suppression. This is the value the pinned README's usage example passes to `post_process_object_detection`; it was not tuned by this repository and is not calibrated for any deployment. It can be overridden per call (`detect(..., threshold=)`), and the smoke run shows how it behaves on drawn scenes: the shapes scene gave exactly the three drawn objects at 0.1, 0.2 and 0.3 and five boxes at 0.05 (duplicates), the icon scene gave six boxes at 0.1 for four objects (two duplicates at 0.106 and 0.111), and a blank image gave none. A deployment owns tuning it on its own labelled images and vocabulary: raise it (0.2–0.3 on this evidence) when duplicates or false boxes cost more than misses, lower it for recall, apply its own NMS if one box per object is required, and re-tune whenever the camera, scene, phrasing or vocabulary changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, boxes, scores and per-object IoU on two drawn scenes, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines (the shapes scene has no text rendering and is byte-stable across Pillow builds; the icon scene is not). The `score` is a sigmoid, not a calibrated confidence: on the shapes scene the three drawn objects scored 0.92, 0.81 and 0.43 with `box_iou` 0.95–0.97 — the black rectangle scored lowest while being localised best, which is what an uncalibrated ranking signal looks like — and on the icon scene the sports ball scored 0.35 with IoU 0.945. A caller who needs calibrated confidences must fit a calibration map on their own labelled images and vocabulary; a caller who needs an uncertainty estimate for a metric must supply labelled images and compute it over many images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states that the CLIP backbone was trained on publicly available image–caption data assembled by crawling websites and from datasets such as YFCC100M, that the detection heads were fine-tuned on COCO and OpenImages, and that the data section is "to be updated for v2"; the OWLv2 paper adds Objects365 and self-training on WebLI-scale web images with pseudo-labels. Web-crawled image–text data contains photographs of identifiable people, personal environments, text and licence plates, captions with names and demographic descriptors, and copyrighted imagery; personal data in the training corpus is present by construction and was not audited here. This repository distributes code, tests, and documentation; it does not distribute the 619,918,824-byte `model.safetensors`, which is staged locally under `weights/owlv2-base-patch16-ensemble/` and git-ignored, and it ships no sample photographs — the tutorial scene is drawn in code. The operator must audit the images and the phrases they submit; the pipeline performs no content check and will localise `a photo of a person wearing a hijab` as readily as `a red circle`.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — describing and locating people by attribute for surveillance or profiling, weapon or contraband phrases for security screening, hazard phrases for safety interlocks, lesion or defect phrases for medical or industrial inspection — would be admissible only with human review of the detections, a locally measured precision/recall on the deployment's own labelled imagery and vocabulary stratified by the groups named above, a documented threshold, phrasing and NMS policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 9 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and detected with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, prompts, *, threshold)` stage applies exactly the checks `detect` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the normalised queries and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `format_prompts` rejects a bare string, empty, duplicate, over-long or too many phrases; thresholds outside `[0, 1]` (and booleans) are rejected; `detect` raises on a malformed backend object, a label that is not one of the queries, or more than 3,600 detections.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; the processor is used as the snapshot declares it; every result carries `model_id`, `model_revision`, the normalised `queries` and the threshold used.
- **Refusals:** no batching, no download without the explicit flag, no threshold default hidden inside the runner, no NMS or de-duplication that would hide the model's raw behaviour, no pickle deserialisation, no attempt to guess whether a phrase is present in the image.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Phrase-driven false positives:** every patch proposes a box for every phrase; a phrase the image does not contain can still surface just above a low threshold and be labelled with that phrase, so a system that trusts labels finds what it asks for. The only guard is the uncalibrated score.
- **Duplicates without NMS:** one object can be returned as several overlapping boxes at a low threshold (observed at 0.05 and, on icons, at 0.1); counting or cropping by box count over-counts.
- **Person and attribute queries:** free text makes the pipeline a describe-and-locate tool for people with no consent or purpose check, and CLIP's documented social biases shape which descriptions match whom.
- **Phrase truncation and phrasing sensitivity:** a phrase over 16 CLIP tokens is cut silently; scores shift with wording, so a pipeline that changes prompts changes behaviour without any other signal.
- **Automation bias:** tight boxes with high scores invite trust that an uncalibrated sigmoid has not earned; the shapes scene's best-localised object had the lowest score.
- **Aspect and resolution loss:** square padding plus the 960×960 resize degrades wide or tall images and small objects silently.
- **Bias amplification:** any object, scene or population under-represented in web image–text data is reproduced as uneven recall for its name, undetected because no per-group evaluation exists.
- **Resource use:** a 620 MB model and ~3 s per image on the reference CPU; a batch of images saturates a shared host, and the CUDA path was not measured.

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
- `format_prompts(prompts) -> list[str]` (whitespace-collapsed, lower-cased, trailing period removed, distinct); ceilings `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 16`, `MAX_DETECTIONS = 3600`, `DETECTION_THRESHOLD = 0.1`, `INPUT_SCHEMA`.
- `box_iou(a, b) -> float` on xyxy boxes; `validate_inputs(image, prompts, *, threshold, names) -> dict`; `evaluation_report(result, ground_truth_boxes=None, *, sample_kind) -> dict` where `ground_truth_boxes` maps a phrase to its xyxy reference box; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing pad to square (grey, bottom/right), resize to 960×960, CLIP mean/std (`Owlv2ImageProcessor`, snapshot defaults, slow processor as declared); one CLIP text query per phrase, 16 tokens.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.35 s (9 files, 620 MB); load 5.32 s; `detect` on a synthetic 640×480 scene (grey background, a black rectangle at [80, 120, 280, 360], a red disc in [380, 140, 560, 320], a blue triangle in [200, 380, 360, 460]) with prompts `["a black rectangle", "a red circle", "a blue triangle"]` at the default threshold 0.1 → 3 detections in 3.39 s (2.91 s on the second call): `a red circle` [379.3, 138.4, 560.5, 322.4] 0.917, `a blue triangle` [199.7, 380.9, 361.7, 462.2] 0.810, `a black rectangle` [79.7, 119.4, 283.9, 360.6] 0.426; `box_iou` against the drawn boxes 0.974 / 0.971 / 0.950 with agreeing labels; the same scene gave 5 boxes at 0.05 and 3 at 0.2 and 0.3. The sibling RT-DETR pipeline's drawn icon scene (stop sign, traffic light, clock, sports ball) with prompts `a photo of a <thing>` → 6 detections in 3.17 s: all four icons found (`stop sign` 0.835 / IoU 0.896, `clock` 0.681 / 0.942, `traffic light` 0.614 / 0.931, `sports ball` 0.349 / 0.945 — the ball the closed-set detector misses) plus two duplicate boxes (`clock` 0.111, `stop sign` 0.106). 4096×4096 blank image with `["a photo of a cat"]` → 0 detections in 4.43 s.
- Tutorial execution: `tutorials/owlv2_detection_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 68.6 s including the 620 MB staging, same three detections as the smoke run); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, half precision, the fast image processor, image-guided detection, any mAP or precision/recall measurement against labelled images, photographs (the smoke and tutorial use drawn scenes), attribute or relation phrases, phrases near the 16-token limit.

## References

- Minderer, Gritsenko, Houlsby. Scaling Open-Vocabulary Object Detection. NeurIPS 2023. https://arxiv.org/abs/2306.09683
- Minderer et al. Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT). ECCV 2022. https://arxiv.org/abs/2205.06230
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
- Upstream code (Scenic, OWL-ViT project): https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
- Upstream card: https://huggingface.co/google/owlv2-base-patch16-ensemble
- Transformers `OWLv2` documentation: https://huggingface.co/docs/transformers/model_doc/owlv2
