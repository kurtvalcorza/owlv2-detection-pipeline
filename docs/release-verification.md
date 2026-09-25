# Release verification

`tutorials/owlv2_detection_colab.ipynb` (`E2E`, **standalone** carrier) is
**Release-grade** for the exact blob recorded below and returns to **Candidate** whenever the blob
changes, until that exact blob executes top-to-bottom in a clean supported runtime. Source validation, unit tests, generator parity, and the historical inference-only
run are not model-backed evidence for this E2E carrier under DIMER Notebook Specification 2.0.

## Automatic coverage (static, every pull request)

CI and `tools/validate_release_assets.py` verify that:

- the notebook JSON parses, code cells compile, outputs and execution counts are absent, and
  unresolved placeholders are rejected;
- the notebook declares `E2E`, `GUIDED`, Notebook Specification 2.0, and `standalone: true`;
- the primary path performs no repository clone/install/import and embeds `metrics.py`,
  `samples.py`, and `pipeline.py` in dependency order with source SHA-256 parity;
- the inline model manifest and runtime pins match the repository and the generated notebook is
  byte-identical to `tools/build_notebook.py` output;
- the immutable model revision, per-file digests, and local-only SafeTensors load contract agree
  across the package, notebook, model card, README, and weight documentation;
- the default sample uses the three digest-pinned BCCD parquet files at the immutable dataset
  revision, constructs 260 / 40 / 64 image-disjoint records, and runs without a credential or
  upload prompt;
- validation, baselines, frozen evaluation, bounded adaptation, held-out evaluation, inference,
  SafeTensors export, and fresh-base reload are all represented on the default path;
- BYOD is optional and gated off by default, but accepts a zip or directory of images with a
  `boxes.csv` and routes records through the same validation, split, adaptation, evaluation, and
  export semantics; and
- `README.md`, `STATUS.md`, this file, and `tutorials/README.md` agree on one status token for
  the current carrier.

The offline suite exercises dataset validation, digest pinning, split disjointness, metric and
baseline semantics, Hungarian assignment, artifact manifest refusal, notebook parity, and the
inference role helpers. Model-backed tests are skipped when Torch and the verified snapshot are not
available. Neither an offline pass nor a source validator is clean-runtime evidence.

## Supported executor

The release gate is the workspace `kaggle-serial-gpu-test-suite` running the exact committed blob on
a Kaggle `NvidiaTeslaT4` kernel. The harness must download the notebook from an immutable 40-character
commit SHA, verify its Git blob SHA-1, execute it with `nbclient` in a fresh interpreter, honour an
installation-cell restart, and harvest `evidence/run_summary.json` plus the executed notebook. Only
one fleet kernel may run at a time.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. Resolve the exact commit and notebook blob under review; confirm the source/static gate is green.
2. Execute the exact blob on Kaggle Tesla T4 with `USE_BYOD = False` and all other sample-path
   defaults, using the serial suite and a clean output directory.
3. Confirm the notebook installs the inline pins, restarts if required, and records
   `NOTEBOOK_SOURCE.repository_revision` equal to the committed revision.
4. Confirm all 9 model files and all 3 BCCD parquet files are staged from their immutable revisions
   and pass byte-count and SHA-256 verification before use.
5. Confirm the sample split is exactly 260 train / 40 validation / 64 test, with no decoded image
   digest shared between splits, and that every demonstrated invalid dataset/input is refused.
6. Confirm the drawn-scene inference contract completes and writes its input manifest and
   `sample-sanity` report without treating drawn-box IoU as corpus performance.
7. Confirm the empty and grid-prior baselines and the frozen model are scored on the same held-out
   records at the same score and IoU thresholds.
8. Confirm only the OWLv2 class and box heads are trainable; the image/text towers remain frozen;
   validation mAP selects the earliest best epoch; and the test split is not used for selection.
9. Confirm the held-out comparison reports per-phrase AP, mAP, precision, recall, F1, reference and
   predicted box counts, and makes no claim beyond one seeded split of one sample.
10. Confirm `adapter.safetensors` and its manifest are exported, loaded into a newly constructed
    verified base, and pass the notebook's detection-parity assertion on held-out records.
11. Confirm every output named in `tutorials/README.md` exists and all code cells complete without
    an exception.
12. Append the exact commit, notebook blob, kernel version, runtime/package/device identity, wall
    time, cell counts, staged file/byte counts, metrics, reload-parity result, warnings, and PASS or
    FAIL verdict below. Record no secrets.

A failure of the default path, an unverified asset, an altered carried module, or missing fresh-base
reload parity blocks promotion. A numerical improvement is not assumed: if the adapted model fails
the notebook's stated comparison assertions, record FAIL and remediate the carrier rather than
editing the evidence.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/owlv2_detection_colab.ipynb` at the recorded
commit. Wall time is the serial executor's sum of per-cell durations and includes installs and
downloads.

| Carrier | Commit / notebook blob | Date | Executor | Outcome |
|---|---|---|---|---|
| Current `E2E` carrier | `a772a31` / `c9cd132564ae` (notebook `NOTEBOOK_SOURCE.repository_revision` `49800d8`, the source revision the notebook was bound to; `49800d8..a772a31` changes only the notebook) | 2026-09-25 (22:33–22:44 UTC) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-owlv2-detection` v2), serial suite, `USE_BYOD = False`, sample-path defaults; blob fetched at the 40-char SHA and Git-blob verified; HF cache clean at start | **PASSED** — 705.1 s wall (pass 1 194.8 s stopped at the install cell with a pip dependency-resolver `CellExecutionError`, kernel restarted after the install cell; pass 2 510.2 s), 11/11 post-restart code cells ok; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins: torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, device `cuda:0`, float32; staged 23 files / 626 MB (9 model files incl. `model.safetensors` 619,918,824 B, sha256 `e1e130b9…99e7`, + 3 BCCD parquet files, all digest-verified); split 260 / 40 / 64, image-disjoint; 4 dataset refusals (duplicate id, box outside image, record without boxes, too small) and 1 over-long-phrase refusal demonstrated; trainable 1,579,526 of 154,966,792 parameters (class + box heads), 8 epochs, best epoch 8 by validation mAP (validation mAP50 0.8734, n = 40); held-out test (64 records, 846 reference boxes, threshold 0.1, IoU 0.5): mAP50 empty 0.0 / grid prior 0.0147 / frozen 0.0723 / adapted 0.8918; precision 0.0 / 0.0146 / 0.1295 / 0.2236; recall 0.0 / 0.2967 / 0.0591 / 0.9894; F1 0.0 / 0.0279 / 0.0812 / 0.3647; per-phrase AP frozen → adapted: platelet 0.0 → 0.8805 (50 refs), red blood cell 0.0 → 0.8476 (731), white blood cell 0.2168 → 0.9473 (65); predicted boxes frozen 386 / adapted 3744, matched 50 / 837; `adapter.safetensors` 6,319,224 B sha256 `21694de50e3f3e5b59116cb482126669d527e3c36ee3b3f281046b14d027df7a`, fresh-base reload parity 8/8 identical detections; preserved output sha256: `owlv2_detection_result.json` `3413c0515876…`, `owlv2_detection_evaluation_report.json` `4375f7c4b904…`, adapter `manifest.json` `8cf4dc477d40…`, `owlv2_detection_train.csv` `0557c03e6b02…`; warning: one `UserWarning` (tensor with `requires_grad=True` converted to a scalar) in the adaptation cell. One seeded split of one sample, one runtime, no dispersion estimate |
| Superseded `TASK-INFERENCE` | `6c9365e` / `ce245f5aa2c5` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-owlv2-detection` v1) | **PASSED** — 8/8 code cells, 243.2 s, 20 files, 622 MB staged; historical inference evidence only, not evidence for the E2E blob |
| Superseded `TASK-INFERENCE` local pre-flight | notebook blob `a37382571bee` (commit `2dd34c2`, generated at `5a8fa2f`) | 2026-09-14 | Local Windows fresh-kernel CPU harness | **PASSED** — 8/8 code cells, 68.6 s; pre-flight only |

## Current status

**Release-grade** for the exact commit `a772a31` / notebook blob `c9cd132564ae` recorded above: the
default sample path executed top-to-bottom on Kaggle Tesla T4 with every gate in the procedure above
satisfied (pinned assets verified, 260 / 40 / 64 image-disjoint split, refusals demonstrated, baselines
and frozen model scored on the same held-out records, heads-only adaptation selected on validation,
adapter export and fresh-base reload parity 8/8). The measured values are one seeded split of one
blood-smear sample on one runtime; they carry no dispersion estimate and are not a benchmark. Any change
to the notebook blob returns the carrier to Candidate until a new exact-blob run is recorded. The
historical inference rows remain regression context only.
