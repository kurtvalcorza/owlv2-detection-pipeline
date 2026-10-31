# Release verification

`tutorials/owlv2_detection_colab.ipynb` (`E2E`, **standalone** carrier) is **Candidate**: blob `51db9a5288685` (commit `d7541b2`, PR #10 head; prose-only requote of the worked answers on top of `3283e836b0ec`) completed one pass with 0 errors on a fresh Colab Tesla T4 on 2026-10-17, Colab account 2 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 12/12 code cells, 585.7 s wall; REL13: hosted Colab T4, this blob — held-out test retained-detection AP50 (VOC matching) adapted 0.891 / frozen 0.072 / grid prior 0.015, all-candidates AP50 frozen 0.0845 / adapted 0.8929, best epoch 8, reload parity 8/8; every printed metric equals the 2026-10-17 account-1 run of `3283e836b0ec`; executed copy under `docs/execution-evidence/2026-10-17/`). REL13: earlier-blob figures, not re-run on blob `51db9a5288685`, follow. The OWL-* review-fix blob `3283e836b0ec` (commit `03ac0e7`, PR #10 head, the 2026-10-16 OWL-* review fixes) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-17 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 12/12 code cells, 565.9 s wall; held-out test retained-detection AP50 (VOC matching) adapted 0.891 / frozen 0.072 / grid prior 0.015, all-candidates AP50 frozen 0.0845 / adapted 0.8929, best epoch 8, reload parity 8/8; executed copy under `docs/execution-evidence/2026-10-17/`); forms and uploads (the REL12 BYOD exercise) were not exercised. REL13: earlier-blob figures, not re-run on blob `3283e836b0ec`, follow: the 2026-09-25
blob `c9cd132564ae` needed a manual restart after its install cell, and the 2026-10-02 review-fix blob (isolated
uv environment) `a545a73b74136ea3` (commit `cbb9f81`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, 12/12 code cells, 598.8 s wall; held-out test retained-detection AP50 adapted 0.8918 / frozen 0.0723 / grid prior 0.0147, reload parity 8/8); the REL12 BYOD exercise has not been run on a hosted runtime. Source validation, unit tests, generator parity, and the historical inference-only
run are not model-backed evidence for this E2E carrier under DIMER Notebook Specification 2.5.

## Automatic coverage (static, every pull request)

CI and `tools/validate_release_assets.py` verify that:

- the notebook JSON parses, code cells compile, outputs and execution counts are absent, and
  unresolved placeholders are rejected;
- the notebook declares `E2E`, `GUIDED`, Notebook Specification 2.5, and `standalone: true`;
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

The release gate is a fresh hosted Tesla T4 runtime running the exact committed blob in one pass: the
workspace `kaggle-serial-gpu-test-suite` on a Kaggle `NvidiaTeslaT4` kernel, or the
`colab-cli-serial-test-suite` on a fresh Google Colab T4 runtime. The harness must download the
notebook from an immutable 40-character commit SHA, verify its Git blob SHA-1, execute every cell in
order in one kernel (no restart is expected; a run that needs one does not pass), and harvest the run
summary plus the executed notebook. Only one fleet runtime may run at a time.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. Resolve the exact commit and notebook blob under review; confirm the source/static gate is green.
2. Execute the exact blob on a hosted Tesla T4 (Kaggle or Colab) with `USE_BYOD = False` and all
   other sample-path defaults, using a serial suite and a clean output directory.
3. Confirm Section 1 builds (or reuses) the isolated hash-locked environment with no restart, and the notebook records
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
   the unrounded validation retained-detection AP50 selects the earliest best epoch; and the test split is not used for selection.
9. Confirm the held-out comparison reports per-phrase and mean retained-detection AP50 (VOC matching), the display-only all-candidates AP50, precision, recall, F1, reference and
   predicted box counts, and makes no claim beyond one seeded split of one sample.
10. Confirm `adapter.safetensors` and its manifest are exported, loaded into a newly constructed
    verified base, and pass the notebook's detection-parity assertion on held-out records.
11. Confirm every output named in `tutorials/README.md` exists and all code cells complete without
    an exception.
12. Append the exact commit, notebook blob, kernel version, runtime/package/device identity, wall
    time, cell counts, staged file/byte counts, metrics, reload-parity result, warnings, and PASS or
    FAIL verdict below. Record no secrets.

A failure of the default path, an unverified asset, an altered carried module, or missing fresh-base
reload parity blocks promotion. A numerical improvement is not assumed: Section 8 records whether the
adapted model reached the frozen mAP and outperformed the training-derived grid prior as `verdicts` instead of stopping the run;
record those verdicts as measured rather than editing the evidence.

## Recorded executions

**Metric label.** "Retained-detection AP50" below is the mean per-phrase AP50 over the detections retained at score ≥ 0.1 (a precision-recall curve truncated at the score threshold). Runs recorded before the 2026-10-16 OWL-* revision scored it with the earlier greedy best-unmatched matcher; the package now uses conventional VOC matching. In a local CPU measurement the switch moved the adapted test value from 0.8914 to 0.8909 and left the frozen 0.0723 unchanged.

Notebook identity is the Git blob id of `tutorials/owlv2_detection_colab.ipynb` at the recorded
commit. Wall time is the serial executor's sum of per-cell durations and includes installs and
downloads.

| Carrier | Commit / notebook blob | Date | Executor | Outcome |
|---|---|---|---|---|
| Worked-answer requote `E2E` carrier of 2026-10-17 (PR #10; superseded as current by blob `cd3e85a1`, recorded in "Hosted Colab T4 runs of blob `cd3e85a1`" below) | `d7541b2` (`d7541b27e34ffb44df10b3d77175543144733433`) / `51db9a5288685` (`51db9a5288685d55c99c7d298c42c245bb0be9cb`; `NOTEBOOK_SOURCE.repository_revision` `297c1d7`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | 2026-10-17 (12:30:22–12:40:07 UTC) | Colab CLI 0.7.4, fresh Colab Tesla T4 on Colab account 2 (session `suite-owlv2-d7541b2-6ce9`, `colab exec -f`, stopped after the run; CLI sequential execution, not a browser Run all; order from `exec.log`, no execution counts), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, `USE_BYOD = False`, sample-path defaults; forms and uploads not exercised; kernel Python 3.13.15; isolated uv environment Python 3.12.12, 49 locked packages, built in 52 s; torch 2.14.0+cu130, transformers 4.57.6, Pillow 11.3.0, pyarrow 25.0.1, scipy 1.18.1; device `cuda:0`, float32 | **PASSED in one pass** (REL13: hosted Colab T4, this blob) — 585.7 s wall, 12/12 code cells in order, 0 errors (cells 3–5 are the carried modules and print nothing); source identity with the committed blob OK; 9 model files verified (`model.safetensors` sha256 `e1e130b9…99e7`); BCCD 364 records / 4,886 boxes, split 260 / 40 / 64; drawn scene frozen IoUs 0.974 / 0.971 / 0.95; 8 epochs, epoch 0 (frozen) validation AP50 0.043, best epoch 8 (validation 0.872); held-out test (64 records, 846 reference boxes, VOC matching): retained-detection AP50 empty 0.0 / grid prior 0.015 / frozen 0.072 / adapted 0.891; precision 0.0 / 0.015 / 0.13 / 0.223; recall 0.0 / 0.297 / 0.059 / 0.987; F1 0.0 / 0.028 / 0.081 / 0.364; per-phrase AP frozen → adapted: platelet 0.0 → 0.8805, red blood cell 0.0 → 0.8461, white blood cell 0.2168 → 0.9473; display-only all-candidates AP50 frozen 0.0845 / adapted 0.8929; predicted boxes frozen 386 / adapted 3,744, matched 50 / 835; `adapted_beats_both_baselines` True; adapter 6,319,224 B, sha256 `21694de50e3f3e5b…`; fresh-base reload parity 8/8. Against the 2026-10-17 account-1 run of `3283e836b0ec` (next row): every printed line is equal after removing timings, worker pid and the carried revision, so every worked-answer figure re-quoted from that run in `d7541b2` also matches this run; no requote needed. Evidence (byte-exact): `docs/execution-evidence/2026-10-17/` — `owlv2_detection_colab_d7541b2_colab-cli-t4.ipynb` sha256 `5d4063aa0aa2a331f53251ad64cf945e9228df01fe59368cfa8b280f08bff424`, `owlv2_detection_colab_d7541b2_exec.log` `f8657ffcf28c628975e0b285642ceb8dfc7a50953855004293b3c09aa5f902d9`, `owlv2_detection_colab_d7541b2_run_summary.json` `9f2c42ca82f06faed8bcf2909d7757b854dc243c6ad7cc876eb61b6ff6b76057`. One seeded split of one sample, one runtime, no dispersion estimate. Status stays Candidate |
| OWL-* review-fix `E2E` carrier of 2026-10-16 (superseded by the prose-only requote `d7541b2`, first row; REL13: earlier-blob figures, not re-run on blob `51db9a5288685`) | `03ac0e7` (`03ac0e79aca651eda6d0f315df41b74e8d8a9abc`) / `3283e836b0ec` (`3283e836b0eca2097d8fefef8e83f0bc9a964c33`; `NOTEBOOK_SOURCE.repository_revision` `2a68754`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | 2026-10-17 (02:57:26–03:06:52 UTC) | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-owlv2-03ac0e7-6ce9`, `colab exec -f`, stopped after the run; CLI sequential execution, not a browser Run all; order from `exec.log`, no execution counts), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, `USE_BYOD = False`, sample-path defaults; forms and uploads not exercised; kernel Python 3.13.15; isolated uv environment Python 3.12.12, 49 locked packages, built in 55 s; torch 2.14.0+cu130, transformers 4.57.6, Pillow 11.3.0, pyarrow 25.0.1, scipy 1.18.1; device `cuda:0`, float32 | **PASSED in one pass, no restart** — 565.9 s wall, 12/12 code cells in order, 0 errors (cells 3–5 are the carried modules and print nothing); 9 model files verified (`model.safetensors` sha256 `e1e130b9…99e7`); BCCD 3 parquet files (4,820,309 B), 364 records / 4,886 boxes, split 260 / 40 / 64 without exact decoded-image overlap; 4 dataset refusals and 1 over-long-phrase refusal; drawn scene frozen IoUs 0.974 / 0.971 / 0.95; frozen `adapted: False`; 8 epochs, epoch 0 (frozen) validation retained-detection AP50 0.043, best epoch 8 (validation 0.872); held-out test (64 records, 846 reference boxes, VOC matching): retained-detection AP50 empty 0.0 / grid prior 0.015 / frozen 0.072 / adapted 0.891 (per-phrase mean 0.0723 / 0.8913); precision 0.0 / 0.015 / 0.13 / 0.223; recall 0.0 / 0.297 / 0.059 / 0.987; F1 0.0 / 0.028 / 0.081 / 0.364; per-phrase AP frozen → adapted: platelet 0.0 → 0.8805, red blood cell 0.0 → 0.8461, white blood cell 0.2168 → 0.9473; display-only all-candidates AP50 frozen 0.0845 / adapted 0.8929; predicted boxes frozen 386 / adapted 3,744, matched 50 / 835; `adapted_beats_both_baselines` True; `adapter.safetensors` 6,319,224 B, sha256 `21694de50e3f3e5b…` (identical to the 2026-09-25 and 2026-10-08 runs); fresh-base reload parity 8/8; one `UserWarning` (tensor with `requires_grad=True` converted to a scalar) in the adaptation cell. Against the local CPU record of the same code (16 October 2026): frozen 0.0723 and all-candidates frozen 0.0845 equal, adapted 0.8913 vs 0.8909, all-candidates adapted 0.8929 vs 0.8935, epoch 8 equal. Against 2026-10-08 (`cbb9f81`): losses, kept-box counts and adapter digest equal; only matcher-dependent values moved (adapted 0.892 → 0.891, red blood cell 0.8476 → 0.8461, matched 837 → 835, best validation 0.873 → 0.872). Evidence (byte-exact): `docs/execution-evidence/2026-10-17/` — `owlv2_detection_colab_03ac0e7_colab-cli-t4.ipynb` sha256 `0c442ab1d7745f5033707937e740b6dac0b291ff74b6b94038fd45011dd6f4c7`, `owlv2_detection_colab_03ac0e7_exec.log` `850400e3e9901d151205115457887404e020cab0bc263827f229c98939d9f471`, `owlv2_detection_colab_03ac0e7_run_summary.json` `27bd9d635cb159212629611653dddae27d4a8d121dc6955939cfe15e5afbee7b`. One seeded split of one sample, one runtime, no dispersion estimate. Status stays Candidate |
| Review-fix `E2E` carrier of 2026-10-08 (isolated uv environment; review fixes OWD-*; superseded by the OWL-* revision, which ran on 2026-10-17, first row; REL13: earlier-blob figures, scored with the earlier greedy matcher) | `cbb9f81` / `a545a73b7413` (`NOTEBOOK_SOURCE.repository_revision` `80e92cf`, the source revision the notebook was generated from; `80e92cf..cbb9f81` is the review-fix commit itself; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | 2026-10-08 (11:46:45 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-owlv2-cbb9f81-6ce9`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, clean model cache, `USE_BYOD = False`, sample-path defaults; kernel Python 3.13.15; isolated uv environment Python 3.12.12, 49 locked packages, built in 56 s; torch 2.14.0+cu130, transformers 4.57.6, Pillow 11.3.0, pyarrow 25.0.1, scipy 1.18.1; device `cuda:0`, float32 | **PASSED in one pass, no restart** — 598.8 s wall, 12/12 code cells in order, 0 errors (cells 3–5 are the carried modules and print nothing); 9 model files fetched and verified (`model.safetensors` sha256 `e1e130b9…99e7`); BCCD 3 parquet files (4,820,309 B), 364 records / 4,886 boxes, split 260 / 40 / 64 image-disjoint; 4 dataset refusals and 1 over-long-phrase refusal; drawn scene frozen IoUs 0.974 / 0.971 / 0.95; frozen `adapted: False`; 8 epochs, epoch 0 (frozen) validation retained-detection AP50 0.043, best epoch 8 (validation retained-detection AP50 0.873); held-out test (64 records, 846 reference boxes): retained-detection AP50 empty 0.0 / grid prior 0.0147 / frozen 0.0723 / adapted 0.8918; precision 0.0 / 0.015 / 0.13 / 0.224; recall 0.0 / 0.297 / 0.059 / 0.989; per-phrase AP frozen → adapted: platelet 0.0 → 0.8805, red blood cell 0.0 → 0.8476, white blood cell 0.2168 → 0.9473; predicted boxes frozen 386 / adapted 3,744, matched 50 / 837; `adapter.safetensors` 6,319,224 B, sha256 `21694de5…` (identical to the 2026-09-25 run); fresh-base reload parity 8/8; one `UserWarning` (tensor with `requires_grad=True` converted to a scalar) in the adaptation cell. Evidence (byte-exact): `docs/execution-evidence/2026-10-08-cbb9f81/` — executed notebook sha256 `9380527bbe43…`, `exec.log` `2d2f8f40bacf…`, `run_summary.json` `91f537a762e8…`. One seeded split of one sample, one runtime, no dispersion estimate |
| Previous `E2E` carrier (in-kernel install; superseded) | `a772a31` / `c9cd132564ae` (notebook `NOTEBOOK_SOURCE.repository_revision` `49800d8`, the source revision the notebook was bound to; `49800d8..a772a31` changes only the notebook) | 2026-09-25 (22:33–22:44 UTC) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-owlv2-detection` v2), serial suite, `USE_BYOD = False`, sample-path defaults; blob fetched at the 40-char SHA and Git-blob verified; HF cache clean at start | **PASSED only after a manual restart** (not a one-pass Run all; RUN1/RUN10) — 705.1 s wall (pass 1 194.8 s stopped at the install cell with a pip dependency-resolver `CellExecutionError`, kernel restarted after the install cell; pass 2 510.2 s), 11/11 post-restart code cells ok; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins: torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, device `cuda:0`, float32; staged 23 files / 626 MB (9 model files incl. `model.safetensors` 619,918,824 B, sha256 `e1e130b9…99e7`, + 3 BCCD parquet files, all digest-verified); split 260 / 40 / 64, image-disjoint; 4 dataset refusals (duplicate id, box outside image, record without boxes, too small) and 1 over-long-phrase refusal demonstrated; trainable 1,579,526 of 154,966,792 parameters (class + box heads), 8 epochs, best epoch 8 by validation mAP (validation retained-detection AP50 0.8734, n = 40); held-out test (64 records, 846 reference boxes, threshold 0.1, IoU 0.5): retained-detection AP50 empty 0.0 / grid prior 0.0147 / frozen 0.0723 / adapted 0.8918; precision 0.0 / 0.0146 / 0.1295 / 0.2236; recall 0.0 / 0.2967 / 0.0591 / 0.9894; F1 0.0 / 0.0279 / 0.0812 / 0.3647; per-phrase AP frozen → adapted: platelet 0.0 → 0.8805 (50 refs), red blood cell 0.0 → 0.8476 (731), white blood cell 0.2168 → 0.9473 (65); predicted boxes frozen 386 / adapted 3744, matched 50 / 837; `adapter.safetensors` 6,319,224 B sha256 `21694de50e3f3e5b59116cb482126669d527e3c36ee3b3f281046b14d027df7a`, fresh-base reload parity 8/8 identical detections; preserved output sha256: `owlv2_detection_result.json` `3413c0515876…`, `owlv2_detection_evaluation_report.json` `4375f7c4b904…`, adapter `manifest.json` `8cf4dc477d40…`, `owlv2_detection_train.csv` `0557c03e6b02…`; warning: one `UserWarning` (tensor with `requires_grad=True` converted to a scalar) in the adaptation cell. One seeded split of one sample, one runtime, no dispersion estimate |
| Superseded `TASK-INFERENCE` | `6c9365e` / `ce245f5aa2c5` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-owlv2-detection` v1) | **PASSED** — 8/8 code cells, 243.2 s, 20 files, 622 MB staged; historical inference evidence only, not evidence for the E2E blob |
| Superseded `TASK-INFERENCE` local pre-flight | notebook blob `a37382571bee` (commit `2dd34c2`, generated at `5a8fa2f`) | 2026-09-14 | Local Windows fresh-kernel CPU harness | **PASSED** — 8/8 code cells, 68.6 s; pre-flight only |

### Hosted Colab T4 runs of blob `cd3e85a1` (2026-10-29)

These runs executed the PR head `b6c83a26e6cbe71aa02b8dbb84e3f731bdec249b`, notebook `tutorials/owlv2_detection_colab.ipynb` blob `cd3e85a1e129a11888be679d8dde2003246664eb`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 19:03:26 UTC start).** Default settings, no overrides; session `suite-owlv2-b6c83a2-6ce9`; wall 583.7 s (`wall_s`); `code_cells` 12, `executed_cells` 9, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 9, 'revision': 'cfd3195ba4ea9592eec887ded089f4c08eff231d'}
{'grid_baseline': {'map50': 0.015, 'precision': 0.015, 'recall': 0.297, 'f1': 0.028}, 'prior_box_sizes': {'a red blood cell': [68.0, 87.5], 'a white blood cell': [133.0, 159.5], 'a platelet': [25.5, 34.0]}, 'n_predicted_boxes': 17152, 'note': 'grid prior (median training box per phrase tiled over the image)'}
{'frozen_model_test': {'map50': 0.072, 'precision': 0.13, 'recall': 0.059, 'f1': 0.081}, 'adapted': False, 'n': 64, 'reference_boxes': 846, 'predicted_boxes': 386, 'matched': 50, 'verdict': 'measured', 'seconds': 50.5}
{'map50': {'empty': 0.0, 'grid': 0.015, 'frozen': 0.072, 'adapted': 0.891}}
{'precision': {'empty': 0.0, 'grid': 0.015, 'frozen': 0.13, 'adapted': 0.223}}
{'recall': {'empty': 0.0, 'grid': 0.297, 'frozen': 0.059, 'adapted': 0.987}}
{'f1': {'empty': 0.0, 'grid': 0.028, 'frozen': 0.081, 'adapted': 0.364}}
{'delta_vs_frozen': {'map50': 0.819, 'precision': 0.093, 'recall': 0.928, 'f1': 0.283}}
{'all_candidates_ap50_display_only': {'frozen': 0.0845, 'adapted': 0.8929}}
{'reload_parity': {'compared_at': 'label, score to 4 dp, box to 1 dp', 'identical_detections': 8, 'of': 8}, 'reloaded_best_epoch': 8}
```

**DAT20 refusal record (2026-10-29, blob `cd3e85a1`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_PATH` `''` → `'/etc/hostname'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `owlv2_detection-16`, dropping 16 cells (5 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `a12c55ae65796406ca5147b69a5c70860aad87889efbd10d4e09ceed864e2fbc` (Git blob `2799da3a`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: hostname: expected a zip of images plus boxes.csv (or set BYOD_PATH to a folder holding them) ending in .zip. Fix the upload or BYOD_PATH and run this cell again.
```

This run says nothing about the 16 cut cells, which did not run in it. Session `suite-owlv2-b6c83a2-6ce9`, wall 105.1 s. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_default.ipynb` | `f790297f15a6915c15f4231bb28e3d64eb0428a30196ae0cd24525295d6d031f` | `f790297f15a6915c15f4231bb28e3d64eb0428a30196ae0cd24525295d6d031f` (unchanged, no local path) |
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_default_exec.log` | `78f1ba2c580ddf73cf048795f8d1221d949131474d5165e0725e71778cf283ef` | `e67ebdac444d1b1bca4c696e38a178ceacf4786ebd97972850ddd9df968ee749` (2 local-path replacements) |
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_default_run_summary.json` | `03b18b26beaf6241f4459ad938d8b5d7082ded6527a5880614b1814c16d24dda` | `03b18b26beaf6241f4459ad938d8b5d7082ded6527a5880614b1814c16d24dda` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_refusal.ipynb` | `fc4dde2806849134474218db53eeb01ca5339694acf44a4198643a0a87f5d394` | `fc4dde2806849134474218db53eeb01ca5339694acf44a4198643a0a87f5d394` (unchanged, no local path) |
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_refusal_exec.log` | `0889640208c92e0482d3a626e7312c33a94fcae999a6b8a2cd21af25528fc923` | `56715bff4f4f9f1fdc8212c6c3f3cbbcd09c499a43780af78fb51d4cf0767194` (2 local-path replacements) |
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_refusal_run_summary.json` | `17008ff909241fa42212940d994dd63834ae6d3290ce03944739fa15d51bb39a` | `17008ff909241fa42212940d994dd63834ae6d3290ce03944739fa15d51bb39a` (unchanged, no local path) |
| `owlv2_detection_colab_b6c83a2_colab-cli-t4_refusal_run_copy.ipynb` | `a12c55ae65796406ca5147b69a5c70860aad87889efbd10d4e09ceed864e2fbc` | `a12c55ae65796406ca5147b69a5c70860aad87889efbd10d4e09ceed864e2fbc` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12), a browser `Run all` of the interactive journeys (REL2), and every other open gate named in this file; these runs close none of them.

## Current status

**Candidate.** Blob `51db9a5288685` (commit `d7541b2`) passed in one pass on a fresh Colab Tesla T4 on 2026-10-17, Colab account 2 (first row of Recorded executions; every printed metric equals the account-1 run of `3283e836b0ec`). REL13: earlier-blob figures, not re-run on blob `51db9a5288685`, follow. Blob `3283e836b0ec` (commit `03ac0e7`) passed in one pass on a fresh Colab Tesla T4 on 2026-10-17 (first row of Recorded executions; Colab CLI sequential execution, not a browser Run all; forms and uploads not exercised). REL13: earlier-blob figures, not re-run on blob `3283e836b0ec`, follow. Blob `a545a73b` replaced the in-kernel install with an isolated hash-locked uv
environment and carries the 2026-10-02 review fixes (OWD-*) and the 2026-10-16 review fixes (OWL-*: spec 2.5
migration, use notice, DAT9 overlap statement, run labels, one-message BYOD refusals; text and refusal
handling only, default-path computation unchanged). Blob `a545a73b` ran on 2026-10-08 (above); the regenerated
blob `3283e836b0ec` ran on 2026-10-17 (first row), and a
BYOD acceptance run plus one rejected input (REL12) are not yet recorded. For the previous blob
(commit `a772a31` / notebook blob `c9cd132564ae`) the
default sample path executed on Kaggle Tesla T4 after a manual restart after the install cell, with the other gates in the procedure above
satisfied (pinned assets verified, 260 / 40 / 64 image-disjoint split, refusals demonstrated, baselines
and frozen model scored on the same held-out records, heads-only adaptation selected on validation,
adapter export and fresh-base reload parity 8/8). The measured values are one seeded split of one
blood-smear sample on one runtime; they carry no dispersion estimate and are not a benchmark. Any change
to the notebook blob returns the carrier to Candidate until a new exact-blob run is recorded. The
historical inference rows remain regression context only.

Blob `cd3e85a1` (code commit `7544f4f`, the DAT20 refusal fix) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its default path (2026-10-29; 12 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-29; `USE_BYOD=True;BYOD_PATH=/etc/hostname`: exactly one `Refused:` line at cell `owlv2_detection-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`), and the hosted runs of earlier blobs, including `51db9a52`, are history for it (REL14).
