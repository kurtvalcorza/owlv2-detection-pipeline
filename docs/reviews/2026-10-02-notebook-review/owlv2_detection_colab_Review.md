# owlv2_detection_colab.ipynb — Notebook Review (Framework v1)

**Readiness: Needs revision.** The default path runs, and the reviewed notebook blob is the exact blob recorded as
passing on a clean Kaggle T4 run. The adaptation lesson is well bounded. It uses two baselines and the frozen model on
an image-disjoint split, selects the epoch on validation, and verifies reload parity. Three problems hold it back.
Run all needs a manual restart after the install cell (OWD-M1). The adapted heads overwrite the shared `pipe` in
place, so every documented rerun (threshold, seed, epochs, learning rate, BYOD) scores already-tuned heads as the
"frozen model" (OWD-M2). Hard `assert`s stop the notebook before export whenever an honest result does not rank the
models in the expected order (OWD-M3). BYOD also rejects any set smaller than 50 distinct images, although the
notebook says the minimum is 8 (OWD-m1).

The adapted model does **not** "find nothing" at the default threshold, which was the failure in the
conditional-detr/detr siblings. On the recorded T4 run, the adapted heads predict 3,744 boxes for 846 reference
boxes and match 837 of them (recall 0.989, precision 0.224). The frozen model predicts 386 boxes and matches 50.
The open question is over-prediction, which is covered under OWD-S1.

Findings: 0 Blocker, 3 Major, 9 Minor, 4 Suggestion. Prefix `OWD`.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/owlv2-detection-pipeline` |
| Notebook | `tutorials/owlv2_detection_colab.ipynb` (25 cells: 11 code, 14 markdown; no outputs saved) |
| Reviewed revision | `origin/main` = `55a7e2c35a967ea11ffa6a805559616299aaa314` (GitHub API `commits/main`, 2026-10-04) |
| Notebook blob | `c9cd132564ae526f8d0defc5b83e4c49b3d831f2`, identical to the blob recorded at `a772a31` in `docs/release-verification.md` |
| Spec baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main` `b1cfe13`). The notebook declares spec `2.0` in `metadata.dimer` |
| Profile / mode | `E2E` / `GUIDED`, standalone carrier (3 package modules carried verbatim; generator `tools/build_notebook.py` + `tools/notebook_template.py`; `--check` passes) |
| Audience / prerequisites | Basic Python, NumPy and PIL; xyxy boxes, IoU, AP and precision/recall (cell 1) |
| Supported runtime | "Google Colab or Kaggle, Python 3.12; CPU or CUDA", CUDA used automatically (cell 1) |
| Promised outcomes | Pinned install; digest-verified 9-file snapshot (no pickle); digest-pinned BCCD parquet sample split 260/40/64 by image; four dataset refusals; drawn-scene inference contract with an input manifest and an over-long-phrase refusal; empty and grid-prior baselines plus the frozen model on 64 held-out records; bounded head fine-tuning (class + box heads, 1.58M params) with validation-mAP epoch selection; held-out four-way comparison; six panels; drawn scene re-detected; safetensors adapter export with fresh-base reload parity; optional BYOD (zip + `boxes.csv`) and optional experiments |
| Open PRs | None. The E2E carrier PR #7 is merged (it produced this revision) |

### Evidence actually obtained

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear contract and score semantics (uncalibrated sigmoid, caller-owned threshold, no NMS), honest limits. Stale "candidate / not yet recorded" text (OWD-m2). Panel text does not match the panels (OWD-m3). The 2.2 guided layer is partial (OWD-m7) |
| Clean default | Documented execution evidence | Kaggle Tesla T4, 2026-09-25, exact blob `c9cd1325`: PASSED, 705.1 s, 11/11 post-restart code cells, **pass 1 stopped at the install cell and the kernel was restarted** (OWD-M1). Test mAP50: empty 0.0, grid 0.0147, frozen 0.0723, adapted 0.8918. Reload parity 8/8. No Colab record (OWD-m9). Not re-executed in this review |
| Active learning | Direct execution, **reduced scale, CPU** (conda env eo-notebook-test, torch 2.13.0+cpu vs pin 2.14.0, transformers 4.57.6; real pinned snapshot; 8 train / 2 val / 2 test synthetic drawn-shape records with phrases `type alpha`/`type beta`, 3 epochs, lr 1e-4); source inspection | After `adapt`, a Section 6 rerun labels the adapted heads as the frozen model (mAP50 1.0 against the true frozen 0.0), and a Section 7 rerun's epoch 0, noted `"frozen model"`, starts from the tuned heads (OWD-M2). Full-scale experiments **not verified** |
| Reuse and recovery | Direct execution (no model needed for P2); documented evidence for reload and refusals | BYOD zips of 8, 20, 37, 38 and 49 distinct images are rejected in cell 13 with `"N records; 8..5000 are required"`, and the message does not name the split; 50 images are accepted (OWD-m1). The upload widget was not exercised. Four dataset refusals, one phrase refusal and reload parity 8/8 are documented on the T4 run |

Limitations: no GPU, no Colab, no full-scale run, and no BCCD read (the local env has no pyarrow), so P3 used
synthetic stand-in records. The local torch (2.13.0) differs from the pin (2.14.0). The snapshot came from the
local clone's pre-staged `weights/`, which `verify_snapshot` checks. Probe P0 confirmed that the carried cells
equal the modules apart from the documented standalone rewrites (`DEFAULT_WEIGHTS_DIR` and the removed relative
imports).

## 2. Separate judgments

- **Technical correctness:** the default path is sound. Snapshot and corpus digests are checked before use, the
  splits are image-disjoint by decoded-pixel digest, only the heads are trainable, the test split is not used for
  selection, and the artifact manifest is checked before deserialisation. The defects are in-place mutation of
  the shared `pipe` across reruns (OWD-M2), the restart-forcing install (OWD-M1), and an unpinned `pyarrow` that
  the default path imports (OWD-m5).
- **Promise fulfilment:** the default promises are met on the recorded T4 run. These promises fail:
  - "Run all … no configuration edit" holds only after a restart (OWD-M1).
  - "BYOD … through the same … cells" fails for any data that does not reproduce the expected ranking (OWD-M3).
  - "BYOD needs at least eight images" (OWD-m1).
  - "the frozen detections and the adapted detections side by side" (OWD-m3).
  - The optional experiments promise to compare "both the frozen and the adapted model", which OWD-M2 contaminates.
- **Learner experience:** the scientific framing is careful. It covers baselines first, precision and recall read
  together, leakage, and what one seeded split does not establish. The text still describes a release-grade carrier
  as an unrun candidate (OWD-m2). The experiments are one sentence with no rerun steps (OWD-S2). The 4.4×
  over-prediction is left unexplained (OWD-S1).
- **Spec conformance:** these applicable MUSTs are unresolved:
  - RUN1, RUN10 and ENV6 (restart).
  - SRC2 (hidden state on rerun).
  - DAT12 and DAT19 (BYOD limit and message).
  - DAT14 and RUN9 (asserts on the BYOD path).
  - REL12 (BYOD verification).
  - SRC3 (knowingly stale instructions).
  - ENV2 (unpinned pyarrow).

  The 2.2 guided layer (GDL, SHOULD) is partial.

## 3. Findings

### OWD-M1 — Major: Run all needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py:47–70` (`_INSTALL_GUARD`). The PINS come from `pyproject.toml`.
- **Observed issue:** `pip install` of `torch==2.14.0`, `numpy==2.5.3`, `pillow==11.3.0` and the other pins runs
  into the live kernel. If a pre-imported distribution changes, the stale-module guard raises *"Restart the
  runtime, then rerun from the top"*.
- **Consequence:** a learner who presses Run all on Kaggle (and, by the same mechanism, probably on Colab, where
  NumPy and Pillow are preloaded) hits an error at cell 3 and must restart by hand. That breaks the one-pass
  `Run all` contract, which the opening text promises.
- **Evidence:** documented execution evidence. In `docs/release-verification.md` (Recorded executions, row 1),
  pass 1 stopped at the install cell with a pip dependency-resolver `CellExecutionError`, and the kernel was
  restarted after the install cell. The blob is the same as the reviewed one. Source inspection confirms the guard
  in cell 3, and the restart is also named in cell 24 Troubleshooting. Colab itself: **not verified**.
- **Recommended correction:** replace the in-kernel install with the fleet's **uv isolated-environment pattern**.
  A carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked
  `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env,
  so the kernel's preloaded NumPy/torch are never replaced. The reference is
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main.
  Make the change in the generator (`tools/build_notebook.py` install block) and update Section 1 and the
  Troubleshooting text.
- **Acceptance check:** a fresh Kaggle T4 (and Colab T4) Run all of the new blob completes every code cell in one
  pass with no restart and no error output. The record in `docs/release-verification.md` states "no restart".

### OWD-M2 — Major: `adapt()` tunes the shared heads in place, so documented reruns score tuned heads as "frozen"

- **Cell/section:** cells 17, 19 and 21 (Sections 6–8), cell 24 *Optional experiments*, and BYOD in cell 13.
  Source: `src/owlv2_detection_pipeline/pipeline.py:865` copies the best epoch into the live model. The backup
  (`:803`) is restored only on exception (`:869`). The epoch-0 history entry is hard-labelled `"frozen model"` (`:828`).
- **Observed issue:** after Section 7, `pipe` *is* the adapted model. The notebook tells learners to set
  `THRESHOLD` "before Section 6" and compare "both the frozen and the adapted model", change `SPLIT_SEED`, raise
  `EPOCHS`, change `LEARNING_RATE`, or re-run from Section 4 with BYOD. Each of these reruns Section 6 and/or 7 on
  the already-tuned heads. The "frozen model" row and the epoch-0 `frozen model` entry then report the previous
  adaptation, and a second `adapt` continues training from it. The `adapted: True` flag that `evaluate` returns is
  not printed in cell 17.
- **Consequence:** the central comparison (frozen → adapted) of every optional experiment and of BYOD is silently
  wrong. On BYOD, the "frozen" baseline is a BCCD-tuned detector. Learners are likely to conclude that adaptation
  did little because "frozen" already scores high.
- **Evidence:** direct execution, reduced-scale CPU (probe P3). The true frozen validation mAP50 is 0.0 (0 boxes).
  After `adapt` (best epoch 2, heads moved by up to 4.0e-4), a Section 6 rerun reports "frozen" test mAP50 1.0
  (24 boxes, `adapted: true`). A Section 7 rerun's epoch 0, noted `"frozen model"`, reports val mAP50 1.0, equal
  to the first run's selected epoch. In probe P3a (phrases the frozen model already grounds, lr 1e-3, 1 epoch),
  epoch 0 won and no mutation was visible, so the defect appears whenever adaptation helps, which is the default
  case. The full-scale BCCD contamination size is **not measured**.
- **Recommended correction:** keep the frozen base separate from the adapted state. Snapshot the frozen head
  tensors once after load (or reload from the verified snapshot), and have `adapt` start from that snapshot by
  default. Alternatively, store the adapter on a copy, or add `pipe.reset_heads()` and call it at the top of
  Section 6. Print `frozen_test['adapted']` in cell 17 and refuse to label a non-frozen evaluation "frozen". Add
  exact rerun instructions ("run from Section 6 after changing X") to the experiments paragraph in
  `tools/notebook_template.py:510`.
- **Acceptance check:** after a full default run, re-run Sections 6–8 unchanged. Frozen test mAP50 and the
  epoch-0 val mAP50 equal the first run's frozen values (`adapted: False`), and the adapted result reproduces
  within run-to-run tolerance. A test runs `adapt` twice on the same records and asserts that the epoch-0 val
  metrics match.

### OWD-M3 — Major: hard asserts crash honest BYOD and experiment outcomes before export

- **Cell/section:** cell 21 (Section 8): `assert adapted_test['map50'] >= frozen_test['map50']` and
  `assert adapted_test['map50'] > baseline_grid['map50']` (`tools/notebook_template.py:402–403`).
- **Observed issue:** the epoch is selected on validation mAP, which does not guarantee that test mAP is at or
  above frozen. On BYOD data where the frozen OWLv2 already grounds the phrases well, or when the documented
  ×10 learning rate is tried, an honest "adaptation did not help on test" outcome raises `AssertionError`.
  Section 9 (panels, scene re-detection, adapter export, reload parity, `owlv2_detection_result.json`) then never
  runs.
- **Consequence:** BYOD and the optional experiments, which the notebook promises flow through "the same … export
  and reload-parity cells", stop with a bare `AssertionError` and no explanation. The notebook also teaches that
  adaptation must win, which contradicts its own "do not assume" framing (cells 16 and 18).
- **Evidence:** source inspection. Direct execution (P3a) shows that one epoch at lr 1e-3 on stand-in records can
  collapse validation mAP50 from 1.0 to 0.0 (40 boxes). The test-split ordering was not measured at full scale.
  The default BCCD run passes the asserts (0.8918 against 0.0723 and 0.0147, documented).
- **Recommended correction:** turn the comparison into a reported verdict (for example `adapted_beats_frozen`,
  `adapted_beats_grid`) with an explanatory message. Keep hard asserts only for contract invariants (reload
  parity), or gate the ranking asserts on `not USE_BYOD` and on default hyperparameters.
- **Acceptance check:** with a BYOD or stand-in set where adapted test mAP < frozen test mAP, the notebook
  completes Section 9, writes the adapter and result JSON, and prints a verdict that says adaptation did not beat
  the frozen model.

### OWD-m1 — Minor: BYOD promises ≥ 8 images, but fewer than 50 distinct images are rejected without naming the split

- **Cell/section:** cell 0 *Bring Your Own Data* ("at least eight images") and cell 13
  (`validate_dataset(part)` per split, default `min_records=8`). `split_dataset` uses 15 % validation and 20 % test
  (`samples.py:325–326`).
- **Evidence:** direct execution (probe P2). Zips of 8, 20, 37, 38 and 49 distinct images are rejected (for
  example 49 → validation 7: `"7 records; 8..5000 are required"`). 50 is accepted.
- **Consequence / correction:** a learner with 10–49 images follows the stated contract and gets an unexplained
  failure. State the real minimum (50 distinct images at the default fractions), or validate the splits with a
  lower `min_records`. Prefix the message with the split name and the remedy.
- **Acceptance check:** the stated minimum is accepted end to end. One image fewer is rejected with a message that
  names the split, the count, and the needed total.

### OWD-m2 — Minor: stale "candidate / not yet recorded / queued run" text in a Release-grade notebook

- **Cell/section:**
  - Cell 0 (`tools/notebook_template.py:67`, `:88`): "Supported-runtime timing and model-backed results have not
    yet been recorded for this candidate; the queued Kaggle Tesla T4 clean-runtime run is the execution gate". It
    also contains the broken clause "and on the queued clean-runtime run will measure the frozen model".
  - Cell 1 (`:120`).
  - Cell 24 (`:484–486`): "This candidate … this source-only build makes no numerical claim".
- **Consequence:** the README, STATUS and release-verification say Release-grade with a 705 s T4 record, so the
  learner sees conflicting status and gets no runtime expectation (UX12). The text is also a knowingly stale
  instruction (SRC3).
- **Evidence:** source inspection (probe P1).
- **Correction / acceptance check:** state the recorded runtime and timing (as a measured value for the named
  environment) and remove the candidate wording. A grep for `queued`, `not yet been recorded` and `source-only
  build` in the notebook returns nothing. Note that any notebook edit returns the carrier to Candidate until a new
  exact-blob run is recorded, so bundle this with the OWD-M1 fix.

### OWD-m3 — Minor: Section 9 promises frozen-detection panels that are not drawn

- **Cell/section:** cell 22 says "the reference boxes, the frozen detections and the adapted detections side by
  side" (`tools/notebook_template.py:411`). Cell 23 draws two panels, reference and adapted (`:433`), and prints
  `'panels': ['reference boxes', 'adapted detections']`.
- **Evidence:** source inspection (probe P1).
- **Correction / acceptance check:** either add a frozen panel (the frozen detections must then be captured before
  `adapt`, see OWD-M2) or correct the text. The panel count in the text equals the number of panels in each saved
  PNG.

### OWD-m4 — Minor: the BYOD branch is Colab-only, although Kaggle is a stated runtime

- **Cell/section:** cell 13 BYOD branch: `from google.colab import files; files.upload()`. Cell 1 lists "Google
  Colab or Kaggle". README and `load_byod_dataset` also accept a directory, which the notebook path cannot reach.
- **Consequence:** on Kaggle, `USE_BYOD = True` fails with `ModuleNotFoundError: google.colab`.
- **Evidence:** source inspection. Not executed on Kaggle.
- **Correction / acceptance check:** add a `BYOD_PATH = ''  # @param` location field (EXE2). When it is set, read
  that zip or directory without importing `google.colab`. On Kaggle, BYOD with `BYOD_PATH` set to a dataset
  directory reaches Section 9.

### OWD-m5 — Minor: the default path imports `pyarrow`, which is neither pinned nor installed

- **Cell/section:** the carried `samples.read_corpus` imports `pyarrow.parquet` (cell 9; `samples.py:111`). PINS
  in cell 3 omit pyarrow, and `pyproject.toml` lists it only under `dev`.
- **Consequence:** the default sample path depends on whatever pyarrow the hosted image ships. It works on
  Colab and Kaggle today, but fails with `ModuleNotFoundError` in a plain Jupyter env. It is an unpinned principal
  dependency (ENV1/ENV2).
- **Evidence:** source inspection. The local review env without pyarrow could not call `read_corpus`.
- **Correction / acceptance check:** add an exact `pyarrow==…` pin to the runtime dependencies (or to the uv
  lockfile from OWD-M1). The notebook prints its pyarrow version in Section 1.

### OWD-m6 — Minor: Release-grade without REL12 BYOD verification

- **Evidence:** `docs/release-verification.md` records only the `USE_BYOD = False` run. No BYOD acceptance or
  rejection run through Section 9 is recorded (source inspection).
- **Correction / acceptance check:** record one BYOD run (representative zip reaching export and reload) plus one
  rejected incompatible input, with revision and runtime.

### OWD-m7 — Minor: the spec 2.2 guided layer is partial, and the notebook declares spec 2.0

- **Observed issue:** the notebook has no *How to use this notebook* section (GDL2) and no roadmap (GDL3). The
  ~78 KB of carried module cells are not labelled **Infrastructure** or collapsed (GDL11). There are no predictions
  before the comparisons (GDL7), no check-your-reasoning checkpoints (GDL9) and no conclusion template (GDL14).
  Troubleshooting omits the BYOD and memory cases (GDL13). The notebook does have "Look for" and "Read it in this
  order" notes (GDL8, partial).
- **Evidence:** source inspection (probe P1: no "Infrastructure", no "How to use this notebook").
  `metadata.dimer.notebook_spec = '2.0'`.
- **Correction / acceptance check:** add the GDL elements in `tools/notebook_template.py` and declare spec 2.2. A
  review against GDL1–GDL15 finds each element present.

### OWD-m8 — Minor: Section 3 prints a hard-coded weight "source"

- **Cell/section:** cell 11 prints `getattr(pipe, 'source', 'local-snapshot')` (`tools/build_notebook.py:548`).
  `Owlv2DetectionPipeline` has no `source` attribute, so the printed value is always the fallback literal.
- **Evidence:** source inspection, plus direct execution (`hasattr(pipe, 'source')` is `False`).
- **Correction / acceptance check:** print the real `WEIGHTS_DIR` and the verified `model.safetensors` digest
  (`pipe.weight_sha256`) instead.

### OWD-m9 — Minor: no Colab record, and no CPU timing, although both are stated runtimes

- **Observed issue:** the Colab badge is the primary entry point and cell 1 says "CPU or CUDA". The only
  clean-runtime record is Kaggle T4. Nothing tells the learner how long the 364-image 960×960 path takes on CPU.
- **Evidence:** documented execution evidence (release-verification lists Kaggle only).
- **Correction / acceptance check:** record a Colab T4 Run all of the fixed blob. Either state CPU as unsupported
  or slow, or give a labelled estimate (UX12, REL11).

### OWD-S1 — Suggestion: explain the adapted model's 4.4× over-prediction

The recorded adapted run predicts 3,744 boxes for 846 references (precision 0.224, recall 0.989). Without NMS,
duplicates are expected (cell 14 says so). Add a *What to notice* note in Section 8 that ties the box count to
no-NMS duplicates and the threshold, and optionally show the count after a per-phrase NMS as an experiment
(GDL8, UNC).

### OWD-S2 — Suggestion: turn the optional experiments into a Predict → Change → Run → Observe → Explain activity

Name the cell to edit (`THRESHOLD` lives in Section 5, cell 15, not "before Section 6"), give the exact rerun
range, add a reset step (see OWD-M2), and ask for a prediction first (GDL10, UX5).

### OWD-S3 — Suggestion: add paired uncertainty for adapted against frozen

Add a bootstrap over the 64 test records (per-record resampling of mAP50) to put an interval on the deltas, in
line with the notebook's own "no dispersion estimate" caveat (EVAL).

### OWD-S4 — Suggestion: drop or align the unused `torchaudio==2.11.0` pin

The notebook never imports torchaudio, and its minor version differs from `torch==2.14.0`. PyPI metadata shows
no torch requirement for torchaudio 2.11.0 (probe P4), so it is dead install weight with an ABI mismatch.

## 4. Readiness

**Needs revision.** Three Majors are open (OWD-M1 restart, OWD-M2 in-place adaptation contaminating reruns and
BYOD, OWD-M3 ranking asserts), together with these unresolved applicable MUSTs: RUN1/RUN10/ENV6,
SRC2, DAT12/DAT14/DAT19, RUN9, SRC3, ENV2 and REL12. Required execution evidence exists for the default path on
Kaggle T4. Remaining gates after the fixes:
1. A new exact-blob Run all on Kaggle T4 and on Colab T4 with no restart.
2. A BYOD acceptance run and a rejection run.
3. A rerun of Sections 6–8 that reproduces the frozen values.

## 5. Verified versus inferred

- **Verified by direct execution (reduced-scale CPU, labelled):**
  - In-place head mutation contaminates "frozen" reruns (P3).
  - The BYOD minimum is 50 distinct images (P2).
  - The carried-cell parity and the generator `--check` pass (P0, P6).
  - `pipe.source` is absent.
- **Verified from documented evidence:**
  - The restart on Kaggle T4.
  - The default metrics, box counts and reload parity on the exact reviewed blob.
- **Inferred, not executed:**
  - The Colab restart.
  - That the BYOD `google.colab` import fails on Kaggle.
  - That the asserts fire on real BYOD data.
  - That the full-scale size of the OWD-M2 contamination is material.
- **Most likely to be wrong:** the severity of OWD-M3. The mechanism is certain, but on in-domain BYOD data the
  adapted heads usually beat the frozen model, so in practice the asserts may rarely fire.

Probe files: `owlv2_detection_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
