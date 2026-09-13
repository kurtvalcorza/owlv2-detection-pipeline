# Weight provenance and DIMER hosting

- Upstream: `google/owlv2-base-patch16-ensemble`
- Immutable revision: `cfd3195ba4ea9592eec887ded089f4c08eff231d` (the Hub's `main` resolved to this commit on 2026-09-14)
- Weight format: SafeTensors (`model.safetensors`, 619,918,824 bytes, float32). The upstream repository also hosts `pytorch_model.bin` (620,006,741 bytes), a pickle checkpoint that DIMER does not accept and this pipeline neither lists nor loads.
- Manifest: `weights/owlv2-base-patch16-ensemble/dimer-base-manifest.json` (9 files: `README.md`, `added_tokens.json`, `config.json`, `merges.txt`, `model.safetensors`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`; 621,510,370 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter and the Hub's licence tag)
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes` so Windows `core.autocrlf` cannot rewrite the committed small files and break their digests.
- Loader trust boundary: Transformers `Owlv2ForObjectDetection` / `Owlv2Processor` (the slow `Owlv2ImageProcessor` the snapshot declares plus the CLIP BPE tokenizer with `model_max_length` 16) with `trust_remote_code=False`, `local_files_only=True` from the verified directory; nothing is fetched at construction or inference (the smoke run loaded and detected with `HF_HUB_OFFLINE=1`).
