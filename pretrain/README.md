# pretrain/ — the real Columbina training system

This is production code, not a curriculum module: `learning/module_01-33` taught every
individual technique (attention, tokenizer, data pipeline, mixed precision, checkpointing,
SFT masking, DPO) and each notebook stays self-contained on purpose. This package is where
those techniques actually get run for real, over a real multi-month training schedule, so it
needs one shared, editable source of truth instead of N re-pasted copies. See
`C:\Users\alanl\.claude\plans\alright-now-lets-start-purring-volcano.md` for the full plan
this was built from — scope decisions, cost/risk analysis, and the reasoning behind every
number below live there, not repeated here.

**One notebook per phase, not one shared notebook with a mutable stage toggle** — each
`colab/phaseN_*.ipynb` is independently run and tracked, so progress on one phase never gets
confused with another. `colab/` = runs in an actual Google Colab session (GPU or CPU/free-tier
as noted per notebook). `local/` = runs on this machine directly (VS Code, `jupyter notebook`),
never uploaded to Colab, for anything that needs local-only resources (GPU inference of
already-downloaded local model files Colab doesn't have access to).

## Status

**Phase 0 (local infrastructure) — done.** `model.py`, `checkpoint.py`, `tokenizer.py`,
`train.py`, `sft.py` extracted from modules 31/32 and generalized. `scripts/verify_resume_local.py`
proves a killed-and-resumed run is bit-identical to an uninterrupted one across genuinely
separate subprocesses — caught a real bug doing it (LR schedule's `total_steps` must stay
fixed for the whole run; `train.py`'s `--session-step-limit` is the fix). `tests/test_checkpoint_roundtrip.py`
covers `checkpoint.py` in isolation. `sft.py` verified against the real sibling-project data.

**Phase 1 (Colab training notebook) — built, not yet run in real Colab.**
`colab/phase3_pretrain_training.ipynb` (named for the phase it drives, Phase 3's actual
training run — the notebook itself was built during what this repo's history calls "Phase 1")
implements the daily test/train/checkpoint/disconnect rhythm, reusing `verify_resume_local.py`
(now takes `--scratch-dir`) for its TEST gate. Detects Colab vs. local (falls back to local
paths with a clear notice if opened outside Colab, e.g. in VS Code — real training still needs
actual Colab, this machine's GPU doesn't have the VRAM for it at this model size). Dry-run
verified locally end-to-end including a second process correctly resuming a finished run.
`scripts/mirror_checkpoints.ps1` (Drive → `C:/Models`, milestones only) built and its
error-handling verified; needs `rclone config` (Google Drive OAuth) to actually do anything.

**Phase 2 (real data pipeline + local SFT generation) — done and verified with real data,
including both driving notebooks. Ready to actually run.**
- `data_manifest.py` — cross-session ingest progress tracking (phase-sequenced val→test→train,
  all off one continuing HF stream, so there's only ever one `stream_state` to persist, not
  three). Fully unit-tested (`tests/test_data_manifest.py`). Entries carry a per-source
  `text_field` (defaults to `"text"`, overridable — `facebook/empathetic_dialogues` uses
  `"utterance"` instead, caught by a real `KeyError` during end-to-end testing, not guessed in
  advance) and `ingest_source` reads it via `.get()` so an older manifest missing the field
  (e.g. from before this was added) doesn't break on reload.
- `data_pipeline.py` — resumable streaming/tokenizing/shard-writing (`ingest_source`) + final
  shuffle-and-combine into training-ready `.bin` files (`combine_shards_to_corpus`) +
  `load_mix_config`/`init_manifest_from_mix` (reads `configs/data_mix.yaml` directly and
  builds/refreshes manifest entries from it — the "no driver reads the config" gap flagged
  earlier is closed). **Verified against real `HuggingFaceFW/fineweb-edu` data**
  (`scripts/verify_data_pipeline.py`): deliberately interrupted mid-ingest, manifest reloaded
  from disk as a fresh dict (simulating a separate session), resumed correctly through all
  three phases, combined into real decodable corpus files. Uses `datasets` 3.1.0's native
  `IterableDataset.state_dict()`/`load_state_dict()` — verified directly to round-trip
  correctly for FineWeb-Edu and OpenWebText before building anything around it.
- **Real, load-bearing correction to the original plan**: the proposed
  `bigscience/open_subtitles_monolingual` dataset does not exist on HuggingFace, and
  `Helsinki-NLP/open_subtitles`'s loader script is broken (5 configs total, 1 involving
  English). After checking real alternatives directly (not estimating), genuinely clean
  monolingual English "natural dialogue" sources total only **~52.5M tokens** (four sources:
  `deven367/babylm-100M-open-subtitles`, `-switchboard`, `-bnc-spoken`,
  `facebook/empathetic_dialogues` — exact counts in `configs/data_mix.yaml`), nowhere near the
  original 2.5B target. Per the plan's pre-approved contingency, `natural_dialogue`'s target
  shrunk to its real achievable size and `general_english` grew to absorb the difference —
  total pretraining budget stays ~6.5B tokens.
- `generate_sft_data.py` — local generation of new Columbina SFT data via `llama_cpp_python`,
  loading the base `Qwen2.5-7B-Instruct-Q4_K_M` GGUF + the real
  `columbina-qwen25-7b-dpo-v11` LoRA adapter (both already in `C:\Models\`), self-play (same
  model plays both "user" and "Columbina" under different system prompts) seeded from a
  scenario bank covering both the general "teacher-generated" category and all five
  "difficult examples" sub-categories. **Verified with real generated output** — confirmed GPU
  offload (~30 tok/s real decode throughput on the local 4060, matching the plan's estimate),
  confirmed the LoRA-adapted model stays in character (including correctly declining to admit
  being an AI, matching `character.yaml`'s `output_contract`), and confirmed generated
  conversations flow correctly through `sft.py`'s real loader/batcher. Quality filter is a
  lightweight regex identity-leak check, not the sibling project's full 34-category taxonomy
  scoring (that needs a second LLM call per example — real future work, not built here).
- **Both driving notebooks built and verified by actually executing them** (not just the
  underlying `.py` modules in isolation):
  - `colab/phase2_data_prep.ipynb` — reads `configs/data_mix.yaml`, ingests every source in
    order within a configurable session time budget, saves the manifest to Drive after every
    chunk, prints a status table, and runs the final combine once every source is complete or
    exhausted. Executed end-to-end against real FineWeb-Edu/OpenWebText/empathetic_dialogues
    data (small test-scale targets) — all three sources completed, status table correct, real
    `corpus_train.bin`/`corpus_val.bin`/`corpus_test.bin` produced. **CPU-only, no GPU needed**
    — use a free-tier Colab runtime so the A100/L4 budget stays reserved for Phase 3.
  - `local/phase2_generate_sft_data.ipynb` — **runs locally, not in Colab** (needs the local
    GGUF/LoRA files Colab doesn't have access to). Reads target token counts from
    `data_mix.yaml`, estimates conversation counts from a measured
    tokens-per-conversation constant, generates both categories, and reports real (not
    estimated) token yield by actually tokenizing the output. Executed end-to-end for real —
    both categories produced genuine, on-scenario, in-character conversations.
- Operational note found during verification: repeated `load_dataset()` calls in a short
  window hit HuggingFace's anonymous-IP rate limit (HTTP 429). Set an `HF_TOKEN` env var
  before running real ingestion sessions — authenticated requests get a much higher limit.

**Not done yet**: the actual multi-day/week Phase 2 runs themselves (General English to
~6.45B tokens, natural dialogue to ~52.5M, SFT generation to ~30M — both notebooks are ready to
run, just haven't been run to completion), the real ~406M-param training run (Phase 3), SFT
(Phase 4 — its own notebook when that stage's training-loop code exists), DPO (Phase 5,
stretch), local inference script (Phase 6). GitHub remote is set up
(`github.com/YENOSven/gpt-bina`) and pushed to.

## Layout

- `columbina_pretrain/` — the importable package (see docstrings/comments in each file for
  what's genuinely non-obvious; the code itself is the reference for what each does).
- `configs/` — `model_400m.yaml` (not yet wired to code — `train.py`'s CLI args are the real
  interface for now) and `data_mix.yaml` (wired to `data_pipeline.py` via
  `load_mix_config`/`init_manifest_from_mix`, and read directly by `phase2_data_prep.ipynb`
  and `local/phase2_generate_sft_data.ipynb`).
- `colab/phase2_data_prep.ipynb` — Phase 2 data ingestion (CPU-only Colab runtime).
- `colab/phase3_pretrain_training.ipynb` — Phase 3's real GPU training run.
- `local/phase2_generate_sft_data.ipynb` — Phase 2's local Columbina SFT-data generation
  (needs this machine's GPU + local model files, never runs in Colab).
- `scripts/verify_resume_local.py`, `verify_data_pipeline.py` — run these after touching the
  respective module; they're the standing proofs those mechanisms still work.
- `scripts/mirror_checkpoints.ps1` — Drive → `C:/Models` checkpoint mirroring.
- `tests/` — `python -m pytest tests/` from inside `pretrain/`.

## Running things locally

From inside `pretrain/`:
```
python -m pytest tests/
python scripts/verify_resume_local.py
python scripts/verify_data_pipeline.py   # needs network; may hit HF rate limits on repeat runs
```
`generate_sft_data.py` / `local/phase2_generate_sft_data.ipynb` need `llama_cpp_python` with
CUDA support — the sibling Columbina project's venv
(`C:\Users\alanl\OneDrive\Documents\Columbina\columbina_chat\.venv\`) already has a working
build; a matching Jupyter kernel is registered as `columbina-chat-venv`. Point the notebook at
that kernel rather than the main environment's `columbina-py311`, unless `llama_cpp_python`
with CUDA gets installed there too.
