# pretrain/ — the real Columbina training system

This is production code, not a curriculum module: `learning/module_01-33` taught every
individual technique (attention, tokenizer, data pipeline, mixed precision, checkpointing,
SFT masking, DPO) and each notebook stays self-contained on purpose. This package is where
those techniques actually get run for real, over a real multi-month training schedule, so it
needs one shared, editable source of truth instead of N re-pasted copies. See
`C:\Users\alanl\.claude\plans\alright-now-lets-start-purring-volcano.md` for the full plan
this was built from — scope decisions, cost/risk analysis, and the reasoning behind every
number below live there, not repeated here.

## Status

**Phase 0 (local infrastructure) — done.** `columbina_pretrain/model.py`, `checkpoint.py`,
`tokenizer.py`, `train.py`, and `sft.py` are real importable modules, extracted from modules
31/32 and generalized. The two things Phase 0 exists to prove are both proven:

- `scripts/verify_resume_local.py` — a training run killed partway through and resumed from
  its checkpoint produces **bit-identical** losses and parameters to an uninterrupted run
  (verified across genuinely separate subprocesses, not just in-process object reuse). Caught
  and fixed a real bug in the process: the LR schedule's cosine decay must be computed against
  the *whole run's* `total_steps`, not the current session's — an earlier version silently
  passed a smaller `total_steps` to simulate "stopping early" and it corrupted the schedule
  shape instead. `train.py`'s `--session-step-limit` is the fix: `total_steps` stays fixed for
  the life of the run, only `--session-step-limit` differs per daily session.
- `tests/test_checkpoint_roundtrip.py` — fast in-process unit tests of `checkpoint.py` alone
  (model/optimizer state, RNG state, atomic-write cleanup). Run via `python -m pytest tests/`.
- `sft.py` verified against the real sibling-project data (2,232 real examples from
  `columbina_chat/data/canon/qwen25_v10_sft/train.jsonl`): the loss-mask span decodes back to
  exactly the assistant turn's text, and a real batch flows through the model and backward
  pass with no errors. Example lengths in that dataset run 503-632 tokens (character.yaml's
  system prompt included) — comfortably inside even the 1024-token `124m` config, let alone
  the real 2048-token `406m` target.

**Not done yet — everything past Phase 0.** No GitHub remote exists (needs the user's account
— `gh` CLI isn't installed locally and OAuth login needs a browser). No Colab notebook exists
yet. No real corpus has been downloaded/tokenized (`data_pipeline.py` doesn't exist yet —
`configs/data_mix.yaml` documents the target mix, nothing reads it yet). No local SFT-data
generation has run. The `406m` model config has never actually been trained, even briefly —
everything proven so far used the `tiny`/`124m` configs for speed.

## Layout

- `columbina_pretrain/` — the importable package (see docstrings/comments in each file for
  what's genuinely non-obvious; the code itself is the reference for what each does).
- `configs/` — `model_400m.yaml` and `data_mix.yaml` document targets; **neither is wired up
  to code yet** (train.py's CLI args are the real interface for now).
- `colab/` — empty, Phase 1.
- `scripts/verify_resume_local.py` — run this after touching `checkpoint.py` or `train.py`'s
  training loop; it's the standing proof that resume still works.
- `tests/` — `python -m pytest tests/` from inside `pretrain/`.

## Running things locally

From inside `pretrain/`:
```
python -m pytest tests/
python scripts/verify_resume_local.py
```
Both are CPU-only and take a few seconds — no GPU needed for anything in Phase 0.
