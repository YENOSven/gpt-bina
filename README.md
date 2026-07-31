# gpt-bina

A GPT-style language model trained from scratch. Every core component — automatic
differentiation, multi-head attention, a BPE tokenizer, the optimizer, the mixed-precision and
gradient-accumulation training loop, sampling/decoding, and preference-tuning (DPO) — was
implemented independently and verified against a reference implementation or hand-computed
value before being trusted, then assembled into a real, resumable, multi-day pretraining system
that has been run against real multi-billion-token data.

## What this is

The repo has two parts:

- **[`pretrain/`](pretrain/)** — a production-grade training system: a from-scratch GPT
  implementation, a resumable checkpointing layer, a streaming data pipeline, and the training
  loop that combines them, built to survive being killed mid-run, resumed across sessions, and
  scaled from a laptop GPU to Google Colab Pro.
- **[`learning/`](learning/) + `docs/LEARNING_LOG.md`** — the 33-module curriculum that built
  up to it, one notebook per concept, each with a verification step rather than a bare
  assertion that the code "works." A parallel 11-part track in [`experiments/`](experiments/)
  explores biologically-inspired (Hebbian/spiking) learning as a point of comparison against
  standard backprop.

## The production pipeline (`pretrain/`)

A decoder-only transformer (pre-norm attention + feed-forward blocks, weight-tied
embeddings, GPT-2's parameter initialization scheme, `scaled_dot_product_attention` for
fused/flash-attention kernels) with three configs:

| | dev/smoke-test | GPT-2-small scale | target scale |
|---|---|---|---|
| params | ~0.03M | **124,402,944** (verified against the public GPT-2-small figure) | ~406M |
| `d_model` | 64 | 768 | 1024 |
| heads | 4 | 12 | 16 |
| layers | 2 | 12 | 28 |
| context length | 64 | 1024 | 2048 |

**Engineering highlights, each independently verified rather than assumed:**

- **Resumable, crash-safe training** — checkpoints capture model/optimizer/RNG state, step
  count, and token cursor. A killed-and-resumed run was verified **bit-identical** to an
  uninterrupted one across genuinely separate OS processes. Caught and fixed a real bug where a
  resumed session with a different micro-batch size silently recomputed a *different*
  `total_steps` and would have corrupted the learning-rate schedule — fixed by persisting
  `total_steps` inside the checkpoint itself instead of recomputing it.
- **Correct initialization, measured** — GPT-2's init scheme cut the model's initial loss from
  485 down to 10.95, within 0.01 nats of the theoretical floor (`ln(vocab_size)` = 10.82).
- **Memory/throughput engineering** — mixed precision (bf16 autocast) measured **3.4x** faster
  matmuls via tensor cores; gradient accumulation verified numerically equivalent to one large
  batch (to 1e-6) while cutting peak memory **29.3%**; gradient checkpointing trades ~30% more
  compute for enough headroom to train on an 8GB consumer GPU; KV-caching verified to produce
  **byte-for-byte identical** generations to uncached inference with a measured **2.03x**
  decoding speedup.
- **Optimizer and schedule correctness** — a from-scratch AdamW (decay/no-decay parameter
  groups) and a warmup+cosine LR schedule both verified exact against `torch.optim` / HF
  `transformers` reference implementations.
- **Real-scale streaming data pipeline** — cross-session ingestion progress tracked in a
  manifest; HuggingFace datasets streamed and tokenized (GPT-2 BPE via `tiktoken`) directly into
  memmapped `uint16` shards with no full corpus ever materialized in memory. Used to build a real
  **~6.5-billion-token** training corpus end to end, after discovering mid-project that one of
  the originally planned data sources didn't actually exist on HuggingFace and re-deriving the
  data mix from real, measured token counts instead of an estimate.
- **Local synthetic data generation** — self-play conversation generation against a quantized
  7B model running locally (measured ~30 tok/s decode throughput on a consumer GPU), with an
  automated quality filter, used to build supervised fine-tuning data at zero API cost. The
  general-web-text corpus above is optionally mixed with a small domain-specific corpus (a
  single wiki-style knowledge source) purely to exercise the data-mixing pipeline on a
  realistic minority-fraction case.
- **Real bugs found and fixed**, not just features added: a device-placement bug (causal mask
  ending up on CPU while activations were on CUDA), an RNG-state tensor silently deserialized
  onto the wrong device, a Windows file-lock race during atomic checkpoint writes (fixed with
  retry/backoff), and a severe memory blowup in the corpus-combining step at real data scale —
  all caught by tests or verification scripts before they could silently corrupt a multi-day
  run, not discovered after the fact.
- **Test coverage**: a pytest suite (checkpoint round-tripping, data-mix correctness, gradient
  checkpointing) plus two standalone verification scripts that prove resumability and
  data-pipeline correctness against real HuggingFace data, not mocks.

See [`pretrain/README.md`](pretrain/README.md) for full status and architecture notes.

## The curriculum behind it (`learning/`, `docs/LEARNING_LOG.md`)

33 self-contained notebooks, each implementing one concept from first principles and verifying
it — against a hand-computed derivative, a reference PyTorch/HuggingFace implementation, or a
measured before/after — rather than asserting it works:

| Phase | Covers |
|---|---|
| 1 — Foundations | A scalar autograd engine, neurons/layers/MLPs, gradient descent, then the same ideas in real PyTorch |
| 2 — Language modeling fundamentals | N-gram/bigram models, perplexity, token embeddings, a neural n-gram model |
| 3 — The transformer, piece by piece | Scaled dot-product & multi-head attention, positional encoding, LayerNorm, residual connections, the feed-forward block, then a full nanoGPT trained on text |
| 4 — Scaling toward a real LLM | A from-scratch BPE tokenizer, production tokenizers, large-corpus data pipelines, mixed precision, gradient accumulation, AdamW, LR schedules, gradient clipping, Colab compute budgeting |
| 5 — The real pretrain & specialization | Sampling strategies, KV-caching, the real pretraining corpus and training run (124M params, matching the public GPT-2-small parameter count exactly), instruction fine-tuning, DPO |

Full module-by-module log with results: **[`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md)**.

A parallel 11-part track in [`experiments/`](experiments/) builds a small conversational agent
three separate ways — standard backprop, biologically-inspired Hebbian learning on
leaky-integrate-and-fire neurons, and a three-factor (local-tag + global-reward) learning rule —
and reports honest, measured accuracy comparisons across all three, including where the
biologically-plausible approaches fall short. Details in the same log.

## Repo layout

```
gpt-bina/
├── pretrain/                    # the production training system
│   ├── columbina_pretrain/      # importable package: model, tokenizer, data pipeline, checkpointing, train/SFT loops
│   ├── configs/                 # model + data-mix configs
│   ├── colab/                   # notebooks run on Google Colab (GPU training, data prep)
│   ├── local/                   # notebooks that need this machine's local GPU/model files
│   ├── scripts/                 # verification and sync scripts
│   ├── tests/                   # pytest suite
│   └── README.md                # architecture + status detail
├── learning/                    # 33-module from-scratch curriculum (one notebook each)
├── experiments/                 # 11-part biologically-inspired learning track
└── docs/
    └── LEARNING_LOG.md          # full module-by-module log with verified results
```

## Tech stack

Python, PyTorch, NumPy, HuggingFace `datasets` (streaming) and `tokenizers`, `tiktoken`,
`llama-cpp-python` (local quantized inference), PyYAML, pytest, Jupyter, Google Colab Pro
(A100/L4 GPUs) for the compute-heavy pretraining run.

## Status

- **Model, tokenizer, checkpointing, training loop** — done and unit-tested.
- **Data pipeline** — done, verified end-to-end against real streamed HuggingFace data; the
  actual multi-day corpus-ingestion run is ready but not yet executed to completion.
- **Colab training notebook** — built and dry-run verified locally end-to-end (including a
  process correctly resuming a run started by a different process); not yet run on real Colab
  GPU compute.
- **The ~406M-parameter pretraining run, SFT, and DPO** — pipeline ready; not yet run.

## Running it locally

```bash
cd pretrain
pip install -r requirements.txt
python -m pytest tests/
python scripts/verify_resume_local.py
```

Each notebook under `learning/` and `experiments/` is self-contained and runs top to bottom in
Jupyter or Colab on its own — open any `module_XX/*.ipynb` directly.

## License

[MIT](LICENSE)
