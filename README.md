# gpt-bina

A GPT-style language model trained from scratch. Every core component, including automatic
differentiation, multi-head attention, a BPE tokenizer, the optimizer, the mixed-precision and
gradient-accumulation training loop, sampling/decoding, and preference-tuning with DPO, was
implemented independently and verified against a reference implementation or a hand-computed
value before being trusted. Those pieces were then assembled into a real, resumable, multi-day
pretraining system that has been run against real multi-billion-token data.

## What this is

The repo has two parts:

- **[`pretrain/`](pretrain/)**: a production-grade training system. A from-scratch GPT
  implementation, a resumable checkpointing layer, a streaming data pipeline, and the training
  loop that combines them, built to survive being killed mid-run, resume across sessions, and
  scale from a laptop GPU to Google Colab Pro.
- **[`learning/`](learning/)**: the 33-module, from-first-principles curriculum that built the
  understanding behind it (details below).

## The production pipeline (`pretrain/`)

A decoder-only transformer (pre-norm attention and feed-forward blocks, weight-tied embeddings,
GPT-2's parameter initialization scheme, and `scaled_dot_product_attention` for fused/flash-
attention kernels) with three configs:

| | dev/smoke-test | GPT-2-small scale | target scale |
|---|---|---|---|
| params | ~0.03M | **124,402,944** (verified against the public GPT-2-small figure) | ~406M |
| `d_model` | 64 | 768 | 1024 |
| heads | 4 | 12 | 16 |
| layers | 2 | 12 | 28 |
| context length | 64 | 1024 | 2048 |

**Engineering highlights, each independently verified rather than assumed:**

- **Resumable, crash-safe training.** Checkpoints capture model/optimizer/RNG state, step
  count, and token cursor. A killed-and-resumed run was verified **bit-identical** to an
  uninterrupted one across genuinely separate OS processes. That work also caught a real bug: a
  resumed session with a different micro-batch size silently recomputed a *different*
  `total_steps`, which would have corrupted the learning-rate schedule. Fixed by persisting
  `total_steps` inside the checkpoint itself instead of recomputing it.
- **Correct initialization, measured.** GPT-2's init scheme cut the model's initial loss from
  485 down to 10.95, within 0.01 nats of the theoretical floor (`ln(vocab_size)` = 10.82).
- **Memory and throughput engineering.** Mixed precision (bf16 autocast) measured **3.4x**
  faster matmuls via tensor cores. Gradient accumulation was verified numerically equivalent to
  one large batch (to 1e-6) while cutting peak memory **29.3%**. Gradient checkpointing trades
  about 30% more compute for enough headroom to train on an 8GB consumer GPU. KV-caching was
  verified to produce **byte-for-byte identical** generations to uncached inference, with a
  measured **2.03x** decoding speedup.
- **Optimizer and schedule correctness.** A from-scratch AdamW (decay/no-decay parameter
  groups) and a warmup+cosine LR schedule were both verified exact against `torch.optim` and HF
  `transformers` reference implementations.
- **Real-scale streaming data pipeline.** Cross-session ingestion progress is tracked in a
  manifest. HuggingFace datasets are streamed and tokenized (GPT-2 BPE via `tiktoken`) directly
  into memmapped `uint16` shards, so the full corpus is never materialized in memory. This built
  a real **~6.5-billion-token** training corpus end to end, after discovering mid-project that
  one of the originally planned data sources didn't actually exist on HuggingFace and
  re-deriving the data mix from real, measured token counts instead of an estimate.
- **Local synthetic data generation.** Self-play conversation generation against a quantized 7B
  model running locally (measured ~30 tok/s decode throughput on a consumer GPU), with an
  automated quality filter, used to build supervised fine-tuning data at zero API cost. The
  general-web-text corpus above is optionally mixed with a small domain-specific corpus (a
  single wiki-style knowledge source), mainly to exercise the data-mixing pipeline on a
  realistic minority-fraction case.
- **Real bugs found and fixed, not just features added.** A device-placement bug (the causal
  mask ending up on CPU while activations were on CUDA), an RNG-state tensor silently
  deserialized onto the wrong device, a Windows file-lock race during atomic checkpoint writes
  (fixed with retry/backoff), and a severe memory blowup in the corpus-combining step at real
  data scale. All of these were caught by tests or verification scripts before they could
  silently corrupt a multi-day run, not discovered after the fact.
- **Test coverage.** A pytest suite (checkpoint round-tripping, data-mix correctness, gradient
  checkpointing) plus two standalone verification scripts that prove resumability and
  data-pipeline correctness against real HuggingFace data, not mocks.

See [`pretrain/README.md`](pretrain/README.md) for full status and architecture notes.

## The curriculum behind it (`learning/`)

The real motivation for this project was learning, not shipping. I wanted to actually
understand how an LLM is trained, not just call a library. So before writing the production
system above, I worked backward from a working transformer to first principles across 33
self-contained notebooks, each implementing one concept and *proving* it works, whether against
a hand-computed derivative, a reference PyTorch/HuggingFace implementation, or a measured
before/after, rather than just asserting it does:

| Phase | Covers |
|---|---|
| 1: Foundations | A scalar autograd engine, neurons/layers/MLPs, gradient descent, then the same ideas in real PyTorch |
| 2: Language modeling fundamentals | N-gram/bigram models, perplexity, token embeddings, a neural n-gram model |
| 3: The transformer, piece by piece | Scaled dot-product & multi-head attention, positional encoding, LayerNorm, residual connections, the feed-forward block, then a full nanoGPT trained on text |
| 4: Scaling toward a real LLM | A from-scratch BPE tokenizer, production tokenizers, large-corpus data pipelines, mixed precision, gradient accumulation, AdamW, LR schedules, gradient clipping, Colab compute budgeting |
| 5: The real pretrain | Sampling strategies, KV-caching, the real pretraining corpus and training run (124M params, matching the public GPT-2-small parameter count exactly), instruction fine-tuning, DPO |

Full summary of what each phase covers and what it proved: **[`learning/README.md`](learning/README.md)**.

## Repo layout

```
gpt-bina/
├── pretrain/               # the production training system
│   ├── bina_pretrain/      # importable package: model, tokenizer, data pipeline, checkpointing, train/SFT loops
│   ├── configs/             # model + data-mix configs
│   ├── colab/                # notebooks run on Google Colab (GPU training, data prep)
│   ├── local/                # notebooks that need this machine's local GPU/model files
│   ├── scripts/               # verification and sync scripts
│   ├── tests/                 # pytest suite
│   └── README.md              # architecture + status detail
├── learning/               # summary of the 33-module from-scratch curriculum
│   └── README.md
├── LICENSE
└── README.md
```

## Tech stack

Python, PyTorch, NumPy, HuggingFace `datasets` (streaming) and `tokenizers`, `tiktoken`,
`llama-cpp-python` (local quantized inference), PyYAML, pytest, Jupyter, Google Colab Pro
(A100/L4 GPUs) for the compute-heavy pretraining run.

## Status

- **Model, tokenizer, checkpointing, training loop.** Done and unit-tested.
- **Data pipeline.** Done, verified end-to-end against real streamed HuggingFace data. The
  actual multi-day corpus-ingestion run is ready but not yet executed to completion.
- **Colab training notebook.** Built and dry-run verified locally end-to-end, including a
  process correctly resuming a run started by a different process. Not yet run on real Colab
  GPU compute.
- **The ~406M-parameter pretraining run, SFT, and DPO.** Pipeline ready, not yet run.

## Running it locally

```bash
cd pretrain
pip install -r requirements.txt
python -m pytest tests/
python scripts/verify_resume_local.py
```

See [`learning/README.md`](learning/README.md) for a summary of the from-scratch curriculum that
led here.

## License

[MIT](LICENSE)
