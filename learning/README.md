# Learning: a 33-module deep dive into how LLMs actually work

Before attempting a real pretraining run in [`pretrain/`](../pretrain/), I wanted to actually
*understand* every piece of the system I was about to build, not just call
`AutoModelForCausalLM.from_pretrained(...)` and move on. So I worked backward from a working LLM
to first principles: 33 small, self-contained Jupyter notebooks, each isolating exactly one
concept (autograd, attention, a tokenizer, an optimizer, a training loop) and *proving* it works,
whether against a hand-computed value, a reference PyTorch/HuggingFace implementation, or a
measured before/after, rather than just writing code that runs without erroring. The goal was
depth, not speed. Understand why each piece exists and what breaks without it, then earn the
right to use the library version.

This file is a condensed summary of that curriculum. The original notebooks aren't kept in this
repo (see below); the outcome that mattered was the understanding that fed into
[`pretrain/`](../pretrain/), the actual production training system.

## Phase 1: Foundations of learning
Built a scalar automatic-differentiation engine from nothing (backward rules for `+`, `*`,
`**`, `tanh`, `relu`, verified against hand-computed derivatives), composed it into neurons,
layers, and MLPs, and trained one on a toy classification task with hand-rolled gradient
descent. Then redid the same problem in real PyTorch (`nn.Module`, `torch.optim.SGD`) to connect
the from-scratch version to the library abstractions it corresponds to.

## Phase 2: Language modeling fundamentals
Started as simple as language modeling gets: a counting-based bigram model, then perplexity as
the standard evaluation metric (and why smoothing matters: unsmoothed test perplexity is
infinite the first time an unseen bigram shows up). Then token embeddings (proving
`nn.Embedding` is just fast one-hot indexing) and a small neural n-gram model in the style of
Bengio et al. 2003, which, on a deliberately tiny dataset, overfits past a point. An early,
honest lesson about needing enough data before more context helps.

## Phase 3: The transformer, piece by piece
The core of the curriculum. Built and independently verified every component of a transformer
block before assembling them:
- Scaled dot-product attention and multi-head attention (verified a naive per-head Python loop
  matches the vectorized batched version exactly; causal masking checked directly on real
  attention weights).
- Positional encoding (first proved attention is permutation-*equivariant*, meaning shuffling
  tokens just shuffles the output, then showed adding positional encoding breaks that on
  purpose).
- LayerNorm (verified exact against `nn.LayerNorm`; without it, activations in a deep stack
  compound from std 2.8 to 522 across 8 layers).
- Residual connections (a 30-layer stack's input gradient goes from ~1.5e-9, vanished, to ~19
  once residuals are added).
- The feed-forward block, then a full pre-norm transformer block, then stacking blocks into a
  nanoGPT (token and positional embedding, N blocks, final norm, weight-tied output head).
- Finally, trained that nanoGPT on real text and watched it first generalize (perplexity 12.1
  vs. a 55 uniform baseline) and then, with more training, memorize the training text verbatim.
  A direct, hands-on demonstration of *why* scale and real corpora matter, motivating the next
  two phases.

## Phase 4: Scaling toward a real LLM
Moved from toy-scale to the actual infrastructure a real pretraining run needs: a from-scratch
byte-pair-encoding tokenizer, then compared against production tokenizers (`tiktoken`/GPT-2 and
HuggingFace `tokenizers`); efficient large-corpus data pipelines (naive per-document padding
wastes 56.7% of tokens, while concatenate-and-chunk wastes none, verified for correctness); mixed
precision training (bf16 measured ~3.4x faster than fp32 via tensor cores); gradient
accumulation (proved numerically equivalent to one large batch, to 1e-6, while cutting peak
memory 29.3%); a from-scratch AdamW verified exact against `torch.optim`; and warmup plus cosine
learning-rate schedules verified exact against HuggingFace's implementation, with a demonstration
of *why* warmup matters (an aggressive LR without it spikes loss to 106,965 in the first 30
steps). Closed with gradient clipping and the operational reality of training on a compute
budget (Google Colab Pro): checkpointing weights, optimizer state, RNG state, and step, verified
to resume bit-identically.

## Phase 5: The real pretrain
Sampling strategies (greedy/temperature/top-k/top-p, each verified against its own definition;
for example, top-p's kept token set always meets its cumulative-probability threshold) and
KV-caching (verified byte-for-byte identical generations with vs. without caching, at a measured
2.03x speedup). Then the parts that turn all of the above into an actual pretraining run: a real
data pipeline mixing streamed web-scale text with a smaller domain-specific corpus; a real
training run at GPT-2-small scale (124,402,944 parameters, matching the public figure exactly)
that combines every technique from Phase 4 into one loop, including catching a real bug where
the wrong weight-initialization scheme left initial loss at 485 instead of the theoretical floor
of 10.82; instruction fine-tuning with a verified loss mask; and, as a stretch module, Direct
Preference Optimization (DPO) implemented from its loss function, with the frozen reference
model verified unchanged throughout training.

## Where this led

Everything above is deliberately toy-scale: small models, small corpora, built to isolate and
verify one concept at a time as cheaply as possible. **[`pretrain/`](../pretrain/)** is where
those same techniques get run for real. A from-scratch GPT trained at up to ~406M parameters on
a real multi-billion-token corpus, with production concerns (crash-safe resumable checkpointing,
a streaming data pipeline, Colab-scale compute budgeting) that the curriculum above deliberately
stayed out of scope of. See that folder's README for the actual production system and its
verified results.

*Why no notebooks in this repo:* the 33 notebooks themselves were working scratch material for
building understanding, not a maintained artifact, so they aren't kept here. What's preserved is
the production system they led to, in `pretrain/`.
