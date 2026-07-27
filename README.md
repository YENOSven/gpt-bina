# Columbina LLM Training — Learning Roadmap

Goal: understand every building block needed to train an LLM from scratch, then
build one ourselves — ending with a model specialized in natural conversation,
basic real-world knowledge, and Genshin Impact lore/characters. Small/
foundational stuff runs locally; anything GPU-hungry runs on Google Colab.

**Scope decision (2026-07-23):** Phase 4/5 will pretrain on **Google Colab
Pro** ($9.99/mo, ~100 compute units included, ~$0.10/unit top-ups; an A100
burns ~15 units/hr ≈ $1.50/hr). Starting target: **~125M params on ~2.5B
tokens** (GPT-2-small scale) — this fits inside a single month's included
compute units (~5 A100-hours, effectively free beyond the subscription) and
gives a real working checkpoint end to end. Whether to scale up from there
(350M-1B+ params, needing top-up purchases from ~$60 to ~$1500) is a
decision to make *after* Phase 4's infra actually works, not before —
don't over-build for a bigger target prematurely. Corpus: monolingual
English text, mixing general text (real-world knowledge) with Genshin
Impact wiki/lore text (specialization) — see Module 30. Phases 1-3 below
are unaffected and run locally or on free-tier Colab as before.

**Format:** one notebook = one component. Each module is scoped to a single
concept or architectural piece, not a bundle of them — better to have many
small, focused notebooks than a few that try to teach several things at once.
Each notebook is self-contained and runs top to bottom in Jupyter or Colab; if
it needs a class from an earlier module (e.g. the `Value` autograd engine),
that code is copied in with a one-line note on where it's from, rather than
re-explained. This list will keep growing/splitting further as we go — treat
the numbering as provisional, not fixed.

## Phase 1 — Foundations of Learning
- [x] **01 — Autograd engine**: a scalar computation graph (the `Value`
  class) with backward rules for `+`, `*`, `**`, `tanh`, `relu`, verified
  against a hand-computed derivative. Just automatic differentiation, nothing
  else yet.
- [x] **02 — Neurons, layers, MLPs**: compose `Value`s into the shapes we
  call a neural network — `Neuron` → `Layer` → `MLP`. Forward pass only, no
  training. Purely about structure.
- [x] **03 — Training a network**: loss functions + gradient descent. Takes
  the MLP from module 02, teaches it a toy classification task, and
  visualizes what it learned.
- [x] **04 — PyTorch tensors & autograd**: the same ideas as Module 01, but
  with real `torch.Tensor` and `.backward()` — vectorized, GPU-ready.
  Verifies scalar and vectorized gradients against hand-computed
  derivatives, and demonstrates gradient accumulation + running on CUDA.
- [x] **05 — PyTorch `nn.Module` & optimizers**: redoes Module 03's exact
  toy 2-blob task with `nn.Module`/`nn.Sequential` structure and
  `torch.optim.SGD` (with `weight_decay` for L2 reg) instead of hand-rolled
  classes and manual gradient descent — reaches the same 100% accuracy.

## Phase 2 — Language Modeling Fundamentals
- [x] **06 — Counting-based bigram model**: character-level bigram counts
  over a list of Genshin character names, normalized into probabilities
  (Laplace-smoothed), samples name-like strings, scored via average
  negative log-likelihood (2.52) against a uniform-random baseline (3.26).
- [x] **07 — Perplexity**: `exp(NLL)` on Module 06's bigram model, evaluated
  properly on a held-out test split (train ppl 12.74 vs test ppl 15.91),
  plus a demo of why smoothing matters (unsmoothed test perplexity = inf).
- [x] **08 — Token embeddings**: one-hot vs. learned dense vectors, proves
  `one_hot @ E == E[idx]` (why `nn.Embedding` is just fast indexing), trains
  a 2D embedding on next-char prediction and shows vowels cluster together
  (avg pairwise distance 0.40 vs. 3.50 for all characters).
- [x] **09 — Neural n-gram model**: Bengio et al. 2003-style — embed a
  3-character context, concatenate, MLP (1,866 params vs. the 17,576-row
  table counting would need). Reuses Module 07's perplexity eval; on this
  tiny dataset it overfits past ~step 50 (train loss 0.75, test loss ~4.0),
  an honest lesson on needing enough data for more context to pay off.

## Phase 3 — The Transformer, piece by piece
- [x] **10 — Scaled dot-product attention**: Q/K/V on toy tensors by hand,
  scaling by `sqrt(d_k)`, and a causal mask demo (unmasked position 0 puts
  100% weight on future positions — a real bug; masked it's 0%, verified).
- [x] **11 — Multi-head attention**: splits `d_model` into heads, proves a
  naive per-head Python loop matches a vectorized batched version exactly,
  confirms causal masking holds per-head, wraps it as a reusable
  `MultiHeadAttention` module.
- [x] **12 — Positional encoding**: proves attention is permutation-
  equivariant (shuffle-attend-unshuffle reproduces the original output
  exactly), then shows adding sinusoidal positional encoding breaks that
  property on purpose, plus the learned-embedding alternative.
- [x] **13 — Layer normalization**: from-scratch implementation verified
  exact vs. `nn.LayerNorm`; shows unnormalized activations compounding from
  std 2.8 to 522 across 8 layers with a too-large init, vs. pinned at ~1.03
  every layer once `LayerNorm` is added.
- [x] **14 — Residual connections**: 30-layer stack, identical weights,
  gradient at the input goes from 1.5e-9 (vanished) without residuals to
  ~19 with them; also covers pre-norm vs. post-norm placement.
- [x] **15 — The feed-forward block**: `Linear(d_model→4x)→GELU→Linear(4x→d_model)`,
  ReLU-vs-GELU comparison, and a verified proof that every output row
  depends only on its own input row — no cross-token mixing at all, unlike
  attention.
- [x] **16 — Assembling one transformer block**: pre-norm attention +
  residual, pre-norm feed-forward + residual. Verified end-to-end: changing
  a later token never changes an earlier token's output through the whole
  block (not just raw attention weights), gradients flow, shape preserved
  across stacking 4 blocks.
- [x] **17 — Stacking blocks into nanoGPT**: token+positional embedding ->
  N transformer blocks -> final norm -> weight-tied head. Generalized
  attention to batches (verified equivalent to Module 16), verified
  end-to-end causality, and found/explained a real phenomenon: untrained
  weights already predict "repeat the input token" with >99% confidence,
  a structural effect of residual connections + weight tying, not noise.
- [x] **18 — Training nanoGPT on text**: character-level, ~2,650-char toy
  corpus. Best checkpoint hits perplexity 12.1 (vs. uniform baseline 55) —
  real generalization; continuing to 1500 steps drives train loss to 0.19
  while val perplexity balloons to 1065, and the final checkpoint's
  generations are long verbatim chunks of the training text — a direct,
  honest demo of memorization outpacing generalization at this tiny scale,
  motivating Phase 4-5's need for a real, much larger corpus. Also caught
  and fixed a real device-placement bug (causal mask on CPU vs CUDA).

## Phase 4 — Scaling Toward a Real LLM (paid GPU compute)
- [x] **19 — BPE tokenizer from scratch**: 40 merges on a toy corpus learn
  real patterns ("th"+"e"->"the", "and</w>" as a whole token), verified
  round-trip (`decode(encode(text)) == text`) and a measured 1.58x token
  reduction vs. character-level on a sample sentence.
- [x] **20 — Production tokenizers**: `tiktoken` (GPT-2) + HF `tokenizers`
  trained on our own corpus. Same sentence: char-level 41 -> Module 19's
  from-scratch BPE 26 -> HF-trained-on-tiny-corpus 17 -> GPT-2-pretrained 9
  tokens; confirmed GPT-2 fragments unseen proper nouns ("genshin" -> 3
  tokens) while common words stay 1 token.
- [x] **21 — Data pipelines for large corpora**: naive per-doc padding
  wastes 56.7% of tokens on padding (measured); concatenate+chunk wastes 0%,
  verified shift-by-one correctness. A streaming `IterableDataset`
  (one document at a time) produces chunks byte-for-byte identical to the
  materialize-everything approach, confirmed directly.
- [x] **22 — Mixed precision training** (fp16/bf16): a value that overflows
  fp16 to `inf` is fine in bf16; a real GPU benchmark measured bf16 matmul
  at ~3.4x faster than fp32 via tensor cores; `autocast` trains Module 18's
  NanoGPT correctly; fp16 needs `GradScaler`, bf16 doesn't (verified both).
- [x] **23 — Gradient accumulation**: proved exact equivalence (accumulated
  micro-batch gradients match a single big batch to 1e-6) and measured a
  real 29.3% peak-memory reduction (422MB -> 298MB) on this GPU for the
  same effective batch size.
- [x] **24 — The AdamW optimizer**: from-scratch Adam and AdamW both
  verified exact vs. `torch.optim` (needed the correct decay-before-update
  ordering); concretely demonstrated plain Adam's L2-via-gradient gives a
  parameter with small gradient history 1000x more decay than one with
  large history under identical settings — AdamW's decoupled decay is
  uniform regardless.
- [x] **25 — Learning rate schedules**: warmup + cosine decay verified
  exact vs. HF's `get_cosine_schedule_with_warmup` across all 100 steps;
  a deliberately aggressive LR showed training without warmup spiking to a
  loss of 106,965 in the first 30 steps vs. 6.9 with warmup.
- [x] **26 — Gradient clipping & weight decay**: global-norm clipping
  verified exact vs. `clip_grad_norm_`; an outlier batch produced grad
  norm ~1,165 and an unclipped update norm of ~11.6 vs. exactly 0.01
  (lr * max_norm) when clipped. Weight decay itself stays covered in
  Module 24 (AdamW), not re-derived here.
- [x] **27 — Working within Colab Pro**: compute-unit budgeting + session
  limits documented; built a full checkpoint (weights + optimizer state +
  RNG state + step) and verified resuming into a brand-new model/optimizer
  produces bit-identical losses/params to an uninterrupted run.

## Phase 5 — The Real Pretrain & Specialization
- [x] **28 — Sampling strategies**: greedy, temperature, top-k, top-p, all
  from scratch. Verified top-k=1 == greedy always, T=0.01 matches greedy in
  100% of 200 samples, top-p's kept set always meets its cumulative-
  probability threshold, and diversity (unique tokens/200 draws) ranks as
  expected: greedy 1 < top-k=2 2 < top-p=0.9 3 < T=1.0 5.
- [x] **29 — KV-caching**: cache-aware attention with position-offset
  causal masking, verified to produce byte-for-byte identical greedy
  generations with vs. without caching, and measured a real 2.03x speedup
  generating 200 tokens.
- [x] **30 — The pretraining corpus**: real pipeline mixing streamed
  `HuggingFaceFW/fineweb-edu` (general/real-world knowledge) with the
  `mrzjy/multimodal-genshin-impact` wiki (repeated to hit a 10% share),
  tokenized with GPT-2's real tokenizer, written as a memmap-verified
  `uint16` binary. Verified at 200K tokens (actual Genshin fraction landed
  at 0.100); scaling to the real 2.5B-token run is one constant change.
- [x] **31 — The actual pretraining run**: real GPT-2-small config
  (124,402,944 params, matching the public ~124M figure), with GPT-2's
  actual init scheme added (fixed an initial loss of 485 down to 10.95,
  right at the theoretical floor `ln(vocab_size)`=10.82 — a real bug the
  smoke test caught). Combines mixed precision + grad accum + decay/no-
  decay AdamW + warmup/cosine + clipping + checkpointing into one loop,
  verified end-to-end on this GPU. Scaling to the real 2.5B-token Colab Pro
  run needs three constants changed (documented in the notebook).
- [x] **32 — Instruction fine-tuning**: pretrains Module 18's toy model,
  then SFTs it on 4 Q/A pairs with a verified loss mask (manual vs.
  vectorized match to 1e-4). Held-out prompt ("Who is Diluc?", not in the
  SFT set): incoherent output before fine-tuning, a properly-formatted
  full sentence immediately after "### Answer:" afterward — real behavior/
  format learning, with the expected memorization caveat (train loss ->
  0.0005 on just 4 examples).
- [x] **33 — Stretch: RLHF/DPO concepts**: implemented DPO (not full RLHF's
  reward-model+PPO, explained conceptually instead). Initial loss matched
  the theoretical `ln(2)` exactly (policy==reference), preference margin
  grew from 0 to ~753 over 200 steps, and the frozen reference model's
  weights were verified unchanged throughout.

## Experiments — Other Kinds of Neural Networks

A separate, ongoing track from the `module_XX` curriculum above (which is
scoped tightly to "what it takes to train an LLM"). Here we step back and
poke at other things that go by the name "neural network," starting with
the biological original. Same one-component-per-notebook rule applies;
folders are `experiments/NN_topic/`, numbered in the order explored, not
implying any dependency between them the way the modules do.

- [x] **01 — The human (biological) neuron**: a leaky integrate-and-fire
  `Neuron` (membrane potential, leak, threshold, refractory period) —
  contrasted with Module 02's continuous `tanh` neuron: discrete spikes, no
  global loss, no backward pass. Learns instead via Hebbian plasticity
  ("cells that fire together, wire together"), demonstrated with a
  classical-conditioning setup: a stimulus paired with a reflex-triggering
  input vs. an unpaired control firing at the same ~20% rate. Paired
  synapse saturates to weight 1.000; the never-paired control lands at
  0.400 — a real, measured divergence from local correlation alone.

A second thread within this track (02-11) sketches a 5-stage toy pipeline for
human language production — comprehension -> meaning/memory/emotion/social
context -> response planning -> word/sentence construction -> motor
commands. Stages 02-06 are standalone PyTorch networks, each with its own
toy task, synthesizing stand-in inputs for what the previous stage would
produce rather than actually consuming it. Stage 07 then does the wiring for
real: stages 02-05 combined into one connected, trained conversational agent
(standard backprop-trained artificial neurons). Stage 08 rebuilds that exact
agent a second time on experiment 01's biological neuron instead, for a
direct, controlled comparison of the two learning mechanisms. Stage 09
rebuilds it a third time with real within-group recurrent connections and
inter-group associative links (Hebb's original "cell assembly" concept),
per explicit follow-up feedback on how 08 should have been wired. Stage 10
rebuilds it a fourth time, replacing 09's clamped (supervised-shortcut)
training with synaptic tagging and capture — local Hebbian tags + a global
success/error broadcast, no clamping anywhere. Stage 11 rebuilds it a fifth
time, keeping 10's tag-and-broadcast mechanism but stabilizing it with a
researched, individually-tested fix (Oja's rule)
(06/motor-commands stays deliberately unconnected — see its entry below).

- [x] **02 — Language comprehension**: embed -> mean-pool -> classify 16
  toy sentences into 4 intents (question/statement/greeting/command),
  hitting 100% train accuracy. The honest result: on a genuinely unseen
  sentence with two out-of-vocabulary words, it doesn't get *uncertain* —
  it confidently (p=1.00) misclassifies it as "greeting." A 16-sentence
  vocabulary has no way to represent "I don't know."
- [x] **03 — Meaning + memory + emotion + social context**: a `FusionNet`
  projects 4 separate input streams (meaning, memory, emotion, social
  context) into a shared space and predicts a response `tone` from a rule
  that only depends on emotion + formality. Test accuracy 84% (not 100% —
  the rule's hard threshold makes boundary cases genuinely ambiguous). A
  robustness check (swap in fresh random meaning/memory, holding
  emotion/social fixed) flipped the prediction 1 time out of 3 — since all
  4 streams get summed into one vector, irrelevant-stream noise can
  occasionally tip a near-boundary decision.
- [x] **04 — Response planning**: classifies a situational vector into one
  of 4 response strategies (answer directly / ask a clarifying question /
  empathize / give an instruction) from a 3-factor hidden rule
  (clarity/distress/directness). Test accuracy 88%; all 4 hand-built probe
  situations (one per rule branch) predicted the intended strategy.
- [x] **05 — Word and sentence construction**: a plan-conditioned
  `GRUCell` generator, teacher-forced on 16 toy sentences (4 per plan).
  Summed loss 66.76 -> 3.78 (~0.05/token — genuinely confident, not just
  low). Greedy decoding from each of the 4 plans exactly reproduces one
  memorized training sentence, word for word — the same memorization
  lesson Module 18 already taught: with only 4 examples per plan there's
  no shared structure to generalize from, so the network memorizes instead
  of composing.
- [x] **06 — Motor commands**: character-embeddings -> masked-mean-pool ->
  regress to a 3-dim toy articulatory vector (jaw height / lip rounding /
  tongue frontness), against a smooth, deterministic vowel-counting
  heuristic target. Unlike stage 05's arbitrary discrete mapping, this
  target *is* a smooth function of spelling — and it shows: held-out test
  MSE 0.00380 (RMSE ~6%) on 21 words never seen during training, a real
  generalization result, not memorization.
- [x] **07 — Linking it together: a small conversational agent**: wires
  stages 02-05 into one real `ConversationalAgent` (15,837 params) — actual
  tensors flowing stage to stage, not stand-ins — plus a persistent
  `GRUCell` memory carried across turns within a conversation. Trained on 5
  toy conversations (14 turns): all 4 auxiliary classifiers hit 100% train
  accuracy and all 14 generated responses matched targets exactly, including
  two same-plan turns needing different words (fixed stage 05's plan-only
  collapse by conditioning generation on `[fused ; plan_embed]` instead of
  plan alone). The key verification: the same "okay" turn, with identical
  text/emotion/social features, predicted `tone=supportive` with memory
  carried from a real distress conversation vs. `tone=urgent` with memory
  forcibly zeroed — an isolated, measured proof memory changes the agent's
  read of a situation, not just decorative wiring. A held-out probe
  conversation (4 of 8 words never seen in training) got plausible,
  on-topic responses, though at 14 training turns that's not strong evidence
  of real generalization either way. Stage 06 (motor commands) stays
  unconnected on purpose — text output, not articulation, is what a
  conversation needs.
- [x] **08 — The same conversation, on biological neurons**: rebuilds
  experiment 07's exact agent and exact 14-turn dataset, but every
  classifier is a population of experiment 01's real `Neuron` (leaky
  integrate-and-fire) trained with a supervised/clamped Hebbian rule
  (spike-count voting) instead of Adam + backprop; "generation" becomes
  **associative recall** over the 14 trained responses, since pure Hebbian
  learning has no established mechanism for word-by-word composition.
  Direct, fair comparison (separate train-then-evaluate passes, after
  catching a real bug where evaluating mid-training flattered the numbers):
  intent 93%, emotion 100%, tone 79%, plan 71%, response 93% — all real,
  meaningfully below backprop's 100% everywhere, the expected cost of no
  error-correcting signal. Memory still measurably matters (the identical
  "okay" test again: correct recall with memory carried vs. a different,
  wrong stored response with memory zeroed). The stark divergence: on the
  untrained probe, this version falls back to the same wrong stored
  response ("hello it is good to see you") for both novel lines, while
  experiment 07's backprop version produced plausible, on-topic responses
  — a real, visible illustration of associative recall's inability to
  interpolate toward anything not memorized, unlike a trained distributed
  representation.
- [x] **09 — Cell assemblies: groups that wire themselves, then wire
  together**: rebuilds experiment 08's agent a second time, per explicit
  follow-up feedback that groups should have real within-group (recurrent)
  connections, trained on their own first, *then* linked to other groups
  via separately-trained associative connections — Hebb's original 1949
  "cell assembly" concept. Concepts are patterns across a group of
  `Neuron`s wired to each other, not independent accumulators; an
  `InterGroupLink` turns one group's settled state into "charge" biasing
  another's (the literal "happy is given as a charge to those groups"
  mechanism). Three real bugs surfaced and fixed, not smoothed over: (1)
  reading a group's state at one instant instead of spike counts over the
  settling window — LIF refractory periods make driven neurons oscillate,
  not hold a static "on" state; (2) randomly-overlapping concept patterns
  causing cross-talk that alone dropped intent accuracy to 57% independent
  of recurrence entirely, fixed with disjoint pattern assignment; (3) the
  inter-group link saturating to one dominant response regardless of input
  — the same learning-rate trap as experiment 08, recurring at the link
  level. Fair (train-then-evaluate) results: intent 100%, emotion 100%,
  tone 86%, plan 86%, response 86% — beats experiment 08 on 3 of 5 tasks.
  Follow-up request ("related topics should have more similar patterns")
  led to a population/place-cell encoding for emotion — each neuron gets a
  random point in the valence-arousal circumplex (a standard psychology
  model), and a concept's pattern is whichever neurons are nearest to its
  coordinate — verified directly: `angry`/`anxious` (distance 0.14) share
  2/3 neurons, `happy`/`sad` (distance 1.80) share none, and accuracy
  matched or beat the disjoint-pattern version. Recurrence itself: real,
  trained, present — but four separate controlled tests (weakened/noisy
  cues, cue-then-silence, classic Hopfield-style corrupted-state recovery)
  found no measurable robustness benefit at this scale, reported honestly
  rather than dropped. The "okay" memory test now shows tone, plan, *and*
  response all shifting together through the literal charge-passing
  pathway, the most complete version of this test across 07/08/09.
- [x] **10 — Synaptic tagging and capture: local tags, global reward**:
  rebuilds experiment 09's agent a fourth time, replacing clamped
  (supervised-shortcut) training with a real named neuroscience mechanism —
  Frey & Morris's synaptic tagging and capture. Local Hebbian coincidence
  (pre-synaptic input active *and* the neuron fires) leaves an eligibility
  tag on that synapse; a single *global*, non-specific success/error signal
  is broadcast after the group's own **unforced** natural response is
  compared to the truth; only tagged synapses react. A genuine three-factor
  learning rule — no clamped target pattern anywhere. Needed a real
  addition clamped training never did: membrane noise for spontaneous
  activity, since zero-initialized weights that never fire spontaneously
  can never generate a tag to learn from. Toy sanity check bootstraps from
  all-zero weights to perfect (4/4) classification purely through
  exploration. Real, verified cost: a 10-seed sweep on intent alone shows
  genuine variance (`mean=9.1/13, range=7-11`) that clamped training simply
  doesn't have. Full pipeline: intent 79%, emotion 86%, tone 79%, plan 64%,
  response 36% — accuracy falls in lockstep with distance from the raw
  input (Response converges charge from 3 upstream groups on top of its own
  14-way exploration problem), the structural signature of sparse, delayed,
  global reward compounding through a multi-stage system, not a tuning
  failure (9+ epoch/lr combinations tested). Four-way comparison across
  07/08/09/10 now spans the full spectrum from fully-supervised backprop to
  reward-modulated three-factor Hebbian learning, with accuracy tracking
  exactly how much of the correct answer each mechanism is told directly
  versus has to discover through trial and error.
- [x] **11 — Stabilizing synaptic tagging: real research, three fixes, one
  kept**: given experiment 10's three named flaws (weight instability,
  recurrence with no proven benefit, sparse reward crashing the 14-way
  Response classifier), did actual `WebSearch` research (not recalled
  knowledge) into named, citable fixes, then tested each individually
  before combining, keeping only what testing actually validated. **Oja's
  rule** (self-normalizing Hebbian update, replacing hand-tuned weight
  clipping) — verified 10-seed win on a toy task (mean 9.1/13 -> 11.1/13)
  and kept; full pipeline vs. experiment 10: intent 79%->93%, emotion
  86%->86%, tone 79%->79%, plan 64%->64%, response 36%->43% — matches or
  beats on every task. **Asynchronous settling** (the textbook-correct
  answer to "recurrence should converge to a fixed point, not oscillate")
  — looked harmless on an isolated 4-concept toy test, then caused Response
  to collapse to 0% once combined with real inter-group charge in the full
  pipeline; reverted, and documented as a case where a fix validated in
  isolation still failed on integration. **Margin-based graded reward +
  persistent eligibility trace** (the textbook-correct answer to "flat
  binary reward is a weak signal in a 14-way space") — swept 6 learning
  rates against flat binary reward on the Response task; margin never won
  (best margin mean ~3.2/14 vs. best binary mean ~4.0/14); not adopted.
  Fair (train-then-evaluate) results vs. all prior conversational agents:
  intent 93%, emotion 86%, tone 79%, plan 64%, response 43%. Notably still
  *behind* experiment 09's clamped cell assemblies on tone/plan/response —
  an honestly-reported reminder that stabilizing *how* weights update
  can't make up for reward-modulated learning getting less direct
  supervision than clamping in the first place. The deepest flaw across
  all four Hebbian-family agents (08/09/10/11) remains unsolved by any of
  this: none can generate, only recall stored responses, confirmed again
  on this notebook's novel probe (both held-out lines converge to the
  identical response). Equilibrium propagation, predictive coding, and
  surrogate-gradient spiking networks were identified as real candidates
  for a future experiment 12, not yet built.

## Repo layout
```
module_01_autograd_engine/
module_02_neurons_and_layers/
module_03_training_a_network/
module_04_pytorch_tensors_autograd/
module_05_nn_module_and_optimizers/
module_06_counting_bigram_model/
module_07_perplexity/
module_08_token_embeddings/
module_09_neural_ngram_model/
module_10_scaled_dot_product_attention/
module_11_multi_head_attention/
module_12_positional_encoding/
module_13_layer_normalization/
module_14_residual_connections/
module_15_feed_forward_block/
module_16_transformer_block/
module_17_nanogpt_architecture/
module_18_training_nanogpt/
module_19_bpe_tokenizer/
module_20_production_tokenizers/
module_21_data_pipelines/
module_22_mixed_precision_training/
module_23_gradient_accumulation/
module_24_adamw_optimizer/
module_25_learning_rate_schedules/
module_26_gradient_clipping_and_weight_decay/
module_27_working_within_colab_pro/
module_28_sampling_strategies/
module_29_kv_caching/
module_30_pretraining_corpus/
module_31_pretraining_run/
module_32_instruction_finetuning/
module_33_rlhf_dpo_concepts/
experiments/
  01_human_neuron/
  02_language_comprehension/
  03_meaning_memory_emotion_context/
  04_response_planning/
  05_word_sentence_construction/
  06_motor_commands/
  07_connected_conversation/
  08_biological_conversation/
  09_cell_assembly_conversation/
  10_synaptic_tagging_conversation/
  11_stabilized_conversation/
```
