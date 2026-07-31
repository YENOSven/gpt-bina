"""Phase-0 proof (see the plan's Training Infrastructure section): a training run
killed partway through and resumed from its checkpoint must produce results
bit-identical to an uninterrupted run over the same steps. Each invocation below is
a genuinely separate `python` subprocess (not just fresh objects in this process),
so nothing can accidentally leak state between "sessions" the way in-process reuse
could.

Note on scope: the model has no dropout and data is read by a deterministic
sequential cursor (no random sampling), so nothing in the current training loop
actually consumes RNG state mid-run. The RNG save/restore in checkpoint.py is real
and exercised (loaded and restored every call) but this particular test can't prove
it matters yet -- that only becomes observable once something stochastic (dropout,
data shuffling) is added to the loop. Said plainly rather than implied.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PRETRAIN_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCRATCH = PRETRAIN_DIR / ".resume_verify_scratch"

TOTAL_STEPS = 12
SPLIT_AT = 6  # first subprocess runs steps 0..SPLIT_AT-1, second resumes and runs the rest
CORPUS_TOKENS = 20_000  # small enough to exercise the sequential-cursor wrap-around too


def run(args):
    result = subprocess.run(
        [sys.executable, "-m", "bina_pretrain.train", *args],
        cwd=PRETRAIN_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"subprocess failed: {' '.join(args)}")
    return result.stdout


def make_corpus(path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50257, size=CORPUS_TOKENS, dtype=np.uint16)
    tokens.tofile(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch-dir", default=str(DEFAULT_SCRATCH),
                    help="where to write the throwaway proof corpus/checkpoints -- pass a "
                         "Drive-mounted path (e.g. from Colab) to prove the mechanism against "
                         "real Drive I/O instead of local disk")
    args = p.parse_args()
    SCRATCH = Path(args.scratch_dir)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    corpus_path = SCRATCH / "corpus.bin"
    make_corpus(corpus_path)

    common = [
        "--config", "tiny", "--corpus-path", str(corpus_path),
        "--micro-batch-size", "2", "--grad-accum-steps", "2",
        "--warmup-steps", "3", "--device", "cpu",
    ]

    # --- uninterrupted run: all TOTAL_STEPS in one process ---
    uninterrupted_ckpt = SCRATCH / "uninterrupted.pt"
    uninterrupted_log = SCRATCH / "uninterrupted_losses.json"
    print("running uninterrupted...")
    run([*common, "--total-steps", str(TOTAL_STEPS), "--checkpoint-every", str(TOTAL_STEPS),
         "--save-path", str(uninterrupted_ckpt), "--log-path", str(uninterrupted_log)])

    # --- interrupted run: two separate subprocess invocations. --total-steps stays 12 for
    # BOTH (it's the LR schedule's fixed target for the whole run, same as the uninterrupted
    # run above) -- only --session-step-limit differs, exactly like a daily Colab session
    # stopping partway through a schedule sized for the whole multi-month run. (An earlier
    # version of this script passed a smaller --total-steps to part A instead, which silently
    # changed the cosine-decay shape and made the runs diverge for the wrong reason.)
    interrupted_ckpt = SCRATCH / "interrupted.pt"
    log_a = SCRATCH / "interrupted_losses_a.json"
    log_b = SCRATCH / "interrupted_losses_b.json"
    print(f"running interrupted, part A (steps 0..{SPLIT_AT - 1})...")
    run([*common, "--total-steps", str(TOTAL_STEPS), "--session-step-limit", str(SPLIT_AT),
         "--checkpoint-every", str(SPLIT_AT),
         "--save-path", str(interrupted_ckpt), "--log-path", str(log_a)])
    print(f"'killed' after step {SPLIT_AT - 1}. running part B, resumed (steps {SPLIT_AT}..{TOTAL_STEPS - 1})...")
    run([*common, "--total-steps", str(TOTAL_STEPS), "--checkpoint-every", str(TOTAL_STEPS - SPLIT_AT),
         "--resume-from", str(interrupted_ckpt), "--save-path", str(interrupted_ckpt), "--log-path", str(log_b)])

    # --- compare ---
    uninterrupted_losses = json.loads(uninterrupted_log.read_text())
    interrupted_losses = json.loads(log_a.read_text()) + json.loads(log_b.read_text())

    print(f"\nuninterrupted losses: {[round(x, 6) for x in uninterrupted_losses]}")
    print(f"interrupted  losses: {[round(x, 6) for x in interrupted_losses]}")
    assert len(uninterrupted_losses) == len(interrupted_losses) == TOTAL_STEPS
    assert uninterrupted_losses == interrupted_losses, "per-step losses diverged after resume"

    ckpt_a = torch.load(uninterrupted_ckpt, weights_only=False)
    ckpt_b = torch.load(interrupted_ckpt, weights_only=False)
    for (name_a, tensor_a), (name_b, tensor_b) in zip(ckpt_a["model"].items(), ckpt_b["model"].items()):
        assert name_a == name_b
        assert torch.equal(tensor_a, tensor_b), f"parameter {name_a} diverged after resume"
    assert ckpt_a["global_token_cursor"] == ckpt_b["global_token_cursor"]
    assert ckpt_a["step"] == ckpt_b["step"] == TOTAL_STEPS - 1

    print("\nPASS: killed-and-resumed run is bit-identical to an uninterrupted run "
          f"(losses, all {len(ckpt_a['model'])} parameter tensors, and the data cursor all match exactly).")


if __name__ == "__main__":
    main()
