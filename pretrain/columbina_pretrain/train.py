import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from columbina_pretrain import checkpoint as ckpt
from columbina_pretrain.model import CONFIG_124M, CONFIG_406M, CONFIG_TINY, build_model

PEAK_LR = 6e-4
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
BETAS = (0.9, 0.95)

CONFIGS = {"tiny": CONFIG_TINY, "124m": CONFIG_124M, "406m": CONFIG_406M}


def lr_multiplier(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def load_corpus(bin_path):
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


def get_batch_sequential(data, block_size, batch_size, cursor):
    """Reads batch_size non-overlapping (block_size+1)-token windows starting at
    `cursor`, advancing it deterministically -- no RNG involved, so a resumed run
    reads exactly the tokens an uninterrupted run would have read next."""
    chunk_size = block_size + 1
    total_needed = batch_size * chunk_size
    if cursor + total_needed > len(data):
        cursor = 0  # single-pass wrap; real corpora are sized so this essentially never fires
    xs, ys = [], []
    for i in range(batch_size):
        start = cursor + i * chunk_size
        window = torch.from_numpy(data[start:start + chunk_size].astype(np.int64))
        xs.append(window[:-1])
        ys.append(window[1:])
    new_cursor = cursor + total_needed
    return torch.stack(xs), torch.stack(ys), new_cursor


def make_optimizer(model):
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=PEAK_LR, betas=BETAS,
    )


def run_training(
    config_name, corpus_path, total_steps, warmup_steps, micro_batch_size, grad_accum_steps,
    save_path, checkpoint_every, resume_from=None, device=None, seed=42, session_step_limit=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[config_name]
    model = build_model(config, device, seed=seed)
    optimizer = make_optimizer(model)

    step = 0
    global_token_cursor = 0
    wall_clock_seconds_trained = 0.0
    if resume_from is not None:
        state = ckpt.try_resume(resume_from, model, optimizer, map_location=device)
        if state is not None:
            step = state["step"] + 1
            global_token_cursor = state["global_token_cursor"]
            wall_clock_seconds_trained = state["wall_clock_seconds_trained"]

    corpus = load_corpus(corpus_path)
    block_size = config["max_seq_len"]
    losses = []
    session_start = time.time()
    session_stop_step = total_steps if session_step_limit is None else min(total_steps, step + session_step_limit)

    while step < session_stop_step:
        lr = PEAK_LR * lr_multiplier(step, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        accumulated_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y, global_token_cursor = get_batch_sequential(corpus, block_size, micro_batch_size, global_token_cursor)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1)) / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        losses.append(accumulated_loss)

        if (step + 1) % checkpoint_every == 0 or step == session_stop_step - 1:
            wall_clock_seconds_trained += time.time() - session_start
            session_start = time.time()
            ckpt.save_checkpoint(model, optimizer, step, global_token_cursor, save_path, wall_clock_seconds_trained)

        step += 1

    return losses, model, optimizer


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="124m", choices=list(CONFIGS))
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--save-path", required=True)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--log-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--session-step-limit", type=int, default=None,
                    help="stop after this many steps THIS invocation, without changing "
                         "--total-steps (which stays fixed as the LR schedule's target for "
                         "the whole run, exactly like a daily Colab session vs. the overall run)")
    args = p.parse_args()

    losses, _, _ = run_training(
        config_name=args.config, corpus_path=args.corpus_path, total_steps=args.total_steps,
        warmup_steps=args.warmup_steps, micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps, save_path=args.save_path,
        checkpoint_every=args.checkpoint_every, resume_from=args.resume_from,
        device=args.device, seed=args.seed, session_step_limit=args.session_step_limit,
    )
    print(f"ran {len(losses)} step(s), final loss {losses[-1]:.6f}" if losses else "no steps ran")
    if args.log_path:
        with open(args.log_path, "w") as f:
            json.dump(losses, f)


if __name__ == "__main__":
    _cli()
