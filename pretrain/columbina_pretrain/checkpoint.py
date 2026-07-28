import os
import random
import tempfile
import time

import numpy as np
import torch

_RENAME_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)  # ~3.1s worst case before giving up


def _replace_with_retry(tmp_path, path):
    """os.replace can transiently fail with PermissionError/WinError 5 on Windows when the
    destination is momentarily locked by OneDrive's sync client, an antivirus scanner, or the
    search indexer -- observed for real running a local-fallback session inside a
    OneDrive-synced folder. Real transient locks like this typically clear within
    milliseconds; retrying with a short backoff rides them out instead of losing a whole
    training session to one badly-timed checkpoint write."""
    last_error = None
    for delay in (0.0,) + _RENAME_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as e:
            last_error = e
    raise last_error


def save_checkpoint(model, optimizer, step, global_token_cursor, path, wall_clock_seconds_trained=0.0, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "step": step,
        "global_token_cursor": global_token_cursor,
        "wall_clock_seconds_trained": wall_clock_seconds_trained,
        "extra": extra or {},
    }
    # Colab can disconnect at any instant; write-then-rename means a killed process
    # never leaves a half-written, unloadable file at `path`.
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
        _replace_with_retry(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(path, model, optimizer=None):
    # Always deserialize onto CPU -- a previous version of this function accepted a
    # map_location and forwarded it straight to torch.load, which (with map_location="cuda")
    # moved EVERY tensor in the checkpoint onto the GPU during load, including the RNG state
    # ByteTensor. torch.set_rng_state() below requires a CPU tensor specifically and raises
    # TypeError on a CUDA one -- a real bug that stayed hidden because every earlier test
    # either never resumed a real checkpoint at all, or only ever did so with device="cpu".
    # model.load_state_dict/optimizer.load_state_dict correctly place values on whatever
    # device model/optimizer are already on regardless of what device the incoming state_dict
    # tensors start on, so loading onto CPU and letting those calls handle placement is both
    # correct and simpler -- no map_location parameter needed at all.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    rng = checkpoint["rng_state"]
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])

    return {
        "step": checkpoint["step"],
        "global_token_cursor": checkpoint["global_token_cursor"],
        "wall_clock_seconds_trained": checkpoint.get("wall_clock_seconds_trained", 0.0),
        "extra": checkpoint.get("extra", {}),
    }


def try_resume(path, model, optimizer=None):
    """Returns the resume-state dict if a checkpoint exists at `path`, else None
    (fresh-start case). Callers use `is None` to decide fresh-init vs resume."""
    if not os.path.exists(path):
        return None
    return load_checkpoint(path, model, optimizer)
