import os
import random
import tempfile

import numpy as np
import torch


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
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
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


def try_resume(path, model, optimizer=None, map_location=None):
    """Returns the resume-state dict if a checkpoint exists at `path`, else None
    (fresh-start case). Callers use `is None` to decide fresh-init vs resume."""
    if not os.path.exists(path):
        return None
    return load_checkpoint(path, model, optimizer, map_location=map_location)
