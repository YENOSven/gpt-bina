import json
import os
import tempfile

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"
STATUS_EXHAUSTED = "exhausted"  # source ran out before target_train_tokens was reached

PHASE_ORDER = ["val", "test", "train", "done"]


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(manifest, path):
    # same atomic write-then-rename pattern as checkpoint.py -- a session can be interrupted
    # mid-write at any point, and a half-written manifest would otherwise corrupt every
    # future session's idea of what's already been ingested.
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def new_entry(hf_dataset, config, hf_split, target_train_tokens, val_fraction=0.01, test_fraction=0.01):
    """One entry = one continuous pass through one HF streaming source. val and test are
    carved off the FRONT of that single stream first (phases "val" then "test"), and train
    continues from wherever that leaves off -- so there's only ever one stream_state to track
    (not three independent ones), and val/test tokens can never later be re-seen by train. This
    relies on the upstream source not being suspiciously ordered (true for FineWeb-Edu/
    OpenWebText's streaming shard order) rather than a per-document hash -- simpler, and just
    as valid for a source with no meaningful document ordering."""
    # held out as a fraction of train's target, not of the total -- e.g. val_fraction=0.01 with
    # target_train_tokens=100 gives target_val=1, matching the "98 train / 1 val / 1 test" split
    target_val = round(target_train_tokens * val_fraction)
    target_test = round(target_train_tokens * test_fraction)
    return {
        "hf_dataset": hf_dataset,
        "config": config,
        "hf_split": hf_split,
        "phase": "val",
        "targets": {"val": target_val, "test": target_test, "train": target_train_tokens},
        "tokens_ingested": {"val": 0, "test": 0, "train": 0},
        "stream_state": None,
        "shards_written": {"val": [], "test": [], "train": []},
        "status": STATUS_NOT_STARTED,
    }


def get_or_create_entry(manifest, source_name, hf_dataset, config, hf_split, target_train_tokens, **kwargs):
    if source_name not in manifest:
        manifest[source_name] = new_entry(hf_dataset, config, hf_split, target_train_tokens, **kwargs)
    return manifest[source_name]


def current_phase_remaining(entry):
    phase = entry["phase"]
    if phase == "done":
        return 0
    return max(0, entry["targets"][phase] - entry["tokens_ingested"][phase])


def record_shard(entry, phase, shard_filename, token_count, stream_state):
    """Call once per shard written, for whichever phase is currently active."""
    entry["shards_written"][phase].append({
        "filename": shard_filename,
        "token_count": token_count,
        "phase_offset_start": entry["tokens_ingested"][phase],
    })
    entry["tokens_ingested"][phase] += token_count
    entry["stream_state"] = stream_state
    _advance_phase_if_done(entry)


def _advance_phase_if_done(entry):
    while entry["phase"] != "done" and current_phase_remaining(entry) <= 0:
        idx = PHASE_ORDER.index(entry["phase"])
        entry["phase"] = PHASE_ORDER[idx + 1]
    entry["status"] = STATUS_COMPLETE if entry["phase"] == "done" else STATUS_IN_PROGRESS


def mark_exhausted(entry):
    """Call when the underlying stream runs out of data before train's target is reached --
    this is a real, honest outcome to surface (see the plan's natural-dialogue-dedup risk),
    not something to silently paper over by looping the source or under-filling silently."""
    if entry["status"] != STATUS_COMPLETE:
        entry["status"] = STATUS_EXHAUSTED


def is_complete(entry):
    return entry["status"] == STATUS_COMPLETE


def remaining_train_tokens(entry):
    return max(0, entry["targets"]["train"] - entry["tokens_ingested"]["train"])
