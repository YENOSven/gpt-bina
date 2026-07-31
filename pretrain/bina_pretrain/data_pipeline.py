import os

import numpy as np
import yaml
from datasets import load_dataset

from bina_pretrain import data_manifest as dm
from bina_pretrain.tokenizer import EOT, encode

# stage_1_pretrain categories in configs/data_mix.yaml use one of two ways to size a source's
# individual target from its category total: "fraction" (general_english: split a big shared
# target by ratio) or "measured_tokens" (natural_dialogue: sources are small enough that we
# just take ~everything each one actually has, already measured directly rather than estimated)
GENERAL_ENGLISH = "general_english"
NATURAL_DIALOGUE = "natural_dialogue"


def load_mix_config(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_manifest_from_mix(manifest, mix_config):
    """Ensures a data_manifest entry exists for every source listed under
    configs/data_mix.yaml's stage_1_pretrain, computing each source's own
    target_train_tokens from its category. Never resets an already-in-progress entry
    (get_or_create_entry's existing behavior) -- safe to call at the top of every session."""
    stage = mix_config["stage_1_pretrain"]
    val_fraction = stage["split"]["val"]
    test_fraction = stage["split"]["test"]

    for src in stage[GENERAL_ENGLISH]["sources"]:
        target = int(stage[GENERAL_ENGLISH]["target_tokens"] * src["fraction"])
        dm.get_or_create_entry(manifest, src["name"], src["hf_dataset"], src.get("config"), "train",
                                target, val_fraction=val_fraction, test_fraction=test_fraction,
                                text_field=src.get("text_field", "text"))

    for src in stage[NATURAL_DIALOGUE]["sources"]:
        # take ~everything measured (train+val+test together should land close to the source's
        # real total rather than deliberately falling short of a scarce resource)
        target = int(src["measured_tokens"] / (1 + val_fraction + test_fraction))
        dm.get_or_create_entry(manifest, src["name"], src["hf_dataset"], src.get("config"), "train",
                                target, val_fraction=val_fraction, test_fraction=test_fraction,
                                text_field=src.get("text_field", "text"))

    return manifest


def all_source_names(mix_config):
    stage = mix_config["stage_1_pretrain"]
    return ([s["name"] for s in stage[GENERAL_ENGLISH]["sources"]]
            + [s["name"] for s in stage[NATURAL_DIALOGUE]["sources"]])


def ingest_source(manifest, source_name, shard_dir, tokens_per_shard, max_tokens_this_call=None):
    """Resumes `manifest[source_name]` from wherever it last left off (fresh start if this is
    the first call), streaming+tokenizing from the underlying HF source and writing shard
    files to `shard_dir`. Mutates `manifest` in place -- call data_manifest.save yourself
    afterward, same convention as checkpoint.py leaving the caller in control of persistence.
    Stops after `max_tokens_this_call` tokens (a session budget) or once the source is fully
    ingested (val+test+train all reach their targets) or runs dry, whichever comes first.
    Returns the number of tokens ingested this call."""
    entry = manifest[source_name]
    if dm.is_complete(entry):
        return 0

    ds = load_dataset(entry["hf_dataset"], entry["config"], split=entry["hf_split"], streaming=True)
    if entry["stream_state"] is not None:
        ds.load_state_dict(entry["stream_state"])
    it = iter(ds)

    os.makedirs(shard_dir, exist_ok=True)
    tokens_this_call = 0
    exhausted = False

    while not dm.is_complete(entry) and not exhausted:
        if max_tokens_this_call is not None and tokens_this_call >= max_tokens_this_call:
            break

        phase = entry["phase"]
        budget = dm.current_phase_remaining(entry)
        if max_tokens_this_call is not None:
            budget = min(budget, max_tokens_this_call - tokens_this_call)
        budget = min(budget, tokens_per_shard)

        shard_tokens = []
        while len(shard_tokens) < budget:
            try:
                text = next(it)[entry.get("text_field", "text")]
            except StopIteration:
                exhausted = True
                break
            shard_tokens.extend(encode(text))
            shard_tokens.append(EOT)

        if not shard_tokens:
            break

        shard_index = len(entry["shards_written"][phase])
        shard_filename = f"{source_name}_{phase}_{shard_index:06d}.bin"
        np.array(shard_tokens, dtype=np.uint16).tofile(os.path.join(shard_dir, shard_filename))

        dm.record_shard(entry, phase, shard_filename, len(shard_tokens), ds.state_dict())
        tokens_this_call += len(shard_tokens)

    if exhausted:
        dm.mark_exhausted(entry)

    return tokens_this_call


def _split_into_documents(tokens):
    """Splits a shard's flat EOT-separated token array into a list of document arrays, using
    vectorized numpy slicing (np.where + array slices) rather than a Python-level loop over
    every token -- the same split, but the difference between milliseconds and minutes once a
    shard has millions of tokens."""
    eot_positions = np.where(tokens == EOT)[0]
    docs = []
    start = 0
    for eot_pos in eot_positions:
        if eot_pos > start:
            docs.append(tokens[start:eot_pos])
        start = eot_pos + 1
    if start < len(tokens):
        docs.append(tokens[start:])
    return docs


def combine_shards_to_corpus(manifest, source_names, shard_dir, phase, out_path, seed=42):
    """One-time (non-resumable, meant to be re-run from scratch if interrupted) step: writes
    the final training-ready corpus_<phase>.bin by shuffling shard order across all sources,
    then a smaller local shuffle of the documents within each shard, streaming straight to
    `out_path` one shard at a time.

    NOT a full global shuffle of every individual document -- deliberately, after a real bug:
    the first version of this function read every document into a Python list of Python ints,
    holding the WHOLE corpus as native Python objects at once. That's fine at the toy scale it
    was tested at, but at real scale (billions of tokens) it needs 50-100x more memory than the
    packed data actually occupies (a Python int is ~28 bytes vs. 2 bytes packed), which was
    observed forcing heavy OS-level disk paging on a real ~6.5B-token run. Shuffling at the
    shard level (a few hundred shards, not millions of documents) plus a within-shard document
    shuffle keeps peak memory bounded to one shard's size (~tokens_per_shard * 2 bytes)
    regardless of total corpus size, while still interleaving every source throughout the
    output -- shard-level shuffling is a standard tradeoff real large-scale pipelines make for
    exactly this reason, not a shortcut unique to this project."""
    import random

    rng = random.Random(seed)

    shard_refs = []  # (source_name, phase, shard_filename)
    for source_name in source_names:
        entry = manifest[source_name]
        for shard in entry["shards_written"][phase]:
            shard_refs.append(shard["filename"])
    rng.shuffle(shard_refs)

    total_tokens = 0
    eot_bytes = np.array([EOT], dtype=np.uint16).tobytes()
    with open(out_path, "wb") as out_f:
        for shard_filename in shard_refs:
            tokens = np.fromfile(os.path.join(shard_dir, shard_filename), dtype=np.uint16)
            docs = _split_into_documents(tokens)
            rng.shuffle(docs)
            for doc in docs:
                out_f.write(doc.tobytes())
                out_f.write(eot_bytes)
                total_tokens += len(doc) + 1

    return total_tokens
