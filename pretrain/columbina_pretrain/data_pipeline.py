import os

import numpy as np
from datasets import load_dataset

from columbina_pretrain import data_manifest as dm
from columbina_pretrain.tokenizer import EOT, encode


def ingest_source(manifest, source_name, shard_dir, tokens_per_shard, text_field="text", max_tokens_this_call=None):
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
                text = next(it)[text_field]
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


def combine_shards_to_corpus(manifest, source_names, shard_dir, phase, out_path, seed=42):
    """One-time (non-resumable, meant to be re-run from scratch if interrupted -- it's fast
    relative to ingestion) step: reads every already-ingested shard for `phase` across the
    given sources, splits back into documents on EOT, shuffles document order across ALL
    sources together (so e.g. FineWeb-Edu and OpenWebText pages interleave rather than
    appearing in source-sized blocks), and concatenates into one flat uint16 file -- same
    format module 30 already proved (np.memmap-compatible, EOT-separated documents)."""
    import random

    all_docs = []
    for source_name in source_names:
        entry = manifest[source_name]
        for shard in entry["shards_written"][phase]:
            tokens = np.fromfile(os.path.join(shard_dir, shard["filename"]), dtype=np.uint16)
            doc = []
            for token_id in tokens:
                if token_id == EOT:
                    if doc:
                        all_docs.append(doc)
                    doc = []
                else:
                    doc.append(int(token_id))
            if doc:
                all_docs.append(doc)

    random.Random(seed).shuffle(all_docs)

    stream = []
    for doc in all_docs:
        stream.extend(doc)
        stream.append(EOT)

    tokens = np.array(stream, dtype=np.uint16)
    tokens.tofile(out_path)
    return len(tokens)
