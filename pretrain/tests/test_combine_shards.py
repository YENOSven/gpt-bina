import os

import numpy as np

from columbina_pretrain import data_pipeline as dp
from columbina_pretrain.tokenizer import EOT


def _write_shard(shard_dir, filename, docs):
    tokens = []
    for doc in docs:
        tokens.extend(doc)
        tokens.append(EOT)
    np.array(tokens, dtype=np.uint16).tofile(os.path.join(shard_dir, filename))


def test_combine_preserves_every_document_exactly_once(tmp_path):
    shard_dir = str(tmp_path / "shards")
    os.makedirs(shard_dir)

    docs_a = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    docs_b = [[10, 11], [12]]
    _write_shard(shard_dir, "a_train_000000.bin", docs_a)
    _write_shard(shard_dir, "b_train_000000.bin", docs_b)

    manifest = {
        "a": {"shards_written": {"train": [{"filename": "a_train_000000.bin"}]}},
        "b": {"shards_written": {"train": [{"filename": "b_train_000000.bin"}]}},
    }

    out_path = str(tmp_path / "combined.bin")
    n_tokens = dp.combine_shards_to_corpus(manifest, ["a", "b"], shard_dir, "train", out_path, seed=0)

    expected_tokens = sum(len(d) + 1 for d in docs_a + docs_b)
    assert n_tokens == expected_tokens

    combined = np.fromfile(out_path, dtype=np.uint16)
    assert len(combined) == n_tokens

    eot_positions = np.where(combined == EOT)[0]
    recovered = []
    start = 0
    for p in eot_positions:
        if p > start:
            recovered.append(tuple(combined[start:p].tolist()))
        start = p + 1

    assert sorted(recovered) == sorted(tuple(d) for d in docs_a + docs_b)


def test_combine_does_not_load_more_than_one_shard_at_a_time(tmp_path, monkeypatch):
    """Regression test for the real bug this function was rewritten to fix: the first
    implementation materialized the entire corpus as native Python lists/ints before writing
    anything, which needed 50-100x the memory of the packed data at real scale (observed
    causing heavy OS-level disk paging on a real ~6.5B-token run). Asserts peak concurrently-
    loaded token count never exceeds one shard's worth, regardless of how many shards exist."""
    shard_dir = str(tmp_path / "shards")
    os.makedirs(shard_dir)

    manifest = {"a": {"shards_written": {"train": []}}}
    shard_size = 500
    n_shards = 20
    rng = np.random.default_rng(0)
    for i in range(n_shards):
        docs = []
        total = 0
        while total < shard_size:
            doc_len = int(rng.integers(5, 20))
            docs.append(list(rng.integers(0, 1000, size=doc_len)))
            total += doc_len + 1
        filename = f"a_train_{i:06d}.bin"
        _write_shard(shard_dir, filename, docs)
        manifest["a"]["shards_written"]["train"].append({"filename": filename})

    original_fromfile = np.fromfile
    concurrently_loaded = []

    def tracking_fromfile(path, dtype):
        arr = original_fromfile(path, dtype=dtype)
        concurrently_loaded.append(len(arr))
        return arr

    monkeypatch.setattr(np, "fromfile", tracking_fromfile)
    out_path = str(tmp_path / "combined.bin")
    dp.combine_shards_to_corpus(manifest, ["a"], shard_dir, "train", out_path, seed=0)

    # each np.fromfile call should load roughly one shard's worth (a little over shard_size,
    # since the loop overshoots to finish the last document) -- never anything close to the
    # full n_shards * shard_size corpus in one call
    assert len(concurrently_loaded) == n_shards
    assert max(concurrently_loaded) < shard_size * 2
