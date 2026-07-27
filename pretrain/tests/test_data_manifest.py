from columbina_pretrain import data_manifest as dm


def test_new_entry_starts_in_val_phase_with_correct_targets():
    entry = dm.new_entry("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", target_train_tokens=1000,
                          val_fraction=0.01, test_fraction=0.01)
    assert entry["phase"] == "val"
    assert entry["targets"] == {"val": 10, "test": 10, "train": 1000}
    assert entry["status"] == dm.STATUS_NOT_STARTED
    assert dm.current_phase_remaining(entry) == 10


def test_manifest_round_trips_through_disk(tmp_path):
    path = str(tmp_path / "manifest.json")
    manifest = {}
    dm.get_or_create_entry(manifest, "fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 1000)
    dm.save(manifest, path)
    assert dm.load(path) == manifest


def test_phases_advance_in_order_val_test_train_done():
    entry = dm.new_entry("src", None, "train", target_train_tokens=100, val_fraction=0.1, test_fraction=0.1)
    assert entry["targets"] == {"val": 10, "test": 10, "train": 100}

    # val phase: one shard covers it exactly
    dm.record_shard(entry, "val", "src_val_000000.bin", 10, {"pos": 1})
    assert entry["phase"] == "test"
    assert entry["status"] == dm.STATUS_IN_PROGRESS

    # test phase: two shards, neither alone completes it
    dm.record_shard(entry, "test", "src_test_000000.bin", 6, {"pos": 2})
    assert entry["phase"] == "test"
    dm.record_shard(entry, "test", "src_test_000001.bin", 4, {"pos": 3})
    assert entry["phase"] == "train"

    # train phase: not complete until its full target is reached
    dm.record_shard(entry, "train", "src_train_000000.bin", 60, {"pos": 4})
    assert entry["phase"] == "train"
    assert not dm.is_complete(entry)
    assert dm.remaining_train_tokens(entry) == 40

    dm.record_shard(entry, "train", "src_train_000001.bin", 40, {"pos": 5})
    assert entry["phase"] == "done"
    assert dm.is_complete(entry)
    assert dm.remaining_train_tokens(entry) == 0


def test_shard_offsets_are_per_phase_not_global():
    entry = dm.new_entry("src", None, "train", target_train_tokens=100, val_fraction=0.01, test_fraction=0.01)
    entry["phase"] = "train"  # jump straight to train phase for this test
    dm.record_shard(entry, "train", "a.bin", 30, {})
    dm.record_shard(entry, "train", "b.bin", 20, {})
    assert entry["shards_written"]["train"][0]["phase_offset_start"] == 0
    assert entry["shards_written"]["train"][1]["phase_offset_start"] == 30


def test_mark_exhausted_does_not_override_an_already_complete_entry():
    entry = dm.new_entry("src", None, "train", target_train_tokens=10, val_fraction=0, test_fraction=0)
    dm.record_shard(entry, "train", "a.bin", 10, {})
    assert dm.is_complete(entry)
    dm.mark_exhausted(entry)
    assert dm.is_complete(entry), "exhaustion after a source is already complete must not downgrade its status"


def test_mark_exhausted_flags_a_source_that_ran_dry_early():
    entry = dm.new_entry("src", None, "train", target_train_tokens=1000, val_fraction=0, test_fraction=0)
    entry["phase"] = "train"
    dm.record_shard(entry, "train", "a.bin", 50, {})
    dm.mark_exhausted(entry)
    assert entry["status"] == dm.STATUS_EXHAUSTED
    assert not dm.is_complete(entry)


def test_load_missing_manifest_returns_empty_dict(tmp_path):
    assert dm.load(str(tmp_path / "does_not_exist.json")) == {}
