import os

from bina_pretrain import data_manifest as dm
from bina_pretrain import data_pipeline as dp

MIX_YAML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "data_mix.yaml")


def test_load_mix_config_reads_the_real_committed_file():
    config = dp.load_mix_config(MIX_YAML_PATH)
    stage = config["stage_1_pretrain"]
    assert stage["general_english"]["target_tokens"] > 0
    assert stage["natural_dialogue"]["target_tokens"] > 0
    assert len(stage["general_english"]["sources"]) == 2
    assert len(stage["natural_dialogue"]["sources"]) == 4


def test_init_manifest_from_mix_creates_one_entry_per_source_with_sane_targets():
    config = dp.load_mix_config(MIX_YAML_PATH)
    manifest = {}
    dp.init_manifest_from_mix(manifest, config)

    names = dp.all_source_names(config)
    assert set(manifest.keys()) == set(names)
    assert "fineweb_edu" in manifest and "openwebtext" in manifest
    assert "babylm_open_subtitles" in manifest

    fineweb = manifest["fineweb_edu"]
    openwebtext = manifest["openwebtext"]
    stage = config["stage_1_pretrain"]
    # 70/30 split of general_english's total, within float-rounding tolerance
    assert abs(fineweb["targets"]["train"] - stage["general_english"]["target_tokens"] * 0.70) < 2
    assert abs(openwebtext["targets"]["train"] - stage["general_english"]["target_tokens"] * 0.30) < 2

    # natural_dialogue sources: train+val+test should land close to each source's measured total
    subs = manifest["babylm_open_subtitles"]
    total_target = sum(subs["targets"].values())
    measured = next(s["measured_tokens"] for s in stage["natural_dialogue"]["sources"] if s["name"] == "babylm_open_subtitles")
    assert abs(total_target - measured) / measured < 0.02

    # every entry starts in the val phase, not_started
    for entry in manifest.values():
        assert entry["phase"] == "val"
        assert entry["status"] == dm.STATUS_NOT_STARTED


def test_init_manifest_from_mix_does_not_reset_in_progress_entries():
    config = dp.load_mix_config(MIX_YAML_PATH)
    manifest = {}
    dp.init_manifest_from_mix(manifest, config)
    dm.record_shard(manifest["fineweb_edu"], "val", "fineweb_edu_val_000000.bin", 500, {"shard_idx": 3})

    # calling init again (as every session's startup cell does) must not wipe that progress
    dp.init_manifest_from_mix(manifest, config)
    assert manifest["fineweb_edu"]["tokens_ingested"]["val"] == 500
