"""Proves data_pipeline.py's resumable ingestion against a REAL HuggingFace streaming source
(not a mock) -- deliberately interrupts ingestion partway through, reloads the manifest from
disk as a fresh dict (simulating a genuinely separate session), and confirms it resumes
correctly through all three phases (val -> test -> train), then combines the result into real,
decodable corpus files. Costs a small, real HF Hub API call budget each run -- if you hit a 429
rate-limit error, that's HuggingFace's API throttling repeated dev-machine requests, not a bug
here; wait a few minutes and retry.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pretrain/

from bina_pretrain import data_manifest as dm
from bina_pretrain import data_pipeline as dp
from bina_pretrain.tokenizer import EOT, decode

SCRATCH = tempfile.mkdtemp(prefix="data_pipeline_verify_")
SHARD_DIR = os.path.join(SCRATCH, "shards")
MANIFEST_PATH = os.path.join(SCRATCH, "manifest.json")


def main():
    print("scratch:", SCRATCH)
    manifest = {}
    dm.get_or_create_entry(
        manifest, "fineweb_edu_test", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
        target_train_tokens=40_000, val_fraction=0.05, test_fraction=0.05,
    )
    entry = manifest["fineweb_edu_test"]
    print(f"targets: {entry['targets']}")

    total = 0
    call_num = 0
    while not dm.is_complete(manifest["fineweb_edu_test"]):
        call_num += 1
        # only the first call is deliberately capped small (to prove resume happens at all);
        # every call after runs to completion in one shot, to keep this script's HF Hub API
        # usage low on repeat runs
        cap = 3000 if call_num == 1 else None
        tokens_this_call = dp.ingest_source(manifest, "fineweb_edu_test", SHARD_DIR, tokens_per_shard=5000, max_tokens_this_call=cap)
        dm.save(manifest, MANIFEST_PATH)
        total += tokens_this_call
        entry = manifest["fineweb_edu_test"]
        print(f"call {call_num}: +{tokens_this_call} tokens, phase now '{entry['phase']}', "
              f"ingested={entry['tokens_ingested']}, status={entry['status']}")
        manifest = dm.load(MANIFEST_PATH)  # simulate a genuinely fresh session
        if call_num > 40:
            raise RuntimeError("ingest didn't converge -- infinite loop guard tripped")

    print(f"\ntotal tokens ingested across {call_num} separate resumed calls: {total}")
    entry = manifest["fineweb_edu_test"]
    # >= not == : the final document of a phase is kept whole (never split mid-document), so a
    # phase's actual token count can slightly exceed its target -- expected, not a bug.
    for phase in ("val", "test", "train"):
        ingested, target = entry["tokens_ingested"][phase], entry["targets"][phase]
        assert ingested >= target, f"{phase}: {ingested} < {target}"
        overshoot = ingested - target
        print(f"{phase}: target={target}, ingested={ingested}, overshoot={overshoot} ({100 * overshoot / target:.1f}%)")
        # bounded by "at most one extra document's worth" (FineWeb-Edu docs run up to a few
        # thousand tokens), not a percentage of target -- meaningless until target is large
        # relative to one document, which these small test targets deliberately aren't
        assert overshoot < 5000, f"{phase} overshoot looks like more than one document: {overshoot}"
    assert dm.is_complete(entry)
    n_shards = sum(len(entry["shards_written"][p]) for p in ("val", "test", "train"))
    print(f"shards written: {n_shards} (val={len(entry['shards_written']['val'])}, "
          f"test={len(entry['shards_written']['test'])}, train={len(entry['shards_written']['train'])})")

    for phase in ("train", "val", "test"):
        out_path = os.path.join(SCRATCH, f"corpus_{phase}.bin")
        n_tokens = dp.combine_shards_to_corpus(manifest, ["fineweb_edu_test"], SHARD_DIR, phase, out_path)
        tokens = np.memmap(out_path, dtype=np.uint16, mode="r")
        assert len(tokens) == n_tokens == entry["tokens_ingested"][phase]
        n_eot = int((tokens == EOT).sum())
        print(f"\n{phase}: {n_tokens} tokens, {n_eot} EOT-separated documents")
        first_doc_end = int(np.argmax(tokens == EOT))
        print(f"  first doc preview: {decode(tokens[:first_doc_end].tolist())[:150]!r}")

    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\nPASS: real FineWeb-Edu data ingested resumably across separate manifest reloads, "
          "phases (val->test->train) progressed correctly, and shards combined into real, "
          "decodable train/val/test corpus files.")


if __name__ == "__main__":
    main()
