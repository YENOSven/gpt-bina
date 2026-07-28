import os

import torch

from columbina_pretrain import checkpoint as ckpt
from columbina_pretrain.model import CONFIG_TINY, build_model
from columbina_pretrain.train import make_optimizer


def _train_one_step(model, optimizer):
    device = next(model.parameters()).device
    x = torch.randint(0, CONFIG_TINY["vocab_size"], (2, 8), device=device)
    y = torch.randint(0, CONFIG_TINY["vocab_size"], (2, 8), device=device)
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, CONFIG_TINY["vocab_size"]), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def test_checkpoint_roundtrip_is_exact(tmp_path):
    model = build_model(CONFIG_TINY, device="cpu", seed=0)
    optimizer = make_optimizer(model)
    _train_one_step(model, optimizer)  # give the optimizer real (non-zero) momentum state to round-trip

    path = tmp_path / "ckpt.pt"
    ckpt.save_checkpoint(model, optimizer, step=3, global_token_cursor=12345, path=str(path),
                          wall_clock_seconds_trained=42.0, extra={"note": "test"})

    fresh_model = build_model(CONFIG_TINY, device="cpu", seed=999)  # different seed: must be fully overwritten by load
    fresh_optimizer = make_optimizer(fresh_model)
    state = ckpt.load_checkpoint(str(path), fresh_model, fresh_optimizer)

    assert state["step"] == 3
    assert state["global_token_cursor"] == 12345
    assert state["wall_clock_seconds_trained"] == 42.0
    assert state["extra"] == {"note": "test"}

    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} did not round-trip exactly"

    orig_opt_state = optimizer.state_dict()
    loaded_opt_state = fresh_optimizer.state_dict()
    assert orig_opt_state["param_groups"] == loaded_opt_state["param_groups"]
    for key in orig_opt_state["state"]:
        for buf_name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(orig_opt_state["state"][key][buf_name], loaded_opt_state["state"][key][buf_name])


def test_try_resume_returns_none_when_no_checkpoint_exists(tmp_path):
    model = build_model(CONFIG_TINY, device="cpu", seed=0)
    result = ckpt.try_resume(str(tmp_path / "does_not_exist.pt"), model)
    assert result is None


def test_save_checkpoint_leaves_no_tmp_file_on_disk(tmp_path):
    model = build_model(CONFIG_TINY, device="cpu", seed=0)
    optimizer = make_optimizer(model)
    path = tmp_path / "ckpt.pt"
    ckpt.save_checkpoint(model, optimizer, step=0, global_token_cursor=0, path=str(path))
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"atomic write left temp file(s) behind: {leftovers}"


def test_save_checkpoint_retries_through_transient_permission_errors(tmp_path, monkeypatch):
    """Regression test for a real failure hit running a local-fallback session inside a
    OneDrive-synced folder: os.replace() raised PermissionError (WinError 5) because OneDrive's
    sync client had the destination momentarily locked. Simulates that by making os.replace
    fail twice before succeeding, and asserts save_checkpoint rides it out via retry rather
    than losing the checkpoint (or leaving a stray .tmp file)."""
    model = build_model(CONFIG_TINY, device="cpu", seed=0)
    optimizer = make_optimizer(model)
    path = tmp_path / "ckpt.pt"

    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise PermissionError("[WinError 5] Access is denied (simulated OneDrive lock)")
        return real_replace(src, dst)

    monkeypatch.setattr(ckpt.os, "replace", flaky_replace)
    monkeypatch.setattr(ckpt, "_RENAME_RETRY_DELAYS", (0, 0, 0, 0, 0))  # skip real sleeps in the test

    ckpt.save_checkpoint(model, optimizer, step=5, global_token_cursor=1, path=str(path))

    assert calls["count"] == 3, "expected exactly 2 failures then a success"
    assert path.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_resume_works_on_cuda(tmp_path):
    """Regression test for a real bug: load_checkpoint used to forward a map_location="cuda"
    straight to torch.load, which moved the RNG state ByteTensor onto the GPU during
    deserialization -- torch.set_rng_state() requires a CPU tensor specifically and raised
    TypeError on a CUDA one. Every earlier test either never resumed a real checkpoint at all,
    or only ever did so with device="cpu", so this stayed hidden until a real GPU resume was
    actually exercised. Skips (doesn't fail) on a machine with no CUDA GPU."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA GPU on this machine")

    model = build_model(CONFIG_TINY, device="cuda", seed=0)
    optimizer = make_optimizer(model)
    _train_one_step(model, optimizer)

    path = tmp_path / "ckpt.pt"
    ckpt.save_checkpoint(model, optimizer, step=1, global_token_cursor=99, path=str(path))

    fresh_model = build_model(CONFIG_TINY, device="cuda", seed=999)
    fresh_optimizer = make_optimizer(fresh_model)
    state = ckpt.load_checkpoint(str(path), fresh_model, fresh_optimizer)  # must not raise

    assert state["step"] == 1
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        assert n1 == n2
        assert p2.is_cuda, f"parameter {n2} ended up off-GPU after loading onto a CUDA model"
        assert torch.equal(p1, p2), f"parameter {n1} did not round-trip exactly"
