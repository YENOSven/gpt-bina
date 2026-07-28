import pytest
import torch

from columbina_pretrain.model import CONFIG_124M, build_model


def test_checkpointed_forward_matches_normal_forward_exactly():
    """No dropout anywhere in this model, so recomputing a block during backward must
    produce bit-identical activations to computing it once -- gradient checkpointing is a
    pure memory/compute tradeoff here, never a correctness one."""
    torch.manual_seed(0)
    x = torch.randint(0, CONFIG_124M["vocab_size"], (2, 16))

    plain = build_model(CONFIG_124M, device="cpu", seed=1, gradient_checkpointing=False)
    checkpointed = build_model(CONFIG_124M, device="cpu", seed=1, gradient_checkpointing=True)

    plain.train()
    checkpointed.train()
    out_plain = plain(x)
    out_checkpointed = checkpointed(x)
    assert torch.equal(out_plain, out_checkpointed)

    out_plain.sum().backward()
    out_checkpointed.sum().backward()
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1.grad, p2.grad), f"gradient mismatch at {n1}"


def test_gradient_checkpointing_off_by_default_and_inert_in_eval_mode():
    model = build_model(CONFIG_124M, device="cpu", seed=0)
    assert model.gradient_checkpointing is False

    model = build_model(CONFIG_124M, device="cpu", seed=0, gradient_checkpointing=True)
    model.eval()
    x = torch.randint(0, CONFIG_124M["vocab_size"], (2, 16))
    with torch.no_grad():
        model(x)  # would raise if checkpoint.checkpoint() ran under no_grad/eval; forward() guards on self.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory-reduction claim is only meaningful on real CUDA")
def test_gradient_checkpointing_reduces_peak_cuda_memory():
    from columbina_pretrain.model import CONFIG_406M

    def peak_after_one_step(gradient_checkpointing):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(CONFIG_406M, device="cuda", seed=0, gradient_checkpointing=gradient_checkpointing)
        model.train()
        x = torch.randint(0, CONFIG_406M["vocab_size"], (1, CONFIG_406M["max_seq_len"]), device="cuda")
        y = torch.randint(0, CONFIG_406M["vocab_size"], (1, CONFIG_406M["max_seq_len"]), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, CONFIG_406M["vocab_size"]), y.view(-1))
        loss.backward()
        peak = torch.cuda.max_memory_allocated()
        del model, x, y, logits, loss
        torch.cuda.empty_cache()
        return peak

    peak_without = peak_after_one_step(False)
    peak_with = peak_after_one_step(True)
    assert peak_with < peak_without, (
        f"gradient checkpointing should reduce peak memory: without={peak_without / 1e9:.2f}GB "
        f"with={peak_with / 1e9:.2f}GB"
    )
