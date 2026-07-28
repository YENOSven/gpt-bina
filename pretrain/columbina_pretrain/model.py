import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def split_heads(t, num_heads):
    *batch_dims, seq_len, d_model = t.shape
    d_k = d_model // num_heads
    return t.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)


def merge_heads(t):
    *batch_dims, num_heads, seq_len, d_k = t.shape
    return t.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, num_heads * d_k)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Qh, Kh, Vh = split_heads(Q, self.num_heads), split_heads(K, self.num_heads), split_heads(V, self.num_heads)
        # same math as the hand-rolled Q@K^T/sqrt(d_k)+mask+softmax version this replaced,
        # but dispatches to a fused kernel (flash-attention on supported GPUs) instead of
        # materializing the full attention-weights matrix
        out = F.scaled_dot_product_attention(Qh, Kh, Vh, is_causal=causal)
        return self.Wo(merge_heads(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), causal=True)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, d_ff=None,
                 gradient_checkpointing=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.Wo.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        seq_len = idx.shape[-1]
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(positions)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # recomputes each block's activations during backward instead of keeping all
                # num_layers of them resident at once -- trades ~30% more compute for a large
                # cut in peak memory, the difference between fitting on an 8GB card and not.
                # No dropout anywhere in this model, so recomputation is exactly deterministic;
                # nothing here depends on preserve_rng_state.
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# fast local iteration / infra proofs only (CPU-friendly, seconds not minutes)
CONFIG_TINY = dict(vocab_size=50257, d_model=64, num_heads=4, num_layers=2, max_seq_len=64)

# GPT-2-small (module_31's smoke-test scale) — kept for cheap local tests
CONFIG_124M = dict(vocab_size=50257, d_model=768, num_heads=12, num_layers=12, max_seq_len=1024)

# the real Columbina target size (~406M params) — see pretrain/configs/model_400m.yaml
CONFIG_406M = dict(vocab_size=50257, d_model=1024, num_heads=16, num_layers=28, max_seq_len=2048)


def build_model(config, device, seed=42, gradient_checkpointing=False):
    torch.manual_seed(seed)
    return GPT(**config, gradient_checkpointing=gradient_checkpointing).to(device)
