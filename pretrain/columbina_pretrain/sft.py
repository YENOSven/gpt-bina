import json

import torch
import torch.nn.functional as F

from columbina_pretrain.tokenizer import COLUMBINA_MARKER, EOT, SYSTEM_MARKER, USER_MARKER, encode

_ROLE_MARKERS = {"system": SYSTEM_MARKER, "user": USER_MARKER, "assistant": COLUMBINA_MARKER}


def build_example(messages):
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -- the sibling
    Columbina project's exact JSONL shape. Returns (token_ids, loss_mask) of equal length;
    mask[i]=1 iff token_ids[i] is content belonging to an 'assistant' (Columbina) turn, the
    only tokens the model is trained to predict. Role markers themselves are never loss-active
    -- they're part of what the inference harness supplies, not something the model generates
    -- same prompt/response split module 32 established, extended from one pair to N turns."""
    token_ids = []
    loss_mask = []
    for msg in messages:
        marker_ids = encode(_ROLE_MARKERS[msg["role"]])
        content_ids = encode(msg["content"])
        token_ids.extend(marker_ids)
        loss_mask.extend([0] * len(marker_ids))
        token_ids.extend(content_ids)
        loss_mask.extend([1 if msg["role"] == "assistant" else 0] * len(content_ids))
    token_ids.append(EOT)
    loss_mask.append(0)
    return token_ids, loss_mask


def build_batch(examples, pad_id=EOT):
    """examples: list of {"messages": [...]}. Right-pads to the batch's longest example --
    safe with a causal model since later padding can never affect an earlier real token's
    logits, and padded positions carry mask=0 so they're never trained on either way."""
    per_example = [build_example(ex["messages"]) for ex in examples]
    max_len = max(len(ids) for ids, _ in per_example)
    seq_len = max_len - 1
    batch_size = len(examples)

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    target_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, seq_len))
    for i, (ids, mask) in enumerate(per_example):
        length = len(ids) - 1
        input_ids[i, :length] = torch.tensor(ids[:-1])
        target_ids[i, :length] = torch.tensor(ids[1:])
        loss_mask[i, :length] = torch.tensor(mask[1:], dtype=torch.float)
    return input_ids, target_ids, loss_mask


def masked_loss(logits, targets, mask, vocab_size):
    losses = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="none")
    losses = losses.view(targets.shape)
    return (losses * mask).sum() / mask.sum().clamp(min=1)


def load_jsonl_examples(path):
    """Loads the sibling Columbina project's {"messages": [...]} JSONL format directly --
    same shape reused as-is, no reformatting needed at the file level."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
