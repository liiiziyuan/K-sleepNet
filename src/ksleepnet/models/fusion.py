"""Cross-attention fusion: raw features as Query, statistical features as Key/Value.

At the epoch level both streams are single tokens, so nn.MultiheadAttention
reduces to a learned additive injection of the (projected) statistical anchor
into the raw representation -- exactly the 'knowledge anchor' semantics of the
paper. The same module is reused at fine-tuning time where the raw query is the
sequence of 10 epoch embeddings and K/V is the target epoch's stat embedding.

Candidate A1 (dual token): K/V becomes the 2-token stack [z_stat, anchor].
nn.MultiheadAttention's W_k/W_v are sequence-length independent, so pretrained
single-token fusion weights transfer unchanged; `forward(..., need_weights=True)`
additionally exposes the [B, Lq, Lkv] attention map -- the mass on the anchor
token is the per-sample anchoring strength alpha (logged per epoch, grouped by
true class, in train_log.jsonl).
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """(raw[B, Lq, d], stat[B, Lkv, d]) -> [B, Lq, d] with residual + LayerNorm.

    need_weights=True returns (out, attn_weights[B, Lq, Lkv]) — head-averaged
    softmax weights from nn.MultiheadAttention (output values are identical,
    only the weight map is computed in addition).
    """

    def __init__(self, d: int = 128, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(
        self, raw: torch.Tensor, stat: torch.Tensor, need_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        need_q = 3
        if raw.dim() == need_q - 1:
            raw = raw.unsqueeze(1)
        if stat.dim() == need_q - 1:
            stat = stat.unsqueeze(1)
        if need_weights:
            out, w = self.attn(raw, stat, stat, need_weights=True)
            return self.norm(raw + out), w
        out, _ = self.attn(raw, stat, stat, need_weights=False)
        return self.norm(raw + out)
