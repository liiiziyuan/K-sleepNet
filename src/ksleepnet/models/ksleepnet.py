"""K-SleepNet models: epoch-level multi-task pretraining + sequence-level fine-tuning.

Pretrain (Phase 1, epoch level):
    raw [B,1,3000] --RawEncoder--> z_r [B,128] --\\
                                                  cross-attn (Q=raw, K/V=stat)
    stat [B,153]   --StatEncoder--> z_s [B,128] -/   --> fused [B,128]
    heads: cls 128->5 | SupCon proj 128->64 (L2-normed) | GRL -> domain disc 128->64->1

Fine-tune (Phase 2, sequence of 10 epochs, target = center index):
    raw [B,10,1,3000] -> shared RawEncoder -> [B,10,128]
    center position fused with the target epoch's stat embedding; other positions
    keep their raw-only representation.
    multi-scale pyramid: centered MaxPool over the time axis at scales {1,2,4}
    -> concat [B,10,384] -> TransformerEncoder(2 layers, 4 heads) -> center
    position -> cls 384->5. CE only (per paper).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import RawEncoder, StatEncoder
from .fusion import CrossAttentionFusion
from .grl import GradientReversal


class SupConHead(nn.Module):
    def __init__(self, d: int = 128, out: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=1)


class DomainDiscriminator(nn.Module):
    """128 -> 64 -> n_domains with GRL in front.

    n_domains=1 reproduces the binary BCE head (Eq. 3, single output);
    n_domains>1 is the multi-class subject-ID form of the same objective
    (dj = subject domain label) trained with cross-entropy.
    """

    def __init__(self, d: int = 128, hidden: int = 64, gamma: float = 10.0,
                 n_domains: int = 1, lamb_max: float = 1.0) -> None:
        super().__init__()
        self.grl = GradientReversal(gamma, lamb_max=lamb_max)
        # LayerNorm in front of the GRL bounds the per-sample adversarial
        # gradient that reaches the shared encoder (arms-race control)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, max(1, int(n_domains))),
        )
        self.in_norm = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(self.grl(self.in_norm(z)))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class KSleepNetPretrain(nn.Module):
    """Epoch-level dual-stream model with cls + SupCon + adversarial heads."""

    def __init__(
        self,
        n_classes: int = 5,
        d: int = 128,
        stat_dim: int = 153,
        fs: int = 100,
        attn_heads: int = 4,
        pool_out: int = 64,
        grl_gamma: float = 10.0,
        supcon_dim: int = 64,
        n_domains: int = 1,
        grl_lamb_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_encoder = RawEncoder(fs=fs, d=d, pool_out=pool_out)
        self.stat_encoder = StatEncoder(in_dim=stat_dim, d=d)
        self.fusion = CrossAttentionFusion(d=d, heads=attn_heads)
        self.cls_head = nn.Linear(d, n_classes)
        self.supcon_head = SupConHead(d=d, out=supcon_dim)
        self.domain_disc = DomainDiscriminator(d=d, gamma=grl_gamma, n_domains=n_domains,
                                               lamb_max=grl_lamb_max)

    def forward(self, raw: torch.Tensor, stat: torch.Tensor) -> dict[str, torch.Tensor]:
        """raw [B,1,T]; stat [B,153] -> dict(logits, proj, domain_logit,
        z_raw, z_stat, fused)."""
        z_raw = self.raw_encoder(raw)
        z_stat = self.stat_encoder(stat)
        fused = self.fusion(z_raw, z_stat).squeeze(1)
        logits = self.cls_head(fused)
        return {
            "logits": logits,
            "proj": self.supcon_head(fused),
            "domain_logit": self.domain_disc(fused),
            "z_raw": z_raw,
            "z_stat": z_stat,
            "fused": fused,
        }


class MultiScalePyramid(nn.Module):
    """Centered MaxPool over the epoch axis at scales {1,2,4} -> concat [B,L,3C]."""

    def __init__(self, scales: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.scales = scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> [B, L, C * len(scales)]"""
        feats = []
        for s in self.scales:
            if s == 1:
                feats.append(x)
                continue
            xp = x.permute(0, 2, 1)  # [B, C, L]
            # asymmetric replicate padding keeps output length == L for even s
            xp = F.pad(xp, (s // 2, s - 1 - s // 2), mode="replicate")
            feats.append(F.max_pool1d(xp, kernel_size=s, stride=1).permute(0, 2, 1))
        return torch.cat(feats, dim=2)


class KSleepNetFinetune(nn.Module):
    """Sequence-level classifier on top of the pretrained encoders."""

    def __init__(
        self,
        n_classes: int = 5,
        d: int = 128,
        stat_dim: int = 153,
        fs: int = 100,
        seq_len: int = 10,
        attn_heads: int = 4,
        pool_out: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff: int = 512,
        transformer_dropout: float = 0.1,
        pyramid_scales: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.center = seq_len // 2
        self.raw_encoder = RawEncoder(fs=fs, d=d, pool_out=pool_out)
        self.stat_encoder = StatEncoder(in_dim=stat_dim, d=d)
        self.fusion = CrossAttentionFusion(d=d, heads=attn_heads)
        d_model = d * len(pyramid_scales)
        self.pyramid = MultiScalePyramid(pyramid_scales)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=transformer_heads, dim_feedforward=transformer_ff,
            dropout=transformer_dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.cls_head = nn.Linear(d_model, n_classes)

    def forward(self, raw_seq: torch.Tensor, stat: torch.Tensor) -> dict[str, torch.Tensor]:
        """raw_seq [B, L, 1, T]; stat [B,153] (target epoch only)
        -> dict(logits [B, C], sequence [B, L, d_model])."""
        B, L = raw_seq.shape[0], raw_seq.shape[1]
        z_flat = self.raw_encoder(raw_seq.reshape(B * L, *raw_seq.shape[2:]))
        z = z_flat.reshape(B, L, -1)
        # center: inject the statistical anchor via cross-attention
        z_stat = self.stat_encoder(stat)
        center_fused = self.fusion(z[:, self.center], z_stat).squeeze(1)
        z = torch.cat(
            [z[:, : self.center], center_fused.unsqueeze(1), z[:, self.center + 1 :]], dim=1
        )
        seq_in = self.pyramid(z)  # [B, L, d_model]
        h = self.transformer(seq_in)  # [B, L, d_model]
        logits = self.cls_head(h[:, self.center])
        return {"logits": logits, "sequence": h}
