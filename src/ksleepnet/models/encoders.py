"""Raw-EEG CNN encoder (DeepSleepNet-style two branches) + statistical MLP encoder.

RawEncoder design notes (kernel/stride choices follow the DeepSleepNet
representation layer):
    - small-filter branch: Conv(1->32, k=fs/2, s=fs/16) -> Conv(32->64, k=8)
      -> MaxPool(2) -> Dropout -> Conv(64->128, k=6) -> Conv(128->256, k=6)
    - large-filter branch: Conv(1->128, k=2*fs (large scale for delta/theta),
      s=fs/16) -> Conv(128->256, k=8) -> MaxPool(2) -> Dropout
    - each branch adaptive-avg-pooled to `pool_out` steps -> concat -> FC
      1024 -> FC 128.
At fs=100, one 30s epoch has 3000 samples.
"""

from __future__ import annotations

import torch
from torch import nn


class RawEncoder(nn.Module):
    """Two-branch CNN for one 30s epoch: [B, 1, T] -> [B, d]."""

    def __init__(
        self,
        fs: int = 100,
        d: int = 128,
        pool_out: int = 64,
        dropout: float = 0.5,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        s_first = max(1, fs // 16)
        k_first = fs // 2
        k_large = fs * 2  # 2 s window: captures delta/theta scale waves
        self.small = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=k_first, stride=s_first),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=6, stride=1, padding=3),
            nn.ReLU(),
        )
        self.large = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=k_large, stride=s_first),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        self.pool_out = pool_out
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 256 * pool_out, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, d),
        )

    def _adaptive_pool(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, C, L] -> [B, C, pool_out]
        if h.shape[-1] != self.pool_out:
            h = nn.functional.adaptive_avg_pool1d(h, self.pool_out)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, T] or [B, T] -> [B, d]."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        a0 = self.small(x)                       # [B, 256, Ls] (~0.12 s/pos)
        b0 = self.large(x)                       # [B, 256, Ll]
        a = self._adaptive_pool(a0)              # [B, 256, pool_out]
        b = self._adaptive_pool(b0)              # [B, 256, pool_out]
        return self.head(torch.cat([a, b], dim=1))


class StatEncoder(nn.Module):
    """MLP for the 153-dim statistical feature vector: [B, 153] -> [B, d]."""

    def __init__(self, in_dim: int = 153, hidden: int = 256, d: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
