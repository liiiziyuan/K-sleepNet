"""Automatic Weighted Loss (AWL), Eq. 4 of the paper:

    L = sum_i 0.5 * exp(-s_i) * L_i + 0.5 * s_i,   s_i = log(sigma_i^2)
"""

from __future__ import annotations

import torch
from torch import nn


class AutomaticWeightedLoss(nn.Module):
    """Learnable task weights for {cls, supcon, adv} (num_tasks=3).

    A task may be None (e.g. SupCon on mixup/CutMix batches): it is skipped
    ENTIRELY — neither the 0.5*exp(-s)*L term nor the 0.5*s regularizer is
    added. Adding only 0.5*s would push that log-variance down every mixed
    batch and bias the weights of the active tasks.
    """

    def __init__(self, num_tasks: int = 3) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses: torch.Tensor | None) -> torch.Tensor:
        assert len(losses) == self.log_vars.numel(), "one loss per task weight"
        total: torch.Tensor | None = None
        for s, l in zip(self.log_vars, losses):
            if l is None:
                continue
            term = 0.5 * torch.exp(-s) * l + 0.5 * s
            total = term if total is None else total + term
        if total is None:
            raise ValueError("AutomaticWeightedLoss received no active loss (all tasks None)")
        return total
