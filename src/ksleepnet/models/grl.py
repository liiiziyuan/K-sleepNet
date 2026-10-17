"""Gradient Reversal Layer (GRL) with DANN lambda schedule."""

from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lamb: float) -> torch.Tensor:
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lamb * grad_output, None


class GradientReversal(nn.Module):
    """Identity in forward pass, negated gradient (scaled by lambda) backward.

    Lambda follows the DANN schedule (Ganin et al., 2016):
        lam(p) = 2 / (1 + exp(-gamma * p)) - 1,  p in [0, 1)
    which rises from 0 to 1 as training progresses (per spec: 'lambda rises
    gradually from 0 to 1, schedule lam = 2/(1+exp(-10p))', with the -1 offset
    required to actually start at 0).
    """

    def __init__(self, gamma: float = 10.0, lamb_max: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        # cap on the adversarial gradient scale: the subject-ID adversarial is a
        # hard (non-saturating) task whose reversed gradients collapse the shared
        # encoder once lambda passes ~0.2 at full-scale step counts (observed 3x);
        # a small lamb_max keeps the domain-confusion regularizer but bounded.
        self.lamb_max = float(lamb_max)
        self.lamb: float = 0.0

    def set_progress(self, p: float) -> None:
        """Update lambda from global training progress p in [0, 1]."""
        p = min(max(float(p), 0.0), 1.0)
        self.lamb = min(2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0, self.lamb_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lamb)
