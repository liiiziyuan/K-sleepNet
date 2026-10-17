"""Losses: supervised contrastive (SupCon) + cross-entropy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020), Eq. 2 of the paper.

    Args:
        z: L2-normalized projections [B, d] (e.g. from the SupCon head).
        labels: int64 [B].
        temperature: tau.
    Returns scalar loss; None when the batch has no positive pair.
    """
    if z.shape[0] < 2:
        return None
    # CUDA autocast re-casts fp32 matmul inputs to fp16, so an entry-level
    # .float() is a no-op here; disable autocast for the whole computation
    # (fp16 normalize eps + exp/log chains are not trustworthy, losses.py must
    # own its numerics).
    with torch.autocast(device_type=z.device.type, enabled=False):
        return _supcon_loss_fp32(z.float(), labels, temperature)


def _supcon_loss_fp32(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature  # [B, B]
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

    # numerical stability: subtract row max
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return None
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0,
            weight: torch.Tensor | None = None) -> torch.Tensor:
    """CE supporting both hard labels [B] and soft targets [B, C] (mixup/CutMix).

    `weight` [C]: optional per-class weight. The hard-label path uses it as a
    per-sample multiplier on the unreduced CE, then takes the plain batch mean,
    so a mean-1 weight vector leaves the loss scale unchanged. Default None =
    unweighted CE. Soft targets (mixup) take the label's EXPECTED weight, which
    degrades to the same per-sample weight for a one-hot target.
    """
    if weight is None:
        if targets.dim() > 1:
            return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    # fp32 weights on the logits' device: the reduction happens outside the
    # autocast region's dtype, and the caller may keep the vector on the CPU
    w = weight.to(device=logits.device, dtype=torch.float32)
    if targets.dim() > 1:
        t = targets.float()
        ce = -(t * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
        return (ce * (t * w.unsqueeze(0)).sum(dim=1)).mean()
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing,
                         reduction="none").float()
    return (ce * w[targets]).mean()
