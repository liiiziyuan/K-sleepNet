# K-SleepNet

**K-SleepNet: Knowledge-Enhanced Single-Channel Sleep Staging via Dual-Stream Multi-Task Learning**

[![DOI](https://img.shields.io/doi/10.1109/CSCWD68734.2026.11581617)](https://doi.org/10.1109/CSCWD68734.2026.11581617) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📄 Published in *Proceedings of the 2026 29th International Conference on
> Computer Supported Cooperative Work in Design (CSCWD)* —
> DOI: [10.1109/CSCWD68734.2026.11581617](https://doi.org/10.1109/CSCWD68734.2026.11581617).
>
> This repository is the **architecture-level reference implementation** of the paper.

## Overview

K-SleepNet performs single-channel EEG sleep staging with a dual-stream,
knowledge-enhanced design:

- **Dual-stream encoding** — a DeepSleepNet-style two-branch CNN encodes the raw
  30-s epoch, while an MLP encodes a **153-dim multi-scale statistical feature
  vector** (time / frequency / energy / nonlinear-entropy / peak features over
  {1, 3, 5}-epoch windows; paper Table I);
- **Cross-attention fusion** — the raw embedding (query) attends to the
  statistical knowledge embedding (key/value);
- **Multi-task pretraining (Phase 1)** — joint supervision with sleep-stage
  classification (CE), supervised contrastive learning (SupCon), and a
  domain-adversarial discriminator through a gradient reversal layer, combined
  under an **uncertainty-based adaptive loss weighting** (AWL, Eq. 4);
- **Sequence fine-tuning (Phase 2)** — a multi-scale pyramid ({1,2,4} centered
  max-pooling) plus a Transformer encoder over 10-epoch sequences, classifying
  the center epoch.

<div align="center">
  <img src="assets/architecture.png" width="95%" alt="K-SleepNet architecture"/>
  <p><em>Overall architecture of K-SleepNet (Fig. 1 of the paper).</em></p>
</div>

## Results (from the paper)

Evaluated on Sleep-EDF (79 subjects, Fpz-Cz), SHHS (5,793 subjects, C4-A1),
and a self-collected clinical dataset (50 subjects, C3-O1), against DeepSleepNet,
U-Time, AttnSleep, XSleepNet, SleepDG, Wafformer, and SleePyCo:

| Dataset | ACC | Macro-F1 |
|---|---|---|
| Sleep-EDF | **86.4** | **81.0** |
| SHHS | **88.2** | **82.0** |
| Self-collected | **86.9** | **80.7** |
| Average | **87.2** | **81.2** |

K-SleepNet outperforms the runner-up (SleePyCo) by 1.3% average accuracy and
1.3% average Macro-F1 across the three datasets (full comparison: Table II of
the paper).

## Repository Contents

```
src/ksleepnet/
├── models/
│   ├── encoders.py     # RawEncoder (two-branch CNN) + StatEncoder (153-dim MLP)
│   ├── fusion.py       # Cross-attention fusion (Q=raw, K/V=stat)
│   ├── grl.py          # Gradient reversal layer
│   ├── ksleepnet.py    # KSleepNetPretrain + KSleepNetFinetune + MultiScalePyramid
│   └── losses.py       # SupCon loss + CE (hard/soft labels)
├── train/
│   └── awl.py          # Automatic Weighted Loss (Eq. 4, learnable task weights)
└── data/
    └── features.py     # 153-dim multi-scale statistical features (paper Table I)
```

Quick check that the model and multi-task losses run:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F
from ksleepnet.models import KSleepNetPretrain
from ksleepnet.models.losses import ce_loss, supcon_loss
from ksleepnet.train import AutomaticWeightedLoss

m = KSleepNetPretrain()
out = m(torch.randn(4, 1, 3000), torch.randn(4, 153))
y = torch.randint(0, 5, (4,))
l_cls = ce_loss(out['logits'], y)
l_sup = supcon_loss(out['proj'], y)
l_adv = F.binary_cross_entropy_with_logits(out['domain_logit'], torch.randint(0, 2, (4,)).float())
awl = AutomaticWeightedLoss(num_tasks=3)
print('AWL total:', float(awl(l_cls, l_sup, l_adv)))
"
```

> **Note — architecture-level release.** This repository provides the model
> architecture and feature-extraction implementation as a reference for the
> paper. The training pipeline, hyperparameter configurations, and dataset
> split files are **not** part of this release, so the paper's result tables
> cannot be reproduced from this repository alone.

> **Implementation note.** The original experimental codebase was lost due to a
> server-storage expiration after the paper was accepted. This repository is a
> clean re-implementation of the architecture described in the paper, rebuilt
> afterwards from the manuscript and the surviving engineering records. It
> faithfully reflects the paper's model design, but it is not the exact code
> that produced the published result tables, and a few low-level engineering
> choices may differ from the original implementation.

## Features (`data/features.py`)

51 base features × {1, 3, 5}-epoch windows = 153 dims per epoch:

| Category | Count | Examples |
|---|---|---|
| Time domain | 14 | mean/std/quantiles/skew/kurtosis, first-diff stats |
| Frequency | 11 | abs/relative band power (δ θ α σ β), total power |
| Energy | 13 | signal energy/power/RMS, band energies & ratios |
| Nonlinear / entropy | 6 | spectral / permutation / sample entropy, Hjorth params |
| Peaks | 7 | count, height, prominence, density, zero-crossing rate |

Windows are computed on the concatenated signal (edge-replicated at recording
boundaries); permutation & sample entropy use numba-jitted kernels.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026ksleepnet,
  title     = {K-SleepNet: Knowledge-Enhanced Single-Channel Sleep Staging
               via Dual-Stream Multi-Task Learning},
  author    = {Yu, Qiuchen and Li, Ziyi and Wang, Qisong and Liu, Xinxin and
               Lei, Xiaochen and Zhang, Xurui and Gao, Guosong and Hou, Zhongjie and
               Bai, Hua and Li, Jiayue and Chen, Xiaohe},
  booktitle = {2026 29th International Conference on Computer Supported
               Cooperative Work in Design (CSCWD)},
  year      = {2026},
  pages     = {3646--3651},
  doi       = {10.1109/CSCWD68734.2026.11581617}
}
```

## Acknowledgements

- The raw-epoch encoder design follows the DeepSleepNet representation layer
  (Supratak et al., 2017).
- The supervised contrastive loss follows Khosla et al., 2020.
- The domain-adversarial branch uses the gradient reversal technique of
  Ganin & Lempitsky, 2015.
- The preprocessing protocol follows SleePyCo (Phyo et al.).

## License

To be determined.
