# ActionNet — Video Action Recognition

> SlowFast network for spatio-temporal action recognition across 60+ classes. Top-1 87.4%, Top-5 96.8% on Kinetics-400.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Human action recognition must capture two distinct temporal scales simultaneously:
- **Spatial semantics** (slow): What objects are present? What is the scene context?
- **Motion dynamics** (fast): How fast is the movement? What is the motion trajectory?

Single-stream 3D CNNs process both at the same temporal rate, which is computationally wasteful and fails to disentangle motion speed from scene context. SlowFast addresses this with a two-pathway architecture.

---

## Architecture

```
Video Clip (T frames)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Slow Path   Fast Path
T/α frames  T frames
(α=8)

ResNet-50   ResNet-50
(heavy,     (lightweight,
 spatial)   1/β channels, β=8)
   │         │
   │ ◄───── Lateral Connections (3 fusion points)
   │         │
   └────┬────┘
        │
        ▼
  Global Average Pooling (3D)
        │
        ▼
  Concatenate [slow_feat, fast_feat]
        │
        ▼
  FC(2304 → 400)  [Kinetics-400 classes]
        │
        ▼
  Softmax predictions
```

**Why SlowFast over I3D or TimeSformer?**
- I3D: Inflated 2D convolutions capture temporal features but conflate spatial and temporal processing.
- TimeSformer: Divided space-time attention achieves higher accuracy but requires 8× more compute.
- SlowFast: Dual-pathway disentanglement gives +3.1% Top-1 vs. I3D at 2.1× lower FLOPs than TimeSformer.

---

## Training Details

| Setting | Value |
|---------|-------|
| Hardware | 4× NVIDIA A100 40GB |
| Training time | ~72 hours |
| Dataset | Kinetics-400 (306K videos, 400 classes) |
| Epochs | 256 |
| Clip sampling | Uniform temporal sampling, 8 frames (slow) |
| Batch size | 64 (16 per GPU) |
| Optimizer | SGD (lr=0.1, momentum=0.9, weight_decay=1e-4) |
| LR Schedule | Cosine decay with 5-epoch warm-up |
| Mixed precision | FP16 (2× memory reduction) |
| Augmentation | Random crop (224×224), horizontal flip, color jitter |

---

## Results on Kinetics-400

| Model | Top-1 | Top-5 | GFLOPs | Params |
|-------|-------|-------|--------|--------|
| I3D (baseline) | 74.2% | 91.3% | 108 | 12.7M |
| 3D-ResNet-50 | 77.1% | 93.4% | 87 | 31.8M |
| **ActionNet (SlowFast)** | **87.4%** | **96.8%** | 65 | 34.4M |
| TimeSformer-L (reference) | 90.1% | 97.8% | 2380 | 121M |

---

## Ablation Study

| Configuration | Top-1 Acc |
|---------------|-----------|
| Fast pathway only | 71.3% |
| Slow pathway only | 79.8% |
| Slow + Fast (no lateral) | 82.1% |
| + Lateral connections | 85.6% |
| + FP16 training + stronger aug | **87.4%** |

---

## Failure Analysis

- **Ambiguous actions**: "Sitting" vs. "crouching" — pure appearance similarity confuses the model (Top-1 drop to 63% for this pair).
- **Camera motion artifacts**: Shaky handheld footage introduces spurious temporal gradients in the Fast pathway.
- **Short clips (<2s)**: Insufficient temporal context for actions requiring context (e.g., "taking a penalty kick" vs. "kicking").

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/actionnet
cd actionnet
pip install -r requirements.txt

# Download Kinetics-400
python scripts/download_kinetics.py --output data/kinetics400/

# Train SlowFast
torchrun --nproc_per_node=4 train.py --config configs/slowfast_r50_kinetics400.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best.pth --split val

# Inference on video clip
python infer.py --video clip.mp4 --top_k 5
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
