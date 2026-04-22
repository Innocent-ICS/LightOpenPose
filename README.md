# Lightweight OpenPose — Re-implementation

**Paper:** Osokin, D. (2019). *Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose.* arXiv:1811.12004

**Author:** Innocent Farai Chikwanda
**GPU:** NVIDIA L4 (24 GB VRAM) via Google Colab
**Dataset:** COCO 2017 keypoint benchmark

---

## Overview

This repository is a clean, modular re-implementation of **Lightweight OpenPose**. The original paper demonstrates that a MobileNetV1 backbone with a small multi-stage prediction head can approach full VGG-19 OpenPose accuracy at a fraction of the compute cost, reaching ~40 AP on COCO while running in real-time on CPU.

### Why Lightweight OpenPose?

I originally planned to replicate the full VGG-19 OpenPose model (Cao et al., 2019), but the 40+ GB VRAM requirement for a faithful batch size made this infeasible on a single Colab L4.  Lightweight OpenPose targets the same COCO benchmark with a MobileNetV1 backbone that trains comfortably within 24 GB, making it an ideal pivot that still exercises every key contribution of the original paper (Part Affinity Fields, deep supervision, multi-stage refinement).

---

## Repository Structure

```
.
├── train.py            # Main training entry point
├── evaluate.py         # Standalone COCO AP evaluation
├── infer.py            # Inference visualisation on image(s)
├── setup_colab.py      # One-shot Colab environment bootstrap
├── requirements.txt    # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py       # All hyperparameters, paths, skeleton topology
│   ├── model.py        # MobileNetV1 + CPM + multi-stage heads
│   ├── dataset.py      # COCO loader, target generation, augmentation
│   ├── loss.py         # Masked MSE with deep supervision
│   ├── train.py        # Training loop, optimizer, checkpoint management
│   ├── evaluate.py     # COCO OKS AP evaluation
│   └── visualise.py    # Inference visualisation utilities
```

---

## Setup

### 1. Google Colab (recommended)

```bash
# Step 1 – clone the repo
git clone https://github.com/Innocent-ICS/LightOpenPose.git
cd LightOpenPose

# Step 2 – bootstrap environment (installs deps, mounts Drive, downloads COCO)
python setup_colab.py

# Step 3 – start training
python train.py
```

All data, checkpoints, and logs are persisted to Google Drive under
`MyDrive/Colab Notebooks/OpenPose/`.  Override the root path:

```bash
OPENPOSE_ROOT=/your/path python train.py
```

### 2. Local environment

```bash
pip install -r requirements.txt
# Edit DRIVE_ROOT in src/config.py to point to your local data directory
# Ensure COCO 2017 train2017/, val2017/, and annotations/ are present
python train.py
```

---

## Reproducibility

All random seeds are fixed in `train.py` via `seed_everything(CFG["seed"])` which sets `random`, `numpy`, and `torch` seeds to **42**.

```python
# src/config.py
CFG = {
    ...
    "seed": 42,
}
```

Running `python train.py` twice on the same hardware will produce identical results.

---

## Training

```bash
python train.py
```

**Key hyperparameters** (all in `src/config.py`):

| Parameter | Value |
|---|---|
| Input resolution | 368 × 368 |
| Backbone | MobileNetV1 (truncated, stride 8) |
| Refinement stages | 1 (Phase 1) |
| Batch size | 32 |
| Base LR | 4 × 10⁻⁵ |
| LR milestones | [100, 200, 260] epochs |
| LR decay | × 0.333 at each milestone |
| Total epochs | 280 |
| Warmup | 5 epochs |
| AMP | ✓ |
| Gradient clipping | max norm = 5 |

**Training phases** (set `PHASE` in `src/config.py`):

| Phase | Description |
|---|---|
| 1 | MobileNet backbone + 1 refinement stage, full COCO |
| 2 | Add refinement stages, fine-tune |
| 3 | Full 3 refinement stages, full schedule |

Checkpoints are saved to `CKPT_DIR/latest.pt` every epoch and `CKPT_DIR/best.pt` whenever validation AP improves.  The loop auto-resumes from `latest.pt` on restart.

---

## Evaluation

```bash
# Fast evaluation on 1000 val images (~3 min on L4)
python evaluate.py --max_imgs 1000

# Full COCO val2017 evaluation (~20 min on L4)
python evaluate.py

# Evaluate a specific checkpoint
python evaluate.py --ckpt /path/to/checkpoint.pt
```

**Target metric:** COCO OKS mAP (AP @ 0.50:0.95)

| Model | Backbone | Stages | AP (paper) | AP (ours) |
|---|---|---|---|---|
| OpenPose | VGG-19 | — | 65.3 | — (infeasible on L4) |
| Lightweight OpenPose | MobileNetV1 | 3 | ~40.0 | — |
| **This repo** | MobileNetV1 | 1 | — | **36.7** |

---

## Inference Visualisation

```bash
# Single image
python infer.py --input path/to/image.jpg --output results/

# Directory of images
python infer.py --input path/to/images/ --output results/

# Custom threshold
python infer.py --input image.jpg --threshold 0.15
```

---

## Bugs Fixed vs. Original Notebook

| # | Bug | Fix |
|---|---|---|
| 1 | `num_keypoints=18` included paper neck joint absent from COCO GT | Set `num_keypoints=17` (COCO standard) |
| 2 | `COCO_PAIRS` used paper 18-kp indices, mismatching COCO 17-kp GT | Redefined for 17-kp skeleton (16 limbs) |
| 3 | Eval coordinate scaling ignored `val_preprocess` padding offset | Correct inverse: feature-map → padded input → original |
| 4 | PAF skeleton assembly produced 0 persons at low PAF magnitude | Eval uses per-GT-bbox peak extraction; no PAF assembly |
| 5 | `GradScaler`/`autocast` used deprecated `torch.cuda.amp` import | Use `torch.amp` throughout |
| 6 | Val subset 250 images too noisy | Increased to 1000 images |
| 7 | `best.pt` never written during training | AP eval every 10 epochs inline, best saved on improvement |

---

## Citation

```bibtex
@article{osokin2018lightweight,
  title   = {Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
  author  = {Osokin, Daniil},
  journal = {arXiv preprint arXiv:1811.12004},
  year    = {2018}
}

@inproceedings{cao2019openpose,
  title     = {OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields},
  author    = {Cao, Zhe and Hidalgo, Gines and Simon, Tomas and Wei, Shih-En and Sheikh, Yaser},
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2019}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
