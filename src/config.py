# src/config.py
"""
Central configuration module.
All paths, hyperparameters, and skeleton definitions live here.
Edit PHASE and DRIVE_ROOT as needed before running.
"""

import os
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Training phase  (1 = initial, 2 = more refinement stages, 3 = full)
# ---------------------------------------------------------------------------
PHASE = 1

# ---------------------------------------------------------------------------
# Paths  (override with env-vars for portability)
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path(os.environ.get(
    "OPENPOSE_ROOT",
    "/content/drive/MyDrive/Colab Notebooks/OpenPose"
))

COCO_DIR        = DRIVE_ROOT / "coco"
CKPT_DIR        = DRIVE_ROOT / "checkpoints" / f"phase{PHASE}"
LOG_DIR         = DRIVE_ROOT / "logs"
PREP_DIR        = DRIVE_ROOT / "prepared"
MOBILENET_W     = DRIVE_ROOT / "mobilenet_sgd_68.848.pth.tar"
TRAIN_ANN       = COCO_DIR   / "annotations/person_keypoints_train2017.json"
VAL_ANN         = COCO_DIR   / "annotations/person_keypoints_val2017.json"
TRAIN_IMG       = COCO_DIR   / "train2017"
VAL_IMG         = COCO_DIR   / "val2017"
PREP_TRAIN_PKL  = PREP_DIR   / "train_labels.pkl"
VAL_SUBSET_JSON = PREP_DIR   / "val_subset_1000.json"

for _d in [DRIVE_ROOT, COCO_DIR, CKPT_DIR, LOG_DIR, PREP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# COCO 17-keypoint ordering
# 0:nose  1:L-eye  2:R-eye  3:L-ear   4:R-ear
# 5:L-sho 6:R-sho  7:L-elb  8:R-elb   9:L-wri 10:R-wri
# 11:L-hip 12:R-hip 13:L-kne 14:R-kne 15:L-ank 16:R-ank
# ---------------------------------------------------------------------------
COCO_PAIRS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
assert len(COCO_PAIRS) == 16

# Symmetric flip pairs for horizontal-flip augmentation
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CFG = {
    # ---- Data ----
    "img_size":              368,
    "output_stride":         8,
    "num_keypoints":         17,   # COCO 17-kp standard
    "num_pafs":              16,   # one per limb pair
    "sigma":                 7,    # Gaussian radius for heatmap generation
    "paf_thickness":         1,    # PAF band half-width in feature-map pixels

    # ---- Model ----
    "num_refinement_stages": 1 if PHASE < 3 else 3,

    # ---- Optimiser (paper values) ----
    "base_lr":               4e-5,
    "weight_decay":          5e-4,

    # ---- Schedule ----
    "batch_size":            32,
    "total_epochs":          280,
    "warmup_epochs":         5,
    "lr_milestones":         [100, 200, 260],
    "lr_gamma":              0.333,

    # ---- AMP ----
    "amp":                   True,

    # ---- Normalisation (BGR, paper convention) ----
    "mean":                  (128, 128, 128),
    "scale":                 1.0 / 256.0,

    # ---- Evaluation ----
    "eval_every_n_epochs":   10,
    "eval_max_imgs":         1000,
    "num_workers":           4,

    # ---- Reproducibility ----
    "seed":                  42,
}

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HM_SIZE  = CFG["img_size"] // CFG["output_stride"]   # 46
