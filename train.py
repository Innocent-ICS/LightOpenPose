#!/usr/bin/env python3
"""
train.py – Entry point for training Lightweight OpenPose.

Usage (from repo root):
    python train.py

The script reads all configuration from src/config.py.
Set the OPENPOSE_ROOT environment variable to override the default path.

Quick-resume after a Colab disconnect:
    The training loop auto-resumes from the latest checkpoint in
    CKPT_DIR/latest.pt.  Simply re-run `python train.py`.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    CFG, COCO_PAIRS, DEVICE, PHASE,
    PREP_TRAIN_PKL, TRAIN_ANN, TRAIN_IMG, VAL_ANN, VAL_IMG,
    VAL_SUBSET_JSON,
)
from src.dataset import (
    COCOPoseDataset,
    build_val_data_from_json,
    make_val_subset,
    prepare_train_labels,
)
from src.model import LightweightOpenPose
from src.train import run_training


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(CFG["seed"])

    # ---- data ----
    print("\n[1/4] Preparing data ...")
    prepared_train = prepare_train_labels(TRAIN_ANN, PREP_TRAIN_PKL)
    make_val_subset(VAL_ANN, VAL_SUBSET_JSON, n=CFG["eval_max_imgs"])
    prepared_val = build_val_data_from_json(VAL_SUBSET_JSON)

    train_ds = COCOPoseDataset(prepared_train, TRAIN_IMG, CFG, train=True)
    val_ds   = COCOPoseDataset(prepared_val,  VAL_IMG,   CFG, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )
    print(f"  Train: {len(train_ds):,} images | {len(train_loader):,} batches")
    print(f"  Val  : {len(val_ds):,} images  | {len(val_loader):,} batches")

    # ---- model ----
    print("\n[2/4] Building model ...")
    model = LightweightOpenPose(
        num_keypoints=CFG["num_keypoints"],
        num_pafs=CFG["num_pafs"],
        num_refinement_stages=CFG["num_refinement_stages"],
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {total_params:.2f} M")

    # ---- sanity check ----
    print("\n[3/4] Sanity check ...")
    dummy = torch.randn(1, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE)
    with torch.no_grad():
        outs = model(dummy)
    for i in range(0, len(outs), 2):
        print(f"  Stage {i // 2}: HM {tuple(outs[i].shape)}  PAF {tuple(outs[i + 1].shape)}")
    del dummy, outs

    # ---- training ----
    print(f"\n[4/4] Starting Phase {PHASE} training ...")
    run_training(model, train_loader, val_loader, str(VAL_ANN), VAL_IMG)


if __name__ == "__main__":
    main()
