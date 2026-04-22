#!/usr/bin/env python3
"""
evaluate.py – Standalone COCO keypoint AP evaluation.

Usage:
    # Evaluate best checkpoint on first 1000 val images (fast, ~3 min on L4)
    python evaluate.py --max_imgs 1000

    # Full 5000-image validation (~20 min on L4)
    python evaluate.py

    # Specify a custom checkpoint
    python evaluate.py --ckpt /path/to/checkpoint.pt
"""

import argparse
from pathlib import Path

import torch

from src.config import CFG, CKPT_DIR, DEVICE, VAL_ANN, VAL_IMG
from src.evaluate import evaluate_ap
from src.model import LightweightOpenPose, load_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lightweight OpenPose on COCO val2017")
    parser.add_argument("--ckpt",     type=str, default=None,  help="Path to checkpoint (.pt)")
    parser.add_argument("--max_imgs", type=int, default=None,  help="Limit evaluation to N images")
    args = parser.parse_args()

    # --- resolve checkpoint ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = CKPT_DIR / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = CKPT_DIR / "latest.pt"
    assert ckpt_path.exists(), f"No checkpoint found at {ckpt_path}"

    # --- load model ---
    model = LightweightOpenPose(
        num_keypoints=CFG["num_keypoints"],
        num_pafs=CFG["num_pafs"],
        num_refinement_stages=CFG["num_refinement_stages"],
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    load_state(model, state)
    epoch = state.get("epoch", "?")
    print(f"Evaluating epoch {epoch} from {ckpt_path.name} ...")

    ap = evaluate_ap(
        model, str(VAL_ANN), VAL_IMG, CFG, DEVICE,
        max_imgs=args.max_imgs,
    )
    print(f"\nOKS mAP (AP@0.50:0.95): {ap:.4f}  ({ap * 100:.1f} AP)")


if __name__ == "__main__":
    main()
