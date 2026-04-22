#!/usr/bin/env python3
"""
infer.py – Run inference on one or more images and save visualisations.

Usage:
    python infer.py --input path/to/image.jpg --output results/
    python infer.py --input path/to/images/  --output results/
    python infer.py --input image.jpg --ckpt /path/to/best.pt --threshold 0.15
"""

import argparse
from pathlib import Path

import torch

from src.config import CFG, CKPT_DIR, DEVICE, PHASE
from src.model import LightweightOpenPose, load_state
from src.visualise import save_visualisation


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight OpenPose inference visualisation")
    parser.add_argument("--input",     type=str, required=True, help="Image file or directory")
    parser.add_argument("--output",    type=str, default="results",  help="Output directory")
    parser.add_argument("--ckpt",      type=str, default=None, help="Checkpoint path (default: best.pt)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Keypoint confidence threshold")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- resolve checkpoint ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = CKPT_DIR / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = CKPT_DIR / "latest.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # --- load model ---
    model = LightweightOpenPose(
        num_keypoints=CFG["num_keypoints"],
        num_pafs=CFG["num_pafs"],
        num_refinement_stages=CFG["num_refinement_stages"],
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    load_state(model, state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}  (epoch {state.get('epoch', '?')})")

    # --- collect images ---
    if input_path.is_dir():
        images = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    else:
        images = [input_path]

    for img_path in images:
        out_file = output_path / f"{img_path.stem}_pose.png"
        save_visualisation(
            model, img_path, out_file,
            phase=PHASE, ckpt_name=ckpt_path.name,
        )
        print(f"  {img_path.name} -> {out_file.name}")

    print(f"\nDone.  Saved {len(images)} visualisation(s) to {output_path}/")


if __name__ == "__main__":
    main()
