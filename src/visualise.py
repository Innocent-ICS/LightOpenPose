# src/visualise.py
"""
Inference visualisation utilities.

visualise() runs a single image through the model and returns an
annotated RGB canvas with keypoint circles and limb lines.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.amp import autocast

from src.config import CFG, COCO_PAIRS, DEVICE
from src.dataset import normalise, val_preprocess
from src.model import LightweightOpenPose

# Assign a distinct colour to each limb
_COLOURS = plt.cm.hsv(np.linspace(0, 0.9, len(COCO_PAIRS)))


@torch.no_grad()
def visualise(
    model:      LightweightOpenPose,
    image_path: Path,
    threshold:  float = 0.1,
    cfg:        Optional[dict] = None,
    device:     Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run inference on a single image and return an annotated RGB canvas.

    Args:
        model      : trained LightweightOpenPose (eval mode set internally).
        image_path : path to an image file.
        threshold  : minimum heatmap confidence to plot a keypoint.
        cfg        : config dict (defaults to src.config.CFG).
        device     : torch device (defaults to src.config.DEVICE).

    Returns:
        canvas: (H, W, 3) RGB uint8 numpy array.
    """
    cfg    = cfg    or CFG
    device = device or DEVICE

    img_bgr      = cv2.imread(str(image_path))
    H, W         = img_bgr.shape[:2]
    scale        = cfg["img_size"] / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    pad_t        = (cfg["img_size"] - new_h) // 2
    pad_l        = (cfg["img_size"] - new_w) // 2

    img_pre  = val_preprocess(img_bgr, cfg["img_size"], cfg["mean"])
    img_norm = normalise(img_pre, cfg["mean"], cfg["scale"])
    img_t    = (
        torch.from_numpy(img_norm.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device)
    )

    model.eval()
    with autocast("cuda", enabled=cfg["amp"]):
        outs = model(img_t)

    hm_np = outs[-2].float().cpu().numpy()[0]   # (num_kps+1, 46, 46)

    kp_xy = []
    for k in range(cfg["num_keypoints"]):
        ch = gaussian_filter(hm_np[k], sigma=3)
        if ch.max() < threshold:
            kp_xy.append(None)
            continue
        fy, fx  = np.unravel_index(ch.argmax(), ch.shape)
        x_orig  = int(np.clip((fx * cfg["output_stride"] - pad_l) / scale, 0, W - 1))
        y_orig  = int(np.clip((fy * cfg["output_stride"] - pad_t) / scale, 0, H - 1))
        kp_xy.append((x_orig, y_orig))

    canvas = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for li, (ka, kb) in enumerate(COCO_PAIRS):
        if kp_xy[ka] and kp_xy[kb]:
            c = tuple(int(x * 255) for x in _COLOURS[li][:3])
            cv2.line(canvas, kp_xy[ka], kp_xy[kb], c, 2, cv2.LINE_AA)
    for pos in kp_xy:
        if pos:
            cv2.circle(canvas, pos, 5, (0, 255, 0), -1)
            cv2.circle(canvas, pos, 5, (255, 255, 255), 1)
    return canvas


def save_visualisation(
    model:      LightweightOpenPose,
    image_path: Path,
    out_path:   Path,
    phase:      int = 1,
    ckpt_name:  str = "best.pt",
) -> None:
    """Convenience wrapper: visualise and save as PNG."""
    canvas = visualise(model, image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"Phase {phase} | {ckpt_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualisation to {out_path}")
