# src/evaluate.py
"""
COCO OKS evaluation.

evaluate_ap() runs full COCO keypoint evaluation on a given image subset.

Key design decisions:
  - FIX 3: The coordinate inverse transform correctly accounts for the
    symmetric padding added by val_preprocess().
  - FIX 4: Per-GT-bounding-box peak extraction is used instead of PAF
    assembly so the evaluation isolates heatmap quality.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.amp import autocast

from src.dataset import normalise, val_preprocess
from src.model import LightweightOpenPose


@torch.no_grad()
def evaluate_ap(
    model:    LightweightOpenPose,
    ann_file: str,
    img_dir:  Path,
    cfg:      dict,
    device:   torch.device,
    max_imgs: Optional[int] = None,
    res_file: Optional[Path] = None,
) -> float:
    """
    Compute COCO keypoint AP.

    Args:
        model    : trained LightweightOpenPose (set to eval mode internally).
        ann_file : path to COCO annotation JSON.
        img_dir  : directory containing validation images.
        cfg      : configuration dict (from src.config.CFG).
        device   : torch device.
        max_imgs : if set, evaluate only the first max_imgs images.
        res_file : where to write the detection JSON (temp file).

    Returns:
        AP @ OKS=0.50:0.95  (the primary COCO metric)
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ann_file))
    cat_ids = coco_gt.getCatIds(catNms=["person"])
    img_ids = coco_gt.getImgIds(catIds=cat_ids)
    if max_imgs:
        img_ids = img_ids[:max_imgs]

    model.eval()
    results  = []
    feat_sz  = cfg["img_size"] // cfg["output_stride"]   # 46

    for img_id in img_ids:
        info    = coco_gt.loadImgs(img_id)[0]
        img_bgr = cv2.imread(str(Path(img_dir) / info["file_name"]))
        if img_bgr is None:
            continue

        H_orig, W_orig = img_bgr.shape[:2]

        # Replicate val_preprocess() offsets for exact inverse transform
        scale  = cfg["img_size"] / max(H_orig, W_orig)
        new_h  = int(H_orig * scale)
        new_w  = int(W_orig * scale)
        pad_t  = (cfg["img_size"] - new_h) // 2
        pad_l  = (cfg["img_size"] - new_w) // 2

        img_pre  = val_preprocess(img_bgr, cfg["img_size"], cfg["mean"])
        img_norm = normalise(img_pre, cfg["mean"], cfg["scale"])
        img_t    = (
            torch.from_numpy(img_norm.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(device)
        )

        with autocast("cuda", enabled=cfg["amp"]):
            outs = model(img_t)

        # Final-stage heatmap (hm_s_last)
        hm_np = outs[-2].float().cpu().numpy()[0]   # (num_kps+1, 46, 46)

        anns = coco_gt.loadAnns(
            coco_gt.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        )
        anns = [a for a in anns if a.get("num_keypoints", 0) > 0]
        if not anns:
            continue

        for ann in anns:
            bx, by, bw, bh = ann["bbox"]

            def to_feat(xo: float, yo: float):
                """Original-image coords -> feature-map coords."""
                return (
                    (xo * scale + pad_l) / cfg["output_stride"],
                    (yo * scale + pad_t) / cfg["output_stride"],
                )

            fx0, fy0 = to_feat(bx,      by)
            fx1, fy1 = to_feat(bx + bw, by + bh)
            margin   = 2
            x0 = max(0,       int(fx0) - margin)
            y0 = max(0,       int(fy0) - margin)
            x1 = min(feat_sz, int(fx1) + margin)
            y1 = min(feat_sz, int(fy1) + margin)

            kps_out   = []
            score_sum = 0.0
            cnt       = 0

            for k in range(cfg["num_keypoints"]):
                ch     = gaussian_filter(hm_np[k], sigma=3)
                region = (
                    ch[y0:y1, x0:x1]
                    if (y1 > y0 and x1 > x0)
                    else ch
                )
                if region.size == 0 or region.max() < 0.05:
                    kps_out += [0.0, 0.0, 0]
                    continue

                ry, rx  = np.unravel_index(region.argmax(), region.shape)
                fx      = float(rx + x0)
                fy      = float(ry + y0)

                # FIX 3: feature-map -> padded input -> original image coords
                x_orig = float(
                    np.clip((fx * cfg["output_stride"] - pad_l) / scale, 0, W_orig - 1)
                )
                y_orig = float(
                    np.clip((fy * cfg["output_stride"] - pad_t) / scale, 0, H_orig - 1)
                )
                kps_out   += [x_orig, y_orig, 2]
                score_sum += float(region.max())
                cnt       += 1

            if cnt == 0:
                continue
            results.append({
                "image_id":    img_id,
                "category_id": 1,
                "keypoints":   kps_out,
                "score":       score_sum / cnt,
            })

    if not results:
        print("  No detections produced.")
        return 0.0

    print(f"  Detections: {len(results)}")
    if res_file is None:
        res_file = Path("/tmp/val_detections_openpose.json")
    with open(res_file, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(str(res_file))
    ev = COCOeval(coco_gt, coco_dt, "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])
