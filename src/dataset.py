# src/dataset.py
"""
COCO 2017 pose dataset.

Provides:
  - prepare_train_labels()  : build/cache a flat list of training samples
  - make_val_subset()       : build/cache a 1000-image validation subset JSON
  - COCOPoseDataset         : PyTorch Dataset (train + val modes)
  - Augmentation pipeline   : scale, rotate, crop/pad, horizontal flip
  - Target generation       : Gaussian heatmaps + Part Affinity Fields
"""

import json
import math
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import COCO_PAIRS, FLIP_PAIRS


# ---------------------------------------------------------------------------
# Annotation preparation
# ---------------------------------------------------------------------------

def prepare_train_labels(ann_file: Path, out_pkl: Path) -> List[dict]:
    """
    Parse the COCO training annotations into a compact list.

    Each element is a dict::

        {
            'img_id'   : int,
            'file_name': str,
            'width'    : int,
            'height'   : int,
            'persons'  : [
                {'joints': [(x, y, v), ...],   # 17 joints
                 'bbox'  : [x, y, w, h]},
                ...
            ]
        }

    The result is cached as a pickle so subsequent runs are instant.
    """
    if out_pkl.exists():
        print(f"  Using cached {out_pkl.name}")
        with open(out_pkl, "rb") as f:
            return pickle.load(f)

    from pycocotools.coco import COCO  # lazy import

    print("  Building train labels cache ...")
    coco    = COCO(str(ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=cat_ids)
    out     = []

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        )
        persons = []
        for ann in anns:
            kps = ann.get("keypoints", [])
            if ann.get("num_keypoints", 0) == 0 or not kps:
                continue
            joints = [(kps[i], kps[i + 1], kps[i + 2]) for i in range(0, 51, 3)]
            assert len(joints) == 17, f"Expected 17 joints, got {len(joints)}"
            persons.append({"joints": joints, "bbox": ann["bbox"]})

        if persons:
            out.append({
                "img_id":    img_id,
                "file_name": info["file_name"],
                "width":     info["width"],
                "height":    info["height"],
                "persons":   persons,
            })

    with open(out_pkl, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"  Saved {len(out)} images to {out_pkl.name}")
    return out


def make_val_subset(ann_file: Path, out_json: Path, n: int = 1000) -> None:
    """
    Sample n images from the validation set and write a self-contained
    COCO annotation JSON.  Result is cached; safe to call multiple times.
    """
    if out_json.exists():
        print(f"  Using cached {out_json.name}")
        return
    print(f"  Building {n}-image val subset ...")
    with open(ann_file) as f:
        data = json.load(f)
    imgs   = random.sample(data["images"], min(n, len(data["images"])))
    ids    = {i["id"] for i in imgs}
    subset = {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "categories":  data["categories"],
        "images":      imgs,
        "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
    }
    with open(out_json, "w") as f:
        json.dump(subset, f)
    print(f"  Saved {len(imgs)}-image subset to {out_json.name}")


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def make_heatmaps(
    H: int, W: int,
    joints: List[Tuple],
    sigma: int,
) -> np.ndarray:
    """
    Generate Gaussian heatmaps of shape (num_kps+1, H, W).

    The last channel is a background map: 1 - max over all keypoint channels.
    Visible and occluded keypoints (v=1,2) both produce a Gaussian;
    invisible keypoints (v=0) are skipped.
    """
    hm   = np.zeros((len(joints) + 1, H, W), np.float32)
    size = 6 * sigma + 3
    c    = 3 * sigma + 1
    xr   = np.arange(size, dtype=np.float32)
    g    = np.exp(-((xr - c) ** 2 + (xr[:, None] - c) ** 2) / (2 * sigma ** 2))

    for k, (cx, cy, vis) in enumerate(joints):
        if vis == 0:
            continue
        cx, cy = int(round(cx)), int(round(cy))
        x1, y1 = cx - c, cy - c
        x2, y2 = x1 + size, y1 + size
        gx1, gy1 = max(0, -x1), max(0, -y1)
        gx2, gy2 = size - max(0, x2 - W), size - max(0, y2 - H)
        ix1, iy1 = max(0, x1), max(0, y1)
        ix2, iy2 = min(W, x2), min(H, y2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue
        hm[k, iy1:iy2, ix1:ix2] = np.maximum(
            hm[k, iy1:iy2, ix1:ix2], g[gy1:gy2, gx1:gx2]
        )
    hm[-1] = np.clip(1 - hm[:-1].max(0), 0, 1)
    return hm


def make_pafs(
    H: int, W: int,
    joints: List[Tuple],
    pairs: List[Tuple],
    thickness: int,
) -> np.ndarray:
    """
    Generate Part Affinity Fields of shape (num_pairs*2, H, W).

    For each limb (ka, kb) the two channels encode the unit vector
    (dx, dy) pointing from joint ka to joint kb, drawn as a filled band
    of half-width `thickness` pixels.
    """
    paf = np.zeros((len(pairs) * 2, H, W), np.float32)
    cnt = np.zeros((len(pairs),     H, W), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]

    for i, (ka, kb) in enumerate(pairs):
        if ka >= len(joints) or kb >= len(joints):
            continue
        if joints[ka][2] == 0 or joints[kb][2] == 0:
            continue
        xa, ya = joints[ka][0], joints[ka][1]
        xb, yb = joints[kb][0], joints[kb][1]
        dx, dy = xb - xa, yb - ya
        norm   = math.sqrt(dx ** 2 + dy ** 2) + 1e-6
        ux, uy = dx / norm, dy / norm
        along  = (xx - xa) * ux + (yy - ya) * uy
        perp   = np.abs((xx - xa) * (-uy) + (yy - ya) * ux)
        mask   = (perp < thickness) & (along >= 0) & (along <= norm)
        paf[2 * i    ][mask] += ux
        paf[2 * i + 1][mask] += uy
        cnt[i          ][mask] += 1

    for i in range(len(pairs)):
        m = cnt[i] > 0
        paf[2 * i    ][m] /= cnt[i][m]
        paf[2 * i + 1][m] /= cnt[i][m]
    return paf


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def normalise(
    img_bgr: np.ndarray,
    mean: Tuple,
    scale: float,
) -> np.ndarray:
    """Subtract mean and multiply by scale (paper convention: mean=128, scale=1/256)."""
    return (img_bgr.astype(np.float32) - np.array(mean, np.float32)) * scale


def val_preprocess(
    img_bgr: np.ndarray,
    target: int,
    pad_val: Tuple = (128, 128, 128),
) -> np.ndarray:
    """
    Resize the longest side to `target`, then symmetrically pad the shorter
    side so the output is exactly (target, target, 3).
    """
    H, W  = img_bgr.shape[:2]
    scale = target / max(H, W)
    img   = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    h2, w2 = img.shape[:2]
    top  = (target - h2) // 2
    left = (target - w2) // 2
    return cv2.copyMakeBorder(
        img, top, target - h2 - top, left, target - w2 - left,
        cv2.BORDER_CONSTANT, value=pad_val,
    )


def augment(
    img: np.ndarray,
    joints_list: List[List[Tuple]],
    target: int,
    pad_val: Tuple = (128, 128, 128),
) -> Tuple[np.ndarray, List[List[Tuple]]]:
    """
    Training augmentation pipeline:
        1. Random scale  [0.7, 1.3]
        2. Random rotation  ±40 degrees
        3. Random crop / zero-pad to target size
        4. Horizontal flip with symmetric keypoint swapping (p=0.5)
    """
    H, W = img.shape[:2]

    # --- scale ---
    s   = random.uniform(0.7, 1.3)
    img = cv2.resize(img, (int(W * s), int(H * s)))
    joints_list = [[(x * s, y * s, v) for x, y, v in p] for p in joints_list]

    # --- rotation ---
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    ang = random.uniform(-40, 40)
    M   = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    img = cv2.warpAffine(img, M, (W, H),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=pad_val)
    new_jl = []
    for person in joints_list:
        np_ = []
        for x, y, v in person:
            if v == 0:
                np_.append((0.0, 0.0, 0))
                continue
            nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            np_.append((nx, ny, v if 0 <= nx < W and 0 <= ny < H else 0))
        new_jl.append(np_)
    joints_list = new_jl

    # --- crop / pad ---
    H, W = img.shape[:2]
    ph   = max(0, target - H)
    pw   = max(0, target - W)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw,
                                 cv2.BORDER_CONSTANT, value=pad_val)
        H, W = img.shape[:2]
    y0 = random.randint(0, max(0, H - target))
    x0 = random.randint(0, max(0, W - target))
    img = img[y0:y0 + target, x0:x0 + target]
    joints_list = [
        [(x - x0, y - y0,
          v if x0 <= x < x0 + target and y0 <= y < y0 + target else 0)
         for x, y, v in p]
        for p in joints_list
    ]

    # --- horizontal flip ---
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        W2  = img.shape[1]
        jl2 = []
        for person in joints_list:
            p = [(W2 - x - 1, y, v) for x, y, v in person]
            for a, b in FLIP_PAIRS:
                if a < len(p) and b < len(p):
                    p[a], p[b] = p[b], p[a]
            jl2.append(p)
        joints_list = jl2

    return img, joints_list


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCOPoseDataset(Dataset):
    """
    PyTorch Dataset for COCO 2017 pose estimation.

    Returns:
        img_t    : (3, H, W) float32 normalised tensor
        hm_t     : (num_kps+1, hm_sz, hm_sz) float32
        paf_t    : (num_pafs*2, hm_sz, hm_sz) float32
        hm_mask  : ones_like(hm_t)   – reserved for per-pixel masking
        paf_mask : ones_like(paf_t)
    """

    def __init__(
        self,
        data: List[dict],
        img_dir: Path,
        cfg: dict,
        train: bool = True,
    ) -> None:
        self.data    = data
        self.img_dir = Path(img_dir)
        self.cfg     = cfg
        self.train   = train
        self.hm_sz   = cfg["img_size"] // cfg["output_stride"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        img   = cv2.imread(str(self.img_dir / entry["file_name"]))
        if img is None:
            img = np.full(
                (self.cfg["img_size"],) * 2 + (3,), 128, np.uint8
            )

        joints_list = [p["joints"] for p in entry["persons"]]

        if self.train:
            img, joints_list = augment(
                img, joints_list, self.cfg["img_size"], self.cfg["mean"]
            )
        else:
            img = val_preprocess(img, self.cfg["img_size"], self.cfg["mean"])

        # Scale joints to feature-map coordinates
        sz  = self.hm_sz
        sx  = sz / self.cfg["img_size"]
        sy  = sz / self.cfg["img_size"]
        fm_jl = [[(x * sx, y * sy, v) for x, y, v in p] for p in joints_list]

        gt_hm  = np.zeros((self.cfg["num_keypoints"] + 1, sz, sz), np.float32)
        gt_paf = np.zeros((self.cfg["num_pafs"] * 2,      sz, sz), np.float32)

        for pj in fm_jl:
            gt_hm  = np.maximum(
                gt_hm,
                make_heatmaps(sz, sz, pj, self.cfg["sigma"])
            )
            gt_paf += make_pafs(
                sz, sz, pj, COCO_PAIRS, self.cfg["paf_thickness"]
            )

        if len(fm_jl) > 1:
            gt_paf = np.clip(gt_paf / len(fm_jl), -1.0, 1.0)

        img_t  = torch.from_numpy(
            normalise(img, self.cfg["mean"], self.cfg["scale"]).transpose(2, 0, 1)
        )
        hm_t   = torch.from_numpy(gt_hm)
        paf_t  = torch.from_numpy(gt_paf)
        return img_t, hm_t, paf_t, torch.ones_like(hm_t), torch.ones_like(paf_t)


# ---------------------------------------------------------------------------
# Convenience: build val dataset from a subset JSON
# ---------------------------------------------------------------------------

def build_val_data_from_json(json_path: Path) -> List[dict]:
    """
    Parse a COCO-format val-subset JSON into the same flat list
    format produced by prepare_train_labels().
    """
    with open(json_path) as f:
        vsub = json.load(f)
    ann_by_img: dict = {}
    for a in vsub["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    result = []
    for im in vsub["images"]:
        anns = ann_by_img.get(im["id"], [])
        persons = []
        for a in anns:
            kps = a.get("keypoints", [])
            if a.get("num_keypoints", 0) == 0 or not kps:
                continue
            persons.append({
                "joints": [
                    (kps[i], kps[i + 1], kps[i + 2]) for i in range(0, 51, 3)
                ]
            })
        if persons:
            result.append({
                "img_id":    im["id"],
                "file_name": im["file_name"],
                "width":     im["width"],
                "height":    im["height"],
                "persons":   persons,
            })
    return result
