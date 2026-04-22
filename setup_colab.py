#!/usr/bin/env python3
"""
setup_colab.py – One-shot Colab environment bootstrap.

Run this cell first when starting a new Colab session:

    !python setup_colab.py

It will:
  1. Raise the open-file-descriptor limit.
  2. Install pip dependencies quietly.
  3. Mount Google Drive.
  4. Download COCO 2017 train/val images + annotations (idempotent).
  5. Download the pretrained MobileNetV1 weights (idempotent).
"""

import os
import resource
import subprocess
import sys
from pathlib import Path

# 1. raise FD limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))

# 2. pip install
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "pycocotools", "opencv-python-headless", "albumentations"],
    check=True,
)

# 3. mount drive
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    print("  (not running in Colab – skipping drive mount)")

# 4. paths
DRIVE_ROOT = Path(os.environ.get(
    "OPENPOSE_ROOT",
    "/content/drive/MyDrive/Colab Notebooks/OpenPose"
))
COCO_DIR   = DRIVE_ROOT / "coco"
COCO_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    ("train2017.zip",
     "http://images.cocodataset.org/zips/train2017.zip",
     "train2017"),
    ("val2017.zip",
     "http://images.cocodataset.org/zips/val2017.zip",
     "val2017"),
    ("annotations_trainval2017.zip",
     "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
     "annotations"),
]

for zip_name, url, extracted in DOWNLOADS:
    dest = COCO_DIR / zip_name
    extr = COCO_DIR / extracted
    if extr.exists():
        print(f"  check {extracted}/")
        continue
    if not dest.exists():
        print(f"  Downloading {zip_name} ...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
    print(f"  Extracting {zip_name} ...")
    subprocess.run(["unzip", "-q", str(dest), "-d", str(COCO_DIR)], check=True)

# 5. MobileNetV1 weights
MOBILENET_W = DRIVE_ROOT / "mobilenet_sgd_68.848.pth.tar"
if not MOBILENET_W.exists():
    print("  Downloading MobileNetV1 weights ...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(MOBILENET_W),
         "https://drive.google.com/uc?export=download&id=18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv"],
        check=True,
    )
else:
    print("  check MobileNet weights")

print("\nEnvironment ready.  Now run:  python train.py")
