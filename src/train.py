# src/train.py
"""
Training loop, checkpoint management, and optimizer construction.

Entry point:  python train.py  (see train.py in the project root)
"""

import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from src.config import CFG, CKPT_DIR, DEVICE, LOG_DIR, PHASE
from src.evaluate import evaluate_ap
from src.loss import OpenPoseLoss
from src.model import LightweightOpenPose, load_from_mobilenet, load_state


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model:   LightweightOpenPose,
    base_lr: float,
    wd:      float,
) -> torch.optim.Optimizer:
    """
    Parameter-group Adam with differential LRs.

    - backbone + CPM:        base_lr × 1
    - initial_stage:         base_lr × 1
    - refinement_stages:     base_lr × 4   (faster convergence on new heads)
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    cpm_ids      = {id(p) for p in model.cpm.parameters()}
    init_ids     = {id(p) for p in model.initial_stage.parameters()}
    ref_ids      = {id(p) for p in model.refinement_stages.parameters()}

    def group(ids, mult):
        return {
            "params":       [p for p in model.parameters() if id(p) in ids and p.requires_grad],
            "lr":           base_lr * mult,
            "weight_decay": wd,
        }

    groups = [
        group(backbone_ids, 1.0),
        group(cpm_ids,      1.0),
        group(init_ids,     1.0),
        group(ref_ids,      4.0),
    ]
    return torch.optim.Adam(
        [g for g in groups if g["params"]],
        lr=base_lr,
        weight_decay=wd,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_ckpt(
    model:      LightweightOpenPose,
    opt:        torch.optim.Optimizer,
    sched:      MultiStepLR,
    scaler:     GradScaler,
    epoch:      int,
    best_ap:    float,
    current_ap: Optional[float],
    meta:       dict,
) -> float:
    """
    Save latest.pt every epoch.  Copy to best.pt whenever AP improves.
    """
    state = {
        "state_dict": model.state_dict(),
        "optimizer":  opt.state_dict(),
        "scheduler":  sched.state_dict(),
        "scaler":     scaler.state_dict(),
        "epoch":      epoch,
        "best_ap":    best_ap,
        "meta":       meta,
    }
    torch.save(state, CKPT_DIR / "latest.pt")
    if current_ap is not None and current_ap > best_ap:
        shutil.copy(CKPT_DIR / "latest.pt", CKPT_DIR / "best.pt")
        print(f"  New best AP {current_ap:.4f} -> best.pt")
        return current_ap
    return best_ap


def load_ckpt(
    model:  LightweightOpenPose,
    opt:    torch.optim.Optimizer,
    sched:  MultiStepLR,
    scaler: GradScaler,
    mobilenet_w: Optional[Path] = None,
) -> Tuple[int, float]:
    """
    Auto-resume from latest.pt if it exists.
    Otherwise, load MobileNet backbone weights (Phase 1) or the previous
    phase's best.pt (Phase 2+).
    """
    from src.config import DRIVE_ROOT, MOBILENET_W

    latest = CKPT_DIR / "latest.pt"
    if latest.exists():
        s = torch.load(latest, map_location=DEVICE)
        load_state(model, s)
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        start = s["epoch"]
        best  = s.get("best_ap", 0.0)
        print(f"  Resumed from epoch {start}  best AP {best:.4f}")
        return start, best

    if PHASE == 1:
        mw = mobilenet_w or MOBILENET_W
        if mw.exists():
            load_from_mobilenet(model, torch.load(mw, map_location=DEVICE))
            print("  Phase 1: MobileNetV1 backbone loaded.")
        else:
            print("  WARNING: MobileNet weights not found – random init.")
    else:
        prev = DRIVE_ROOT / "checkpoints" / f"phase{PHASE - 1}" / "best.pt"
        if not prev.exists():
            raise FileNotFoundError(f"Phase {PHASE - 1} best.pt not found at {prev}")
        load_state(model, torch.load(prev, map_location=DEVICE))
        print(f"  Phase {PHASE}: loaded Phase {PHASE - 1} best weights.")
    return 0, 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training(
    model:        LightweightOpenPose,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    val_ann:      str,
    val_img:      Path,
) -> None:
    """
    Full training loop with:
      - LR warmup (linear ramp over warmup_epochs)
      - Multi-step LR decay at milestones
      - AMP (mixed precision)
      - Gradient clipping (max norm = 5)
      - AP evaluation every eval_every_n_epochs epochs
      - CSV logging to LOG_DIR/phase{PHASE}_log.csv
    """
    criterion = OpenPoseLoss()
    optimizer = build_optimizer(model, CFG["base_lr"], CFG["weight_decay"])
    scheduler = MultiStepLR(
        optimizer,
        milestones=CFG["lr_milestones"],
        gamma=CFG["lr_gamma"],
    )
    scaler     = GradScaler("cuda", enabled=CFG["amp"])
    start_epoch, best_ap = load_ckpt(model, optimizer, scheduler, scaler)

    log_path = LOG_DIR / f"phase{PHASE}_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,lr,train_loss,val_loss,ap\n")

    print(f"\n{'=' * 60}")
    print(f"  Phase {PHASE} | Epochs {start_epoch + 1} to {CFG['total_epochs']}")
    print(f"  Batch {CFG['batch_size']} | AMP {CFG['amp']} "
          f"| AP eval every {CFG['eval_every_n_epochs']} epochs")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch + 1, CFG["total_epochs"] + 1):
        t0 = time.time()

        # ----- warmup -----
        if epoch <= CFG["warmup_epochs"]:
            wf = epoch / CFG["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", CFG["base_lr"]) * wf

        # ----- train -----
        model.train()
        acc: dict = defaultdict(float)
        for step, (imgs, gt_hm, gt_paf, hm_mask, paf_mask) in enumerate(train_loader, 1):
            imgs     = imgs.to(DEVICE,     non_blocking=True)
            gt_hm_d  = gt_hm.to(DEVICE,   non_blocking=True)
            gt_paf_d = gt_paf.to(DEVICE,  non_blocking=True)
            hm_m     = hm_mask.to(DEVICE,  non_blocking=True)
            paf_m    = paf_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=CFG["amp"]):
                outs        = model(imgs)
                loss, parts = criterion(outs, gt_hm_d, gt_paf_d, hm_m, paf_m)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            acc["total"] += loss.item()

            if step % 200 == 0:
                print(
                    f"  Ep {epoch:3d} | Step {step:4d}/{len(train_loader)} "
                    f"| Loss {loss.item():.2f} "
                    f"| LR {optimizer.param_groups[0]['lr']:.2e} "
                    f"| {(time.time() - t0) / 60:.1f} min"
                )

        avg_train = acc["total"] / len(train_loader)

        # ----- val loss -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, gt_hm, gt_paf, hm_mask, paf_mask in val_loader:
                imgs     = imgs.to(DEVICE,     non_blocking=True)
                gt_hm_d  = gt_hm.to(DEVICE,   non_blocking=True)
                gt_paf_d = gt_paf.to(DEVICE,  non_blocking=True)
                hm_m     = hm_mask.to(DEVICE,  non_blocking=True)
                paf_m    = paf_mask.to(DEVICE, non_blocking=True)
                with autocast("cuda", enabled=CFG["amp"]):
                    loss, _ = criterion(model(imgs), gt_hm_d, gt_paf_d, hm_m, paf_m)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        if epoch > CFG["warmup_epochs"]:
            scheduler.step()

        # ----- AP evaluation -----
        current_ap = None
        if epoch % CFG["eval_every_n_epochs"] == 0:
            print(f"  Running AP eval at epoch {epoch} ...")
            current_ap = evaluate_ap(
                model, val_ann, val_img, CFG, DEVICE,
                max_imgs=CFG["eval_max_imgs"],
            )
            print(f"  AP @ epoch {epoch}: {current_ap:.4f}")

        best_ap = save_ckpt(
            model, optimizer, scheduler, scaler,
            epoch, best_ap, current_ap,
            {"train_loss": avg_train, "val_loss": avg_val},
        )

        lr_now = optimizer.param_groups[0]["lr"]
        ap_str = f"{current_ap:.4f}" if current_ap is not None else ""
        with open(log_path, "a") as f:
            f.write(f"{epoch},{lr_now:.2e},{avg_train:.4f},{avg_val:.4f},{ap_str}\n")

        print(
            f"\n-- Ep {epoch:3d}/{CFG['total_epochs']}  "
            f"Train {avg_train:.2f}  Val {avg_val:.2f}  "
            f"LR {lr_now:.2e}  {(time.time() - t0) / 60:.1f} min --\n"
        )

    print("Training complete!")
