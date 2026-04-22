# src/model.py
"""
Lightweight OpenPose architecture.

Architecture:
  MobileNetV1 (truncated at conv_dw_11, output stride 8)
  -> CPM projection (512 -> 128 channels)
  -> Initial stage  (3x3 prediction heads for HM and PAF)
  -> N Refinement stages (7x7 prediction heads, input = CPM + prev HM + prev PAF)

Reference:
  Osokin, D. (2019). Real-time 2D Multi-Person Pose Estimation on CPU:
  Lightweight OpenPose. arXiv:1811.12004.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_bn(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    """Standard 3x3 conv + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def _conv_dw(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable convolution: MobileNetV1 core block."""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def _pred_head(in_ch: int, out_ch: int, large: bool = False) -> nn.Sequential:
    """
    Prediction head.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels (num_keypoints+1 or num_pafs*2).
        large:  True for refinement stages (7x7 kernels), False for initial (3x3).
    """
    k, p = (7, 3) if large else (3, 1)
    return nn.Sequential(
        nn.Conv2d(in_ch,  128, k, 1, p, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.Conv2d(128,    128, k, 1, p, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.Conv2d(128,    128, k, 1, p, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.Conv2d(128,    512, 1,       bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        nn.Conv2d(512, out_ch, 1),
    )


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class MobileNetV1(nn.Module):
    """
    MobileNetV1 truncated after conv_dw_11.

    Output spatial stride = 8 (three stride-2 layers in the first 5 blocks).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _conv_bn(  3,  32, 2),   # /2
            _conv_dw( 32,  64, 1),
            _conv_dw( 64, 128, 2),   # /4
            _conv_dw(128, 128, 1),
            _conv_dw(128, 256, 2),   # /8  <- final spatial stride
            _conv_dw(256, 256, 1),
            _conv_dw(256, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
            _conv_dw(512, 512, 1),
        )
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# CPM projection
# ---------------------------------------------------------------------------

class CPM(nn.Module):
    """Convolutional Pose Machine feature projection: 512 -> 128 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LightweightOpenPose(nn.Module):
    """
    Lightweight OpenPose: MobileNetV1 backbone + CPM + multi-stage prediction.

    forward() returns a list:
        [hm_s0, paf_s0, hm_s1, paf_s1, ...]
    All stages contribute to the deep-supervision loss during training.
    Only the final stage (outs[-2], outs[-1]) is used at inference.

    Channel layout:
        heatmap:  num_keypoints + 1  (last channel = background)
        PAF    :  num_pafs * 2       (dx, dy per limb)
    """

    def __init__(
        self,
        num_keypoints: int,
        num_pafs: int,
        num_refinement_stages: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = MobileNetV1()
        self.cpm      = CPM()
        C             = self.cpm.out_channels   # 128

        # Initial stage operates on CPM features only (3x3 heads)
        self.initial_stage = nn.ModuleDict({
            "hm":  _pred_head(C, num_keypoints + 1, large=False),
            "paf": _pred_head(C, num_pafs * 2,      large=False),
        })

        # Refinement stages concatenate CPM + previous predictions (7x7 heads)
        in_ref = C + (num_keypoints + 1) + (num_pafs * 2)
        self.refinement_stages = nn.ModuleList([
            nn.ModuleDict({
                "hm":  _pred_head(in_ref, num_keypoints + 1, large=True),
                "paf": _pred_head(in_ref, num_pafs * 2,      large=True),
            })
            for _ in range(num_refinement_stages)
        ])
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> list:
        feat = self.cpm(self.backbone(x))        # (B, 128, H/8, W/8)
        hm   = self.initial_stage["hm"](feat)
        paf  = self.initial_stage["paf"](feat)
        outs = [hm, paf]
        for stage in self.refinement_stages:
            cat = torch.cat([feat, hm, paf], dim=1)
            hm  = stage["hm"](cat)
            paf = stage["paf"](cat)
            outs.extend([hm, paf])
        return outs   # [hm_s0, paf_s0, hm_s1, paf_s1, ...]


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

def load_from_mobilenet(model: LightweightOpenPose, checkpoint: dict) -> None:
    """Transfer ImageNet MobileNetV1 weights into the backbone only."""
    src = checkpoint.get("state_dict", checkpoint)
    src = {k.replace("module.", ""): v for k, v in src.items()}
    dst = model.state_dict()
    n = 0
    for name in list(dst.keys()):
        src_key = name.replace("backbone.", "", 1)
        if src_key in src and src[src_key].shape == dst[name].shape:
            dst[name] = src[src_key]
            n += 1
    model.load_state_dict(dst, strict=True)
    print(f"  Loaded {n} backbone tensors from MobileNetV1 checkpoint.")


def load_state(model: LightweightOpenPose, checkpoint: dict) -> None:
    """
    Load full model weights.  strict=False allows phase transitions
    where the number of refinement stages changes.
    """
    sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
