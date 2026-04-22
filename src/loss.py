# src/loss.py
"""
OpenPose loss function.

Deep-supervision masked MSE loss summed over all prediction stages.
Both heatmap and PAF heads contribute a term at each stage.
"""

import torch
import torch.nn as nn


class OpenPoseLoss(nn.Module):
    """
    Masked MSE loss over all stages.

    Args:
        stage_outputs: list [hm_s0, paf_s0, hm_s1, paf_s1, ...]
        gt_hm:        (B, num_kps+1, H, W)
        gt_paf:       (B, num_pafs*2, H, W)
        hm_mask:      same shape as gt_hm  (1 = pixel contributes)
        paf_mask:     same shape as gt_paf

    Returns:
        total: scalar loss
        parts: dict of per-stage component losses (for logging)
    """

    def forward(
        self,
        stage_outputs: list,
        gt_hm:   torch.Tensor,
        gt_paf:  torch.Tensor,
        hm_mask:  torch.Tensor,
        paf_mask: torch.Tensor,
    ):
        B     = gt_hm.shape[0]
        total = torch.tensor(0.0, device=gt_hm.device)
        parts: dict = {}

        for s in range(len(stage_outputs) // 2):
            hm_p  = stage_outputs[2 * s]
            paf_p = stage_outputs[2 * s + 1]
            l_hm  = ((hm_p  - gt_hm ) ** 2 * hm_mask ).sum() / B
            l_paf = ((paf_p - gt_paf) ** 2 * paf_mask).sum() / B
            total = total + l_hm + l_paf
            parts[f"s{s}_hm"]  = l_hm.item()
            parts[f"s{s}_paf"] = l_paf.item()

        return total, parts
