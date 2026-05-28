from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthEdgeSmoothCfg:
    weight: float = 0.05


@dataclass
class LossDepthEdgeSmoothCfgWrapper:
    depth_edge_smooth: LossDepthEdgeSmoothCfg


def compute_edge_smooth_loss(rgb, depth):
    """Edge-aware smoothness loss on depth, weighted by RGB gradients.

    Args:
        rgb: [N, 3, H, W] in [0, 1]
        depth: [N, 1, H, W]
    """
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)

    grad_depth_x = (depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs()
    grad_depth_y = (depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs()

    grad_depth_x *= (-1.0 * grad_rgb_x).exp()
    grad_depth_y *= (-1.0 * grad_rgb_y).exp()
    return grad_depth_x.mean() + grad_depth_y.mean()


class LossDepthEdgeSmooth(Loss[LossDepthEdgeSmoothCfg, LossDepthEdgeSmoothCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        static_flag: bool = False,
    ) -> Float[Tensor, ""]:
        # prediction.color: [B, V, 3, H, W] rendered RGB
        # depth_dict['depth']: [B, V, H, W] or [B, V, H, W, 1]
        rgb = prediction.color
        depth = depth_dict['depth']
        if depth.dim() == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        B, V, C, H, W = rgb.shape
        rgb = rearrange(rgb, "b v c h w -> (b v) c h w")
        depth = rearrange(depth, "b v h w -> (b v) 1 h w")

        # car_cam_mask: (B, V, 1, H, W) or (B, V, H, W) — 1=valid, 0=car region
        car_cam_mask = batch["context"]["car_cam_mask"]
        if car_cam_mask.dim() == 5 and car_cam_mask.shape[2] == 1:
            car_cam_mask = car_cam_mask[:, :, 0]
        car_cam_mask = rearrange(car_cam_mask, "b v h w -> (b v) 1 h w")

        mean_depth = depth.mean(dim=(-2, -1), keepdim=True)
        norm_depth = depth / (mean_depth + 1e-8)
        norm_depth = 1.0 / (norm_depth + 1e-8)

        # Mask out car regions before computing edge smooth loss
        rgb = rgb.clamp(0, 1) * car_cam_mask
        norm_depth = norm_depth * car_cam_mask

        loss = compute_edge_smooth_loss(rgb, norm_depth)

        return self.cfg.weight * torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
