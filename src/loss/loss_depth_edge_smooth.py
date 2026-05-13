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

        # Normalized inverse depth (disparity): near-sensitive, scale-invariant
        # inv_depth = 1.0 / (depth + 1e-8) # loss 0.0007
        # mean_inv_depth = inv_depth.mean(dim=(-2, -1), keepdim=True)
        # norm_inv_depth = inv_depth / (mean_inv_depth + 1e-8)
        # loss = compute_edge_smooth_loss(rgb.clamp(0, 1), norm_inv_depth)

        # # Alternative: normalized depth (preserves original gradient distribution)
        mean_depth = depth.mean(dim=(-2, -1), keepdim=True) # loss 0.0009 
        norm_depth = depth / (mean_depth + 1e-8) 
        norm_depth = 1.0 / (norm_depth + 1e-8) 
        loss = compute_edge_smooth_loss(rgb.clamp(0, 1), norm_depth)
        
        # Use inverse depth (disparity) for better gradient distribution
        # inv_depth = 1.0 / (depth + 1e-8) # loss 0.00006
        # loss = compute_edge_smooth_loss(rgb.clamp(0, 1), inv_depth)
        
        return self.cfg.weight * torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
