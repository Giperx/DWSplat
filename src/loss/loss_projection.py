from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from jaxtyping import Float
from torch import Tensor
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss

_DEBUG_DIR = "tmp_debug_projection_loss"
_DEBUG_SAVED = False


@dataclass
class LossProjectionCfg:
    weight: float = 0.1
    norm_mode: str = "mean_std"


@dataclass
class LossProjectionCfgWrapper:
    projection: LossProjectionCfg


def warp_image(src_img, src_mask, ref_depth, ref_K, cam_T_cam, ref_mask, src_dynamic_mask=None, ref_dynamic_mask=None):
    B, C, H, W = src_img.shape
    device = src_img.device

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    fx = ref_K[:, 0, 0].view(B, 1, 1)
    fy = ref_K[:, 1, 1].view(B, 1, 1)
    cx = ref_K[:, 0, 2].view(B, 1, 1)
    cy = ref_K[:, 1, 2].view(B, 1, 1)

    z = ref_depth[:, 0]
    x3d = (u_coords - cx.squeeze()) * z / fx.squeeze()
    y3d = (v_coords - cy.squeeze()) * z / fy.squeeze()
    points_3d = torch.stack([x3d, y3d, z], dim=-1)

    R = cam_T_cam[:3, :3].unsqueeze(0)
    t = cam_T_cam[:3, 3].unsqueeze(0).unsqueeze(0)
    points_src = (points_3d @ R.transpose(-1, -2)) + t

    x_proj = points_src[..., 0] / (points_src[..., 2] + 1e-8)
    y_proj = points_src[..., 1] / (points_src[..., 2] + 1e-8)
    u_src = x_proj * fx.squeeze() + cx.squeeze()
    v_src = y_proj * fy.squeeze() + cy.squeeze()

    u_norm = 2.0 * u_src / (W - 1) - 1.0
    v_norm = 2.0 * v_src / (H - 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1)

    warped_img = F.grid_sample(src_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_mask = F.grid_sample(src_mask, grid, mode='nearest', padding_mode='zeros', align_corners=True)

    if src_dynamic_mask is not None:
        warped_dynamic_mask = F.grid_sample(1.0 - src_dynamic_mask, grid, mode='nearest',
                                            padding_mode='zeros', align_corners=True)
    else:
        warped_dynamic_mask = torch.ones_like(warped_mask)

    warped_img = torch.nan_to_num(warped_img, nan=2.0)
    warped_mask = torch.nan_to_num(warped_mask, nan=0.0)
    warped_dynamic_mask = torch.nan_to_num(warped_dynamic_mask, nan=0.0)

    in_bounds = (
        (u_norm > -1) & (u_norm < 1) &
        (v_norm > -1) & (v_norm < 1)
    ).unsqueeze(1)
    final_mask = warped_mask * in_bounds * ref_mask * warped_dynamic_mask
    if ref_dynamic_mask is not None:
        final_mask = final_mask * (1.0 - ref_dynamic_mask)

    return warped_img, final_mask, grid


def normalize_warped_image(ref_img, warped_img, mask):
    mask_bool = mask.bool()
    if mask_bool.shape[1] != 3:
        mask_bool = mask_bool.repeat(1, 3, 1, 1)

    mask_sum = mask_bool.sum(dim=(-3, -2, -1))
    if torch.any(mask_sum == 0):
        return warped_img

    s_mean = (ref_img * mask_bool).sum(dim=(-3, -2, -1), keepdim=True) / (mask_bool.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8)
    s_var = ((ref_img - s_mean) ** 2 * mask_bool).sum(dim=(-3, -2, -1), keepdim=True) / (mask_bool.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8)
    s_std = torch.sqrt(s_var + 1e-16)

    w_mean = (warped_img * mask_bool).sum(dim=(-3, -2, -1), keepdim=True) / (mask_bool.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8)
    w_var = ((warped_img - w_mean) ** 2 * mask_bool).sum(dim=(-3, -2, -1), keepdim=True) / (mask_bool.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8)
    w_std = torch.sqrt(w_var + 1e-16)

    norm_warped = (warped_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
    return norm_warped * mask


class LossProjection(Loss[LossProjectionCfg, LossProjectionCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        static_flag: bool = False,
    ) -> Float[Tensor, ""]:
        if "aux" not in batch:
            return torch.tensor(0.0, device=prediction.color.device)

        aux_images = batch["aux"]["image"]
        aux_car_masks = batch["aux"]["car_cam_mask"]
        aux_dynamic_masks = batch["aux"]["dynamic_mask"]
        cam_T_cam_all = batch["aux"]["cam_T_cam"]

        # Use GT images as reference (not rendered prediction)
        ref_images = (batch["context"]["image"] + 1) / 2  # [B, V, 3, H, W] GT in [0,1]
        ref_depth = prediction.depth.unsqueeze(2)  # [B, V, 1, H, W]

        B, V, _, H, W = ref_images.shape

        ref_intrinsics = batch["context"]["intrinsics"]
        ref_car_masks = batch["context"]["car_cam_mask"]
        ref_dynamic_masks = batch["context"]["dynamic_mask"]

        total_loss = torch.tensor(0.0, device=ref_images.device)
        count = 0

        global _DEBUG_SAVED
        if not _DEBUG_SAVED:
            os.makedirs(_DEBUG_DIR, exist_ok=True)

        n_cams = V  # 3 cameras per timestep
        # aux layout: [ts0_cam0, ts0_cam1, ts0_cam2, ts2_cam0, ts2_cam1, ts2_cam2]
        # For ref v_idx, only pair with aux_idx=v_idx (ts0 same cam) and aux_idx=v_idx+n_cams (ts2 same cam)

        for b in range(B):
            for v_idx in range(V):
                # Same-camera pairs: ts0 and ts2
                paired_aux_indices = [v_idx, v_idx + n_cams]

                for aux_idx in paired_aux_indices:
                    src_img = aux_images[b:b+1, aux_idx]
                    src_dyn = aux_dynamic_masks[b:b+1, aux_idx]
                    src_car = aux_car_masks[b:b+1, aux_idx]
                    ref_dyn = ref_dynamic_masks[b:b+1, v_idx]

                    warped, mask, grid = warp_image(
                        src_img, src_car,
                        ref_depth[b:b+1, v_idx],
                        ref_intrinsics[b:b+1, v_idx],
                        cam_T_cam_all[b, aux_idx],
                        ref_car_masks[b:b+1, v_idx],
                        src_dynamic_mask=src_dyn,
                        ref_dynamic_mask=ref_dyn,
                    )

                    if mask.sum() < 1:
                        continue

                    warped_norm = warped
                    if self.cfg.norm_mode == "mean_std":
                        warped_norm = normalize_warped_image(
                            ref_images[b:b+1, v_idx], warped, mask
                        )

                    # --- Debug visualization ---
                    if not _DEBUG_SAVED and global_step % 1000 == 0:
                        ref_img = ref_images[b:b+1, v_idx]  # GT

                        # 3. warped src with src masks applied (car+dynamic -> black)
                        # warp the src valid mask to ref view same as warp_image
                        src_valid_for_vis = src_car * (1.0 - src_dyn)  # [1,1,H,W]
                        warped_src_valid = F.grid_sample(
                            src_valid_for_vis, grid, mode='nearest',
                            padding_mode='zeros', align_corners=True
                        )
                        warped_src_vis = warped_norm * warped_src_valid

                        # 4. ref with ref masks applied (car+dynamic -> black)
                        ref_valid = ref_car_masks[b:b+1, v_idx] * (1.0 - ref_dyn)
                        ref_vis = ref_img * ref_valid

                        # 5. final mask (white = compute, black = skip)
                        final_mask_vis = mask.expand(-1, 3, -1, -1).float()

                        vis = torch.cat([
                            src_img,          # 1. src原图
                            warped_norm,      # 2. warp后的src
                            warped_src_vis,   # 3. 应用src mask的warp src
                            ref_vis,          # 4. 应用ref mask的ref图像
                            final_mask_vis,   # 5. 最终合并的纯mask
                        ], dim=-1)

                        save_image(vis[0], os.path.join(
                            _DEBUG_DIR,
                            f"step{global_step}_b{b}_v{v_idx}_aux{aux_idx}.png"
                        ))

                    l1 = torch.abs(warped_norm - ref_images[b:b+1, v_idx])
                    loss = (l1 * mask).sum() / (mask.sum() * 3 + 1e-8)
                    total_loss = total_loss + loss
                    count += 1

        if count > 0:
            total_loss = total_loss / count

        # if not _DEBUG_SAVED:
        #     _DEBUG_SAVED = True
        #     print(f"[ProjectionLoss] Debug images saved to {_DEBUG_DIR}/")

        return self.cfg.weight * torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
