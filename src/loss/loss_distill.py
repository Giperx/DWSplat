import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy

from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.encoder.vggt.utils.rotation import mat_to_quat
from src.utils.point import get_normal_map

def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3
        
        quat = mat_to_quat(R)
        # Note the order of h and w here
        # H, W = image_size_hw
        # fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        # fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        fov_h = 2 * torch.atan(0.5 / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan(0.5 / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding
    
def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)

class DistillLoss(nn.Module):
    def __init__(self, delta=1.0, gamma=0.6, weight_pose=1.0, weight_depth=1.0, weight_normal=1.0, weight_depth_gradient=0.5, weight_depth_norm_head_mse=0.5, weight_depth_norm_head_gradient=0.5, weight_depth_scale=0.1, weight_depth_l1=0.0, weight_depth_edge_aware_log_l1=0.0, weight_depth_edge_aware_gradient=0.0, distill_warmup_steps=3000, distill_warmup_start=0.2, distill_warmup_end=1.0, weight_depth_decay_start_step=20000, weight_depth_decay_end_step=100000, weight_depth_decay_initial=0.05, weight_depth_decay_final=0.01):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight_pose = weight_pose
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal
        self.weight_depth_gradient = weight_depth_gradient
        self.weight_depth_norm_head_mse = weight_depth_norm_head_mse
        self.weight_depth_norm_head_gradient = weight_depth_norm_head_gradient
        self.weight_depth_scale = weight_depth_scale
        self.weight_depth_edge_aware_log_l1 = weight_depth_edge_aware_log_l1
        self.weight_depth_l1 = weight_depth_l1
        self.weight_depth_edge_aware_gradient = weight_depth_edge_aware_gradient
        self.distill_warmup_steps = distill_warmup_steps
        self.distill_warmup_start = distill_warmup_start
        self.distill_warmup_end = distill_warmup_end
        self.weight_depth_decay_start_step = weight_depth_decay_start_step
        self.weight_depth_decay_end_step = weight_depth_decay_end_step
        self.weight_depth_decay_initial = weight_depth_decay_initial
        self.weight_depth_decay_final = weight_depth_decay_final

    def get_distill_warmup_scale(self, global_step, device):
        if self.distill_warmup_steps <= 0:
            return torch.tensor(self.distill_warmup_end, device=device)

        progress = min(max(global_step / self.distill_warmup_steps, 0.0), 1.0)
        scale = self.distill_warmup_start + progress * (
            self.distill_warmup_end - self.distill_warmup_start
        )
        return torch.tensor(scale, device=device)

    def gradient_loss(self, gs_depth, target_depth, target_valid_mask):
        diff = gs_depth - target_depth

        grad_x_diff = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        grad_y_diff = diff[:, :, 1:, :] - diff[:, :, :-1, :]

        # Use & for boolean logic, then convert to float for arithmetic
        mask_x = (target_valid_mask[:, :, :, 1:] & target_valid_mask[:, :, :, :-1]).float()
        mask_y = (target_valid_mask[:, :, 1:, :] & target_valid_mask[:, :, :-1, :]).float()

        grad_x_diff = grad_x_diff * mask_x
        grad_y_diff = grad_y_diff * mask_y

        grad_x_diff = grad_x_diff.clamp(min=-100, max=100)
        grad_y_diff = grad_y_diff.clamp(min=-100, max=100)

        loss_x = grad_x_diff.abs().sum()
        loss_y = grad_y_diff.abs().sum()
        num_valid = mask_x.sum() + mask_y.sum()

        if num_valid == 0:
            return torch.tensor(0.0, device=gs_depth.device)

        gradient_loss = (loss_x + loss_y) / (num_valid + 1e-6)
        return gradient_loss

    def edge_aware_log_l1_loss(self, pred_depth, target_depth, rgb, target_valid_mask, beta=15.0):
        if pred_depth.ndim == 5 and pred_depth.shape[2] == 1:
            pred_depth = pred_depth.squeeze(2)
        if target_depth.ndim == 5 and target_depth.shape[2] == 1:
            target_depth = target_depth.squeeze(2)
        if target_valid_mask.ndim == 5 and target_valid_mask.shape[2] == 1:
            target_valid_mask = target_valid_mask.squeeze(2)

        if pred_depth.ndim == 4:
            pred_depth = pred_depth.flatten(0, 1)
        if target_depth.ndim == 4:
            target_depth = target_depth.flatten(0, 1)
        if target_valid_mask.ndim == 4:
            target_valid_mask = target_valid_mask.flatten(0, 1)

        logl1 = torch.log(1 + torch.abs(pred_depth - target_depth))
        
        if rgb.ndim == 5: # B, V, 3, H, W
            rgb = rgb.flatten(0, 1)
        if rgb.shape[1] == 3: # (B*V), 3, H, W
            rgb = rgb.permute(0, 2, 3, 1) # (B*V), H, W, 3
            
        grad_img_x = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), -1)
        grad_img_y = torch.mean(torch.abs(rgb[:, :-1, :, :] - rgb[:, 1:, :, :]), -1)
        
        lambda_x = torch.exp(-grad_img_x * beta)
        lambda_y = torch.exp(-grad_img_y * beta)
        
        loss_x = lambda_x * logl1[:, :, :-1]
        loss_y = lambda_y * logl1[:, :-1, :]
        
        mask_x = target_valid_mask[:, :, :-1] & target_valid_mask[:, :, 1:]
        mask_y = target_valid_mask[:, :-1, :] & target_valid_mask[:, 1:, :]
        
        loss_x = loss_x[mask_x]
        loss_y = loss_y[mask_y]
        
        if loss_x.numel() == 0 or loss_y.numel() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
            
        return loss_x.mean() + loss_y.mean()

    def edge_aware_gradient_loss(self, pred_depth, target_depth, rgb, target_valid_mask, beta=15.0):
        if pred_depth.ndim == 5 and pred_depth.shape[2] == 1:
            pred_depth = pred_depth.squeeze(2)
        if target_depth.ndim == 5 and target_depth.shape[2] == 1:
            target_depth = target_depth.squeeze(2)
        if target_valid_mask.ndim == 5 and target_valid_mask.shape[2] == 1:
            target_valid_mask = target_valid_mask.squeeze(2)

        if pred_depth.ndim == 4:
            pred_depth = pred_depth.flatten(0, 1)
        if target_depth.ndim == 4:
            target_depth = target_depth.flatten(0, 1)
        if target_valid_mask.ndim == 4:
            target_valid_mask = target_valid_mask.flatten(0, 1)

        diff = pred_depth - target_depth
        
        grad_x_diff = diff[:, :, :-1] - diff[:, :, 1:]
        grad_y_diff = diff[:, :-1, :] - diff[:, 1:, :]
        
        if rgb.ndim == 5: # B, V, 3, H, W
            rgb = rgb.flatten(0, 1)
        if rgb.shape[1] == 3: # (B*V), 3, H, W
            rgb = rgb.permute(0, 2, 3, 1) # (B*V), H, W, 3
            
        grad_img_x = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), -1)
        grad_img_y = torch.mean(torch.abs(rgb[:, :-1, :, :] - rgb[:, 1:, :, :]), -1)
        
        lambda_x = torch.exp(-grad_img_x * beta)
        lambda_y = torch.exp(-grad_img_y * beta)
        
        mask_x = target_valid_mask[:, :, :-1] & target_valid_mask[:, :, 1:]
        mask_y = target_valid_mask[:, :-1, :] & target_valid_mask[:, 1:, :]
        
        grad_x_filtered = grad_x_diff[mask_x] * lambda_x[mask_x]
        grad_y_filtered = grad_y_diff[mask_y] * lambda_y[mask_y]
        
        if grad_x_filtered.numel() == 0 and grad_y_filtered.numel() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        
        return (grad_x_filtered.abs().mean() + grad_y_filtered.abs().mean())

    def align_mask_to_ref(self, mask, ref):
        if ref.ndim == 5 and ref.shape[-1] == 1:
            ref = ref.squeeze(-1)
        if mask.ndim == ref.ndim + 1 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.ndim + 1 == ref.ndim and ref.shape[-1] == 1:
            ref = ref.squeeze(-1)
        if mask.ndim == ref.ndim and mask.shape[-2:] == (ref.shape[-1], ref.shape[-2]):
            mask = mask.transpose(-1, -2)
        if mask.shape != ref.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match reference shape {ref.shape}")
        return mask.bool()

    def scale_anchor_loss(self, pred_depth, target_depth, target_valid_mask):
        if target_valid_mask.sum() == 0:
            target_valid_mask = torch.ones_like(target_valid_mask, dtype=torch.bool)
        pred_sel = pred_depth[target_valid_mask]
        target_sel = target_depth[target_valid_mask]
        pred_med = pred_sel.median()
        target_med = target_sel.median().clamp_min(1e-6)
        return (pred_med - target_med).abs() / target_med
    
    def camera_loss_single(self, cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
        if loss_type == "l1":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
        elif loss_type == "l2":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
        elif loss_type == "huber":
            loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
            loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
            loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_T = torch.nan_to_num(loss_T, nan=0.0, posinf=0.0, neginf=0.0)
        loss_R = torch.nan_to_num(loss_R, nan=0.0, posinf=0.0, neginf=0.0)
        loss_fl = torch.nan_to_num(loss_fl, nan=0.0, posinf=0.0, neginf=0.0)

        loss_T = torch.clamp(loss_T, min=-100, max=100)
        loss_R = torch.clamp(loss_R, min=-100, max=100)
        loss_fl = torch.clamp(loss_fl, min=-100, max=100)

        loss_T = loss_T.mean()
        loss_R = loss_R.mean()
        loss_fl = loss_fl.mean()
        
        return loss_T, loss_R, loss_fl

    def forward(self, pred_pose_enc_list, prediction, batch, depth_dict, global_step=0):
        loss_pose = 0.0

        # if pred_pose_enc_list is not None:
        #     num_predictions = len(pred_pose_enc_list)
        #     pesudo_gt_pose_enc = depth_dict['distill_infos']['pred_pose_enc_list']
        #     for i in range(num_predictions):
        #         i_weight = self.gamma ** (num_predictions - i - 1)
        #         cur_pred_pose_enc = pred_pose_enc_list[i]
        #         cur_pesudo_gt_pose_enc = pesudo_gt_pose_enc[i]
        #         loss_pose += i_weight * huber_loss(cur_pred_pose_enc, cur_pesudo_gt_pose_enc).mean()
        #     loss_pose = loss_pose / num_predictions
        #     loss_pose = torch.nan_to_num(loss_pose, nan=0.0, posinf=0.0, neginf=0.0)
        
        pred_depth = prediction.depth
        pesudo_gt_depth = depth_dict['distill_infos']['depth_map'].squeeze(-1)
        # Detach and convert to bool to avoid autograd issues
        conf_mask = depth_dict['distill_infos']['conf_mask'].detach().bool()
        
        car_cam_mask = batch["context"]["car_cam_mask"]
        if car_cam_mask.dim() == 5 and car_cam_mask.shape[2] == 1:
            car_cam_mask = car_cam_mask[:, :, 0]
        conf_mask = conf_mask & car_cam_mask.bool().detach()
        
        if batch['context']['valid_mask'].sum() > 0:
            conf_mask = batch['context']['valid_mask'].detach().bool()
        conf_mask = self.align_mask_to_ref(conf_mask, pred_depth)
        if conf_mask.sum() == 0:
            conf_mask = torch.ones_like(conf_mask, dtype=torch.bool)
        

        
        distill_scale = self.get_distill_warmup_scale(global_step, pred_depth.device)
        
        # Calculate dynamic weight for depth_l1 based on global_step
        if global_step < self.weight_depth_decay_start_step:
            current_weight_depth_l1 = self.weight_depth_decay_initial
        else:
            decay_progress = min(
                (global_step - self.weight_depth_decay_start_step)
                / max(self.weight_depth_decay_end_step - self.weight_depth_decay_start_step, 1),
                1.0,
            )
            current_weight_depth_l1 = (
                self.weight_depth_decay_initial
                + decay_progress * (self.weight_depth_decay_final - self.weight_depth_decay_initial)
            )
            
        # depth_loss_l1 = torch.tensor(0.0, device=pred_depth.device)
        depth_loss_l1 = (
            torch.abs(pred_depth - pesudo_gt_depth)[conf_mask].mean()
        ) * self.weight_depth_l1 * self.weight_depth * distill_scale
        
        decay_ratio = current_weight_depth_l1 / self.weight_depth_decay_initial
        # depth_loss_gradient = torch.tensor(0.0, device=pred_depth.device)
        depth_loss_gradient = (
            self.gradient_loss(pred_depth, pesudo_gt_depth, conf_mask)
        ) * self.weight_depth_gradient * self.weight_depth * distill_scale * decay_ratio
        depth_loss_scale = (
            self.scale_anchor_loss(pred_depth, pesudo_gt_depth, conf_mask)
        ) * self.weight_depth_scale * self.weight_depth * distill_scale

        rgb = (batch["context"]["image"] + 1) / 2
        # debug print
        # print("shape_rgb:", rgb.shape, "shape_conf_mask:", conf_mask.shape, "shape_pred_depth:", pred_depth.shape, "shape_pesudo_gt_depth:", pesudo_gt_depth.shape)
        # shape_rgb: torch.Size([1, 3, 3, 294, 518]) shape_conf_mask: torch.Size([1, 3, 294, 518]) shape_pred_depth: torch.Size([1, 3, 294, 518]) shape_pesudo_gt_depth: torch.Size([1, 3, 294, 518])
        depth_loss_edge_aware_log_l1 = (
            self.edge_aware_log_l1_loss(pred_depth, pesudo_gt_depth, rgb, conf_mask, beta=15.0)
        ) * self.weight_depth_edge_aware_log_l1 * self.weight_depth * distill_scale * decay_ratio
        
        depth_loss_edge_aware_gradient = (
            self.edge_aware_gradient_loss(pred_depth, pesudo_gt_depth, rgb, conf_mask, beta=15.0)
        ) * self.weight_depth_edge_aware_gradient * self.weight_depth * distill_scale * decay_ratio

        loss_depth = depth_loss_l1 + depth_loss_gradient + depth_loss_scale + depth_loss_edge_aware_log_l1 + depth_loss_edge_aware_gradient
        
        pred_depth =pred_depth.flatten(0, 1)
        pesudo_gt_depth = pesudo_gt_depth.flatten(0, 1)
        # conf_mask = conf_mask.flatten(0, 1)

        # loss_depth = F.mse_loss(pred_depth[conf_mask], pesudo_gt_depth[conf_mask], reduction='none').mean()
      
        pred_depth_from_head = depth_dict['depth_map_norm']
        pesudo_gt_depth_from_head = depth_dict['distill_infos']['depth_map_norm']
        if pred_depth_from_head.ndim == 5 and pred_depth_from_head.shape[-1] == 1:
            pred_depth_from_head = pred_depth_from_head.squeeze(-1)
        if pesudo_gt_depth_from_head.ndim == 5 and pesudo_gt_depth_from_head.shape[-1] == 1:
            pesudo_gt_depth_from_head = pesudo_gt_depth_from_head.squeeze(-1)
        conf_mask_norm = self.align_mask_to_ref(conf_mask, pred_depth_from_head)
        if conf_mask_norm.sum() == 0:
            conf_mask_norm = torch.ones_like(conf_mask_norm, dtype=torch.bool)
        # print("shape_pred_depth_from_head:", pred_depth_from_head.shape, "shape_pesudo_gt_depth_from_head:", pesudo_gt_depth_from_head.shape, "shape_conf_mask:", conf_mask.shape)
        loss_depth_norm_head_gradient = (
            self.gradient_loss(pred_depth_from_head, pesudo_gt_depth_from_head, conf_mask_norm)
        ) * self.weight_depth_norm_head_gradient
        loss_depth_norm_head_mse = (
            F.mse_loss(pred_depth_from_head[conf_mask_norm], pesudo_gt_depth_from_head[conf_mask_norm], reduction='none').mean()
        ) * self.weight_depth_norm_head_mse
        
        # render_normal = get_normal_map(pred_depth, batch["context"]["intrinsics"].flatten(0, 1))
        # pred_normal = get_normal_map(pesudo_gt_depth, batch["context"]["intrinsics"].flatten(0, 1))
       
        # alpha1_loss = (1 - (render_normal[conf_mask] * pred_normal[conf_mask]).sum(-1)).mean()
        # alpha2_loss = F.l1_loss(render_normal[conf_mask], pred_normal[conf_mask], reduction='mean')
        # loss_normal = (alpha1_loss + alpha2_loss) / 2
        # loss_pose = loss_pose * self.weight_pose
        loss_distill = loss_pose + loss_depth + loss_depth_norm_head_mse + loss_depth_norm_head_gradient
        loss_distill = torch.nan_to_num(loss_distill, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss_dict = {
            "loss_distill": loss_distill,
            "loss_distill_scale": distill_scale,
            # "loss_pose": loss_pose * self.weight_pose,
            "loss_depth": loss_depth,
            "loss_depth_norm_head_mse": loss_depth_norm_head_mse,
            "loss_depth_norm_head_gradient": loss_depth_norm_head_gradient,
            "loss_depth_l1": depth_loss_l1,
            "loss_depth_gradient": depth_loss_gradient,
            "loss_depth_scale": depth_loss_scale,
            "loss_depth_edge_aware_log_l1": depth_loss_edge_aware_log_l1,
            "loss_depth_edge_aware_gradient": depth_loss_edge_aware_gradient,
            # "loss_normal": loss_normal * self.weight_normal
        }

        return loss_dict
