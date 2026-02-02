from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossDynamicMaskCfg:
    weight: float
    mode: str = "bce"  # options: 'bce', 'mse'
    eps: float = 1e-6


@dataclass
class LossDynamicMaskCfgWrapper:
    dynamic_mask: LossDynamicMaskCfg


class LossDynamicMask(Loss[LossDynamicMaskCfg, LossDynamicMaskCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        pred_mask = depth_dict.get("dynamic_conf", None)
        # 1. 获取预测的 Dynamic Mask
        # 假设上一轮 Encoder 修改后，mask 存储在 distill_infos 中
        if pred_mask is None:
            # 如果没有预测值，为了不报错返回 0，或者你可以选择抛出异常
            return torch.tensor(0.0, device=prediction.color.device)
            
            # pred_mask 来自 Encoder，通常 shape 为 (B, V, H, W)
            # 且值范围应该是 [0, 1] (经过 Sigmoid)

        # 2. 获取 GT Mask
        # shape 为 (B, V, 1, H, W)
        # 注意：dataset 中 images 是 stack 起来的，这里 batch['context'] 中的数据结构通常是 (B, V, ...)
        gt_mask = batch["context"]["fine_dynamic_masks"]

        # 3. 对齐维度
        # 如果 pred 是 (B, V, H, W)，需要增加一个通道维度匹配 GT
        if pred_mask.dim() == 4 and gt_mask.dim() == 5:
            pred_mask = pred_mask.unsqueeze(2)  # (B, V, 1, H, W) # TODO check here whether run in
        
        if pred_mask.dim() == 5 and gt_mask.dim() == 4:
            gt_mask = gt_mask.unsqueeze(2)  # (B, V, 1, H, W)
        
        # 确保 pred 和 gt 的空间尺寸一致
        # 如果 Encoder 下采样了 (例如 patch 模式)，可能需要插值 GT
        if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
            gt_mask = F.interpolate(
                gt_mask.flatten(0, 1), 
                size=pred_mask.shape[-2:], 
                mode="nearest"
            ).view_as(pred_mask)
        ### 对齐数据类型
        gt_mask = gt_mask.to(pred_mask.dtype)
        # 4. 计算 Loss
        # 论文方法: Cross Entropy (BCE)
        if self.cfg.mode == "bce":
            # 确保 pred 在 0-1 之间 (防止数值不稳定)
            pred_mask = torch.clamp(pred_mask, min=self.cfg.eps, max=1.0 - self.cfg.eps)
            
            # 由于 dynamic_mask 是 dense 的，我们通常对所有像素计算 loss
            loss = F.binary_cross_entropy(pred_mask, gt_mask, reduction='mean')
            
        elif self.cfg.mode == "focal":
            # Focal Loss Implementation
            # Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            alpha = 0.25
            gamma = 2.0
            
            # Clamp predictions to prevent log(0) errors
            probs = pred_mask.clamp(min=1e-6, max=1 - 1e-6)
            
            # Calculate Binary Cross Entropy element-wise
            bce_loss = -(gt_mask * torch.log(probs) + (1 - gt_mask) * torch.log(1 - probs))
            
            # pt is the probability of the true class (p if y=1, 1-p if y=0)
            # effectively pt = exp(-bce_loss)
            pt = torch.exp(-bce_loss)
            
            # alpha_t balancing factor
            alpha_t = alpha * gt_mask + (1 - alpha) * (1 - gt_mask)
            
            # Focal Loss
            focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
            loss = focal_loss.mean()
            
        else:
            raise ValueError(f"Unknown loss mode: {self.cfg.mode}")

        return self.cfg.weight * loss