from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossOpacityCfg:
    weight: float
    weight_mask: float = 1.0
    weight_binary: float = 1.0
    type: Literal["exp", "mean", "exp+mean"] = "exp+mean"


@dataclass
class LossOpacityCfgWrapper:
    opacity: LossOpacityCfg


class LossOpacity(Loss[LossOpacityCfg, LossOpacityCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int = 0,
        static_flag: bool = False,
    ) -> Float[Tensor, ""]:
        # 1. Alpha-Mask Consistency Loss (Original)
        alpha = prediction.alpha
        # valid_mask = batch['context']['valid_mask'].float()
        valid_mask = torch.ones_like(alpha, device=alpha.device).bool()
        
        # car_cam_mask = batch["context"]["car_cam_mask"]
        # if car_cam_mask.dim() == 5 and car_cam_mask.shape[2] == 1:
        #     car_cam_mask = car_cam_mask[:, :, 0]
        # valid_mask = valid_mask & car_cam_mask.bool()
        
        mask_loss = F.mse_loss(alpha, valid_mask.float(), reduction='none').mean()
        # if self.cfg.type == "exp":
        #     opacity_loss = torch.exp(-(gaussians.opacities - 0.5) ** 2 / 0.05).mean()
        # elif self.cfg.type == "mean":
        #     opacity_loss = gaussians.opacities.mean()
        # elif self.cfg.type == "exp+mean":
        #     opacity_loss = 0.5 * torch.exp(-(gaussians.opacities - 0.5) ** 2 / 0.05).mean() + gaussians.opacities.mean()
        # 2. Binary Opacity Constraint Loss (New: op * (1 - op))
        opacities = gaussians.opacities
        binary_loss = torch.mean(opacities * (1.0 - opacities))

        # Combined weighted loss
        total_loss = (self.cfg.weight_mask * mask_loss + 
                      self.cfg.weight_binary * binary_loss)

        return self.cfg.weight * torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
