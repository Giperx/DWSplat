
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.model.model.anysplat import AnySplat
from src.model.model_wrapper import TestCfg

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, ModelCfg, CheckpointingCfg, separate_loss_cfg_wrappers, \
    separate_dataset_cfg_wrappers
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfgWrapper
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.global_cfg import set_cfg
    

@dataclass
class RootCfg:
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    seed: int
    output_path: Path
    test: TestCfg
    
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def evaluate(cfg_dict: DictConfig):
    
    cfg = load_typed_config(cfg_dict, RootCfg,
                            {list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers},)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    print_log_every_n_steps = OmegaConf.select(
        cfg_dict,
        "train.print_log_every_n_steps",
        default=OmegaConf.select(cfg_dict, "trainer.log_every_n_steps", default=50),
    )
    model = AnySplat(
        cfg.model.encoder,
        cfg.model.decoder,
        int(print_log_every_n_steps),
    )
    ckpt_weights = torch.load(cfg.checkpointing.load, map_location='cpu')['state_dict']
    # remove the prefix "encoder.", need to judge if is at start of key
    ckpt_weights = {k[8:] if k.startswith("encoder.") else k: v for k, v in ckpt_weights.items()}
    ckpt_weights = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt_weights.items()}
    missing_keys, unexpected_keys = model.load_state_dict(ckpt_weights, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    if not cfg.output_path.exists():
        cfg.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_path)


if __name__ == "__main__":
    evaluate()

# cat epoch_0-step_17000.part.* > epoch_0-step_17000.tar.gz
# 260129_singleFramesReTrainGSHeadEpoch1Iter10000  260129_singleFramesReTrainGSHeadEpoch2Iter25000
# CUDA_VISIBLE_DEVICES=4 python change_weights.py +experiment=nuscenes +output_path='outputs/exp_OmniVGGT_nuScenes_omnivggt_finetune/2026-04-24_10-27-16_distill_recon_dwsplat_woLora_wg_e5s1w2_vol0.024scale0.03+5.0/checkpoints/epoch_5-step_40000_safe' checkpointing.load='outputs/exp_OmniVGGT_nuScenes_omnivggt_finetune/2026-04-24_10-27-16_distill_recon_dwsplat_woLora_wg_e5s1w2_vol0.024scale0.03+5.0/checkpoints/epoch_5-step_40000.ckpt'

