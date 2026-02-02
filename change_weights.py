
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
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
    model = AnySplat(cfg.model.encoder, cfg.model.decoder)
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
    
    model.save_pretrained(cfg.output_path)


if __name__ == "__main__":
    evaluate()

# cat epoch_0-step_17000.part.* > epoch_0-step_17000.tar.gz
# 260129_singleFramesReTrainGSHeadEpoch1Iter10000  260129_singleFramesReTrainGSHeadEpoch2Iter25000
# CUDA_VISIBLE_DEVICES=0 python change_weights.py +experiment=nuscenes +output_path=/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/finetune_weights/260129_singleFramesReTrainGSHeadEpoch2Iter25000/weights checkpointing.load=/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/finetune_weights/260129_singleFramesReTrainGSHeadEpoch2Iter25000/epoch_1-step_10000.ckpt

