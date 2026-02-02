"""
Utility to load HuggingFace pretrained AnySplat models for fine-tuning.
Handles conversion between HF format and Lightning checkpoint format.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.nn import Module

logger = logging.getLogger(__name__)


def is_hf_model_directory(path: str) -> bool:
    """
    Check if a path is a HuggingFace model directory.
    
    HuggingFace directories typically contain:
    - config.json
    - model.safetensors or pytorch_model.bin
    
    Args:
        path: Path to check
        
    Returns:
        True if path is a HF model directory
    """
    if not os.path.isdir(path):
        return False
    
    path = Path(path)
    
    # Check for HF model files
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors").exists() or
        (path / "pytorch_model.bin").exists() or
        (path / "model-*.safetensors").exists() or
        any(path.glob("pytorch_model*.bin"))
    )
    
    return has_config and has_weights


def is_lightning_checkpoint(path: str) -> bool:
    """
    Check if a path is a Lightning checkpoint file.
    
    Args:
        path: Path to check
        
    Returns:
        True if path is a .ckpt file
    """
    return isinstance(path, str) and path.endswith((".ckpt", ".pt", ".pth"))


def load_hf_model_weights(
    flag_gaussian_head: bool,
    model: Module,
    hf_model_path: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Load weights from HuggingFace model directory into a model.
    Only loads weights for aggregator, camera_head, and depth_head.
    
    Args:
        model: PyTorch model to load weights into
        hf_model_path: Path to HuggingFace model directory
        strict: Whether to strictly require all keys to match (Ignored for partial loading)
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    logger.info(f"Loading HuggingFace model from: {hf_model_path}")
    
    hf_path = Path(hf_model_path)
    
    # Try to load safetensors first, then pytorch_model.bin
    weights_path = None
    if (hf_path / "model.safetensors").exists():
        weights_path = hf_path / "model.safetensors"
        logger.info("Found model.safetensors, loading...")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        except ImportError:
            logger.warning("safetensors not installed, trying pytorch_model.bin")
            weights_path = None
    
    if weights_path is None:
        # Try pytorch_model.bin
        pytorch_bin = list(hf_path.glob("pytorch_model*.bin"))
        if pytorch_bin:
            weights_path = pytorch_bin[0]
            logger.info(f"Loading {weights_path.name}...")
            state_dict = torch.load(str(weights_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model weights found in {hf_model_path}. "
                "Expected model.safetensors or pytorch_model.bin"
            )
    
    # Filter state_dict to only include specific components
    target_components = ["aggregator", "camera_head", "depth_head"]
    if flag_gaussian_head:
        target_components.append("gaussian_param_head")
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 2. 检查键名是否包含目标组件
        if any(component in k for component in target_components):
            # 3. 核心修复：如果 model 对象内部没有 encoder 层级，需要去掉 'encoder.' 前缀
            # 建议增加一个判断或统一处理
            new_key = k
            if k.startswith("encoder."):
                # 只有当你发现加载时 Missing Keys 很多，才启用下面这行代码
                # new_key = k.replace("encoder.", "", 1) 
                pass
            
            filtered_state_dict[new_key] = v
            
    logger.info(f"Filtered weights to only include: {target_components}")
    logger.info(f"Kept {len(filtered_state_dict)}/{len(state_dict)} keys")

    # Load state dict into model
    # We enforce strict=False because we are intentionally doing a partial load
    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state_dict,
        strict=False
    )
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
    
    logger.info(f"Successfully loaded filtered HuggingFace model weights")
    return missing_keys, unexpected_keys


def prepare_checkpoint_path(
    flag_gaussian_head: bool,
    checkpoint_path: Optional[str],
    model: Optional[Module] = None,
) -> tuple[Optional[str], bool]:
    """
    Prepare checkpoint path for Lightning training.
    
    If checkpoint_path is a HuggingFace model directory:
    - Load weights into model (if provided)
    - Return None (don't pass to Lightning, as it's not a .ckpt file)
    
    If checkpoint_path is a Lightning checkpoint:
    - Return as-is for Lightning to load
    
    Args:
        checkpoint_path: Path to checkpoint or HF model
        model: Model to load HF weights into (optional)
        
    Returns:
        Tuple of (checkpoint_path_for_lightning, is_hf_pretrained)
        - checkpoint_path_for_lightning: Path to pass to trainer.fit(), or None
        - is_hf_pretrained: Whether weights were loaded from HF
    """
    if checkpoint_path is None:
        return None, False
    
    checkpoint_path = str(checkpoint_path)
    
    if is_hf_model_directory(checkpoint_path):
        logger.info(f"Detected HuggingFace model directory: {checkpoint_path}")
        
        if model is not None:
            load_hf_model_weights(flag_gaussian_head, model, checkpoint_path, strict=False)
        
        # Return None so Lightning doesn't try to load it as a checkpoint
        return None, True
    
    elif is_lightning_checkpoint(checkpoint_path):
        logger.info(f"Using Lightning checkpoint: {checkpoint_path}")
        return checkpoint_path, False
    
    else:
        logger.warning(
            f"Unknown checkpoint format: {checkpoint_path}\n"
            f"Expected either a .ckpt file or HuggingFace model directory"
        )
        return checkpoint_path, False


def create_lightning_checkpoint_from_model(
    model: Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: int = 0,
    output_path: str = "checkpoint.ckpt",
) -> str:
    """
    Create a Lightning checkpoint from a model and optimizer.
    
    Useful for converting HF pretrained weights to Lightning format.
    
    Args:
        model: The model to checkpoint
        optimizer: Optional optimizer state
        global_step: Current training step
        output_path: Where to save the checkpoint
        
    Returns:
        Path to created checkpoint
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "global_step": global_step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_states"] = [optimizer.state_dict()]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, str(output_path))
    logger.info(f"Saved Lightning checkpoint to: {output_path}")
    
    return str(output_path)