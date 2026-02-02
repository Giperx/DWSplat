from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, List, Dict
import os
import random
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
import logging

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class DatasetNuScenesCfg(DatasetCfgCommon):
    """Configuration for nuscenes dataset loader"""
    name: str
    roots: list[Path]
    # Path to the text file containing scene IDs (e.g., nuScenes_Train.txt)
    split_file_path: Path 
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool
    numTimes: int = 1  # Number of consecutive timestamps to load


@dataclass
class DatasetNuScenesCfgWrapper:
    nuscenes: DatasetNuScenesCfg


class DatasetNuScenes(Dataset):
    """
    NuScenes dataset loader for processed 10Hz data.
    Structure:
        root/processed_10Hz/trainval/{scene_id}/
            images/{timestep:03d}_{cam_id}.jpg
            extrinsics/{timestep:03d}_{cam_id}.txt
            intrinsics/{cam_id}.txt
            fine_dynamic_masks/all/{timestep:03d}_{cam_id}.png
    """
    
    cfg: DatasetNuScenesCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    
    near: float = 0.1
    far: float = 100.0
    
    # Target size for resizing
    # TARGET_SIZE = 224  # 224 336 448
    TARGET_HEIGHT = 252
    TARGET_WIDTH = 448
    
    # Camera mapping based on file naming convention {timestep}_{cam_id}.jpg
    # 0: CAM_FRONT
    # 1: CAM_FRONT_LEFT
    # 2: CAM_FRONT_RIGHT
    # 3: CAM_BACK_LEFT
    # 4: CAM_BACK_RIGHT
    # 5: CAM_BACK
    
    # Camera groups for sampling
    CAM_GROUP_FRONT = [0, 1, 2] # FRONT, FRONT_LEFT, FRONT_RIGHT
    CAM_GROUP_BACK = [5, 4, 3]  # BACK, BACK_RIGHT, BACK_LEFT (Ordered for visual consistency if needed)

    def __init__(
        self,
        cfg: DatasetNuScenesCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.mask_to_tensor = tf.ToTensor()
        
        # Determine root directory (assuming roots[0] points to .../nuscenes/processed_10Hz/trainval or test)
        self.data_root = Path(cfg.roots[0])
        
        # Load scene list from split file
        if self.stage == "train":
            self.scene_ids = self._load_split_file(cfg.split_file_path)
        else:
            # self.scene_ids = self._load_split_file(cfg.split_file_path.replace("Train", "Val"))
            
            new_path = Path(str(cfg.split_file_path).replace("Train", "Val"))
            self.scene_ids = self._load_split_file(new_path)
        
        # Build index of valid samples
        # Each sample is (scene_id, start_timestep_index, camera_group_type)
        self.samples = [] 
        self._build_scene_index()
        
        logger.info(f"nuScenes Dataset: {self.stage}: loaded {len(self.samples)} samples from {len(self.scene_ids)} scenes")
        logger.info(f"Configuration: numTimes={self.cfg.numTimes}")

    def _load_split_file(self, split_path: Path) -> List[str]:
        """Load scene IDs from text file"""
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
            
        with open(split_path, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        return scene_ids

    def _get_available_timesteps(self, scene_path: Path) -> List[int]:
        """Find all unique timesteps in a scene based on image files"""
        image_dir = scene_path / "images"
        if not image_dir.exists():
            return []
        
        # Filenames are like {timestep:03d}_{cam_id}.jpg
        # We need to find timesteps that have all 6 cameras ideally, or at least enough for our groups
        timesteps = set()
        for f in os.listdir(image_dir):
            if f.endswith('.jpg'):
                try:
                    ts_str = f.split('_')[0]
                    timesteps.add(int(ts_str))
                except ValueError:
                    continue
        return sorted(list(timesteps))

    def _build_scene_index(self) -> None:
        """
        Builds the list of valid training/validation samples.
        A sample consists of a Scene ID and a starting timestep index.
        """
        valid_scenes = 0
        
        for scene_id in self.scene_ids:
            scene_path = self.data_root / scene_id
            if not scene_path.exists():
                logger.warning(f"Scene {scene_id} not found at {scene_path}, skipping.")
                continue
                
            timesteps = self._get_available_timesteps(scene_path)
            num_timesteps = len(timesteps)
            required_frames = self.cfg.numTimes
            
            if num_timesteps < required_frames:
                logger.debug(f"Scene {scene_id} has {num_timesteps} frames, need {required_frames}. Skipping.")
                continue

            # Check for camera intrinsics existence (assumed constant per scene/cam)
            intrinsics_path = scene_path / "intrinsics"
            if not intrinsics_path.exists():
                logger.warning(f"Intrinsics missing for {scene_id}")
                continue

            # We create samples using a sliding window or just indexing
            # Logic: For every possible start frame that allows numTimes consecutive frames
            max_start_idx = num_timesteps - required_frames
            
            # For Validation, we might want fixed samples. For training, we can be more flexible.
            # Here we add all possible sliding windows as potential samples.
            # During __getitem__, we will decide which camera group (front vs back) to use.
            # If Stage is Train, we might add logic to shuffle or augment later.
            
            # Training keeps every sliding window, validation/test only keep the first sample per scene
            if self.stage == "train":
                start_indices = range(0, max_start_idx + 1)
            else:
                start_indices = [0]
            
            for i in start_indices:
                # We store indices into the 'timesteps' list, not the timestep value itself
                self.samples.append({
                    'scene_id': scene_id,
                    'start_idx': i,
                    # 'timesteps': timesteps[i : i + required_frames]
                    'timesteps': timesteps[i : i + required_frames][::-1] ### 让最新帧排首位
                })
            
            valid_scenes += 1
            
        if valid_scenes == 0:
            logger.error("No valid scenes found! Check paths and split file.")

    def _read_intrinsics(self, scene_path: Path, cam_id: int) -> np.ndarray:
        """
        Read intrinsic txt file.
        Format: 3x3 matrix values separated by newlines or flattened.
        Example provided suggests 9 lines of floats.
        """
        # file_path = scene_path / "intrinsics" / f"{cam_id}.txt"
        # try:
        #     with open(file_path, 'r') as f:
        #         lines = [float(line.strip()) for line in f if line.strip()]
            
        #     if len(lines) != 9:
        #         raise ValueError(f"Expected 9 values for intrinsic matrix, got {len(lines)}")
            
        #     intr = np.array(lines, dtype=np.float32).reshape(3, 3)
        #     return intr
        # except Exception as e:
        #     logger.error(f"Error reading intrinsics {file_path}: {e}")
        #     raise
        file_path = scene_path / "intrinsics" / f"{cam_id}.txt"
        try:
            with open(file_path, 'r') as f:
                v = [float(line.strip()) for line in f if line.strip()]
            
            # 构造标准的 3x3 矩阵
            intr = np.eye(3, dtype=np.float32)
            
            if len(v) == 9:
                # 如果这 9 个数其实是 [fx, 0, cx, 0, fy, cy, 0, 0, 1] 这种排列
                # 或者如果是 [fx, fy, cx, cy, ...] 这种排列：
                # 根据你 txt 的内容，最稳妥的映射方式是：
                intr[0, 0] = v[0] # fx
                intr[1, 1] = v[1] # fy
                intr[0, 2] = v[2] # cx
                intr[1, 2] = v[3] # cy
            else:
                # 兼容其他长度
                intr = np.array(v).reshape(3, 3)
                
            return intr
        except Exception as e:
            logger.error(f"Error reading intrinsics {file_path}: {e}")
            raise
    def _read_extrinsics(self, scene_path: Path, timestep: int, cam_id: int) -> np.ndarray:
        """
        Read extrinsic txt file.
        Format: 4x4 matrix, providing cam2world (or world2cam? usually cam2world in these datasets).
        Example provided: 4 rows.
        """
        file_path = scene_path / "extrinsics" / f"{timestep:03d}_{cam_id}.txt"
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
            if data.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error reading extrinsics {file_path}: {e}")
            raise

    def _load_image(self, scene_path: Path, timestep: int, cam_id: int) -> torch.Tensor:
        file_path = scene_path / "images" / f"{timestep:03d}_{cam_id}.jpg"
        image = Image.open(file_path).convert('RGB')
        
        # Resize logic
        # if self.cfg.input_image_shape[0] == self.TARGET_SIZE and self.cfg.input_image_shape[1] == self.TARGET_SIZE:
        #     image = image.resize((self.TARGET_SIZE, self.TARGET_SIZE), Image.BILINEAR) # BICUBIC
        if self.cfg.input_image_shape[0] == self.TARGET_HEIGHT and self.cfg.input_image_shape[1] == self.TARGET_WIDTH:
            image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.BILINEAR)
            
        return self.to_tensor(image)

    def _load_mask(self, scene_path: Path, timestep: int, cam_id: int) -> torch.Tensor:
        # file_path = scene_path / "fine_dynamic_masks" / "all" / f"{timestep:03d}_{cam_id}.png"
        file_path = scene_path / "fine_dynamic_masks" / f"{timestep:03d}_{cam_id}.png"
        if not file_path.exists():
            # If mask doesn't exist, return ones (all valid) or zeros depending on usage
            # Assuming 1 is valid/static, 0 is dynamic/masked? Or vice versa.
            # Usually masks: 0 for ignore, 1 for keep. Or dynamic masks: 1 is dynamic object.
            # Returning a default mask of ones (assuming full image is valid static) if missing
            # return torch.ones((1, self.TARGET_SIZE, self.TARGET_SIZE), dtype=torch.float32)
            return torch.ones((0, self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)

        image = Image.open(file_path).convert('L') # Grayscale
        
        # if self.cfg.input_image_shape[0] == self.TARGET_SIZE and self.cfg.input_image_shape[1] == self.TARGET_SIZE:
        #     image = image.resize((self.TARGET_SIZE, self.TARGET_SIZE), Image.NEAREST)
        if self.cfg.input_image_shape[0] == self.TARGET_HEIGHT and self.cfg.input_image_shape[1] == self.TARGET_WIDTH:
            image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.NEAREST)
        ### debug save image
        # debug_save_path = Path("debug_masks") / f"{timestep:03d}_{cam_id}.png"
        # os.makedirs(debug_save_path.parent, exist_ok=True)
        # print(f"*********Debug saving mask to {debug_save_path}")
        # image.save(debug_save_path)
        
        return self.mask_to_tensor(image) # batch, v, h, w

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index_tuple: tuple) -> dict:
        """Get item by index tuple (index, num_context_views, patchsize_h)"""
        index, num_context_views, patchsize_h = index_tuple
        """
        Get item. Note: signature differs slightly from parent if using specialized sampler,
        but standard Dataset expects index only. The wrapper handles batch collation.
        """
        sample_info = self.samples[index]
        scene_id = sample_info['scene_id']
        timesteps = sample_info['timesteps'] # List of ints
        scene_path = self.data_root / scene_id

        # 1. Decide Camera Group (Front vs Back)
        # Randomly select for training, or maybe deterministic based on index for validation?
        # To keep it simple and allow augmentation, we use random for now, or could store in sample_info
        if self.stage == "train":
            use_front_group = random.choice([True, False])
        else:
            # Deterministic for val (e.g., even index front, odd back, or just always front)
            use_front_group = (index % 2 == 0)

        cam_ids = self.CAM_GROUP_FRONT if use_front_group else self.CAM_GROUP_BACK
        
        # cam_ids = self.CAM_GROUP_BACK # 强制使用后视摄像头组进行训练和验证
        
        # We need to collect data for:
        # 3 cameras * numTimes frames
        # Order: Frame 1 (Cam A, B, C), Frame 2 (Cam A, B, C)... 
        
        images = []
        masks = []
        extrinsics = []
        intrinsics = []
        
        # Pre-read intrinsics (constant per cam)
        cam_intrinsics_map = {}
        for cid in cam_ids:
            cam_intrinsics_map[cid] = self._read_intrinsics(scene_path, cid)

        try:
            for ts in timesteps:
                for cid in cam_ids:
                    # Load Image
                    img_tensor = self._load_image(scene_path, ts, cid)
                    images.append(img_tensor)
                    
                    # Load Mask
                    mask_tensor = self._load_mask(scene_path, ts, cid)
                    masks.append(mask_tensor)
                    
                    # Load Extrinsics
                    ext = self._read_extrinsics(scene_path, ts, cid)
                    extrinsics.append(ext)
                    
                    # Intrinsics
                    intr = cam_intrinsics_map[cid].copy()
                    intrinsics.append(intr)

            # Stack everything
            images = torch.stack(images) # (N_views, 3, H, W)
            masks = torch.stack(masks)   # (N_views, 1, H, W)
            extrinsics = torch.from_numpy(np.stack(extrinsics)) # (N_views, 4, 4)
            intrinsics = torch.from_numpy(np.stack(intrinsics)) # (N_views, 3, 3)

            # Determine original image size for normalization
            # The provided intrinsics are based on original image size (likely)
            # Need to know original dims. Assuming 1600x900 from previous code or determining from file
            # Ideally read one image to get size, but for speed assume standard NuScenes
            original_w, original_h = 1600.0, 900.0 
            
            # Normalize Intrinsics and Resize Adjustment
            normalized_intrinsics = intrinsics.clone()
            
            # if self.cfg.input_image_shape[0] == self.TARGET_SIZE and self.cfg.input_image_shape[1] == self.TARGET_SIZE:
            if self.cfg.input_image_shape[0] == self.TARGET_HEIGHT and self.cfg.input_image_shape[1] == self.TARGET_WIDTH:
                # Intrinsics are for 1600x900, we resized to 448x448
                s_x = float(self.TARGET_WIDTH) / original_w
                s_y = float(self.TARGET_HEIGHT) / original_h

                normalized_intrinsics[:, 0, 0] *= s_x # fx
                normalized_intrinsics[:, 1, 1] *= s_y # fy
                normalized_intrinsics[:, 0, 2] *= s_x # cx
                normalized_intrinsics[:, 1, 2] *= s_y # cy
                
                # Update current dimensions for normalization
                # curr_w, curr_h = float(self.TARGET_SIZE), float(self.TARGET_SIZE)
                curr_w, curr_h = float(self.TARGET_WIDTH), float(self.TARGET_HEIGHT)
            else:
                curr_w, curr_h = original_w, original_h

            # Normalize to 0-1 range for view_sampler or model input
            normalized_intrinsics[:, 0, 0] /= curr_w
            normalized_intrinsics[:, 1, 1] /= curr_h
            normalized_intrinsics[:, 0, 2] /= curr_w
            normalized_intrinsics[:, 1, 2] /= curr_h

            # --- View Splitting (Context vs Target) ---
            # Total views = 3 * numTimes
            # We treat all loaded views as potential context/target candidates.
            # The ViewSampler logic usually expects specific indices.
            
            # For simplicity in this loader, we can pass all loaded views to the sampler
            # or split them here. Assuming standard random split from the loaded set.
            
            num_total_views = len(images)
            all_indices = torch.arange(num_total_views)
            
            # Deterministic split: use every view as context and the first three as targets
            context_indices = all_indices
            target_indices = context_indices[:3]
            if len(target_indices) == 0: # Handle numTimes=0 defensive case
                target_indices = context_indices

            # --- Coordinate Normalization ---
            
            # 1. Baseline Scaling (based on context)
            scale = 1.0
            if self.cfg.make_baseline_1 and len(context_indices) > 1:
                ctx_ext = extrinsics[context_indices]
                # Simple heuristic: distance between first and last context camera
                dist = (ctx_ext[0, :3, 3] - ctx_ext[-1, :3, 3]).norm()
                scale = dist
                if scale < 1e-6: scale = 1.0 # Avoid div by zero
                extrinsics[:, :3, 3] /= scale

            # 2. Relative Pose (Center scene around first context camera)
            if self.cfg.relative_pose and len(context_indices) > 0:
                first_ctx_idx = context_indices[0]
                # inv_pose = torch.inverse(extrinsics[first_ctx_idx])
                # Apply inverse of first context cam to all
                # T_new = T_inv * T_old
                # Note: extrinsics are typically c2w. To make first cam identity at origin:
                # New_c2w = First_c2w^-1 * Current_c2w
                # Check your camera_normalization utility for exact math.
                # Assuming simple matrix multiplication here for c2w
                
                # However, many implementations use w2c for normalization logic. 
                # Let's assume standard behavior: transform world coords such that context[0] is at origin.
                # World_new = Context0_w2c * World_old
                # Cam_new_c2w = Context0_w2c * Cam_old_c2w
                c2w_0 = extrinsics[first_ctx_idx]
                w2c_0 = torch.inverse(c2w_0)
                extrinsics = torch.matmul(w2c_0.unsqueeze(0), extrinsics)

            # 3. Rescale to unit cube (fit all positions inside [-1, 1])
            if self.cfg.rescale_to_1cube:
                max_pos = torch.max(torch.abs(extrinsics[:, :3, 3]))
                if max_pos > 0:
                    extrinsics[:, :3, 3] /= max_pos
                    scale *= max_pos # Track total scaling if needed for depth

            # --- Construct Output ---
            
            def build_subset(indices):
                return {
                    "extrinsics": extrinsics[indices],
                    "intrinsics": normalized_intrinsics[indices],
                    "image": images[indices],
                    "fine_dynamic_masks": masks[indices], # Added field
                    "depth": torch.zeros_like(images[indices])[:, 0], # Placeholder depth
                    "near": self.get_bound("near", len(indices)) / scale,
                    "far": self.get_bound("far", len(indices)) / scale,
                    "index": indices,
                }

            scene_id = scene_id + f"_ts{timesteps[0]:03d}_grp{'F' if use_front_group else 'B'}"
            example = {
                "context": build_subset(context_indices),
                "target": build_subset(target_indices),
                "scene": f"nuscenes_{scene_id}",
            }
            
            # --- Augmentation ---
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)

            ### delete crop shims
            
            # 占位符 3D 点和掩码
            context_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool()
            
            target_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool()
            
            example["context"]["valid_mask"] = context_valid_mask * 0 # 返回后续使用时，有batch维度，b v h w
            example["target"]["valid_mask"] = target_valid_mask * 0
            
            return example

        except Exception as e:
            logger.error(f"Error loading sample {scene_id} at {timesteps[0]}: {e}")
            # Fallback strategy: return a random other sample or raise
            raise e

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> torch.Tensor:
        """Get near/far bounds for views"""
        from einops import repeat
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)