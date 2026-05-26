from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
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
import torch.nn.functional as F
from ..model.encoder.vggt.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


@dataclass
class DatasetNuScenesCfg(DatasetCfgCommon):
    """Configuration for nuscenes dataset loader"""
    name: str
    roots: Any
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
    # Path to the text file containing scene IDs (e.g., nuScenes_Train.txt)
    split_file_path: Optional[Path] = None
    numTimes: int = 1  # Number of consecutive timestamps to load
    injuecPose: bool = False  # Whether to GT poses into Omni-VGGT


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
    TARGET_HEIGHT = 252 #252
    TARGET_WIDTH = 448 #448
    
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
    CAM_ID_TO_NAME = {
        0: "CAM_FRONT",
        1: "CAM_FRONT_LEFT",
        2: "CAM_FRONT_RIGHT",
        3: "CAM_BACK_LEFT",
        4: "CAM_BACK_RIGHT",
        5: "CAM_BACK",
    }
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
        self.TARGET_HEIGHT = cfg.input_image_shape[0]
        self.TARGET_WIDTH = cfg.input_image_shape[1]
        self.data_root = self._resolve_data_root(cfg.roots)
        self.split_file_path = self._resolve_split_file_path(cfg.roots, cfg.split_file_path)
        if not self.split_file_path.is_absolute():
            self.split_file_path = self.data_root / self.split_file_path

        # Load scene list from split file.
        if self.stage == "train":
            self.scene_ids = self._load_split_file(self.split_file_path)
        else:
            val_split_file_path = self.split_file_path.with_name(
                self.split_file_path.name.replace("Train", "Val")
            )
            self.scene_ids = self._load_split_file(val_split_file_path)
        
        # Build index of valid samples
        # Each sample is (scene_id, start_timestep_index, camera_group_type)
        self.samples = [] 
        self._build_scene_index()
        
        logger.info(f"nuScenes Dataset: {self.stage}: loaded {len(self.samples)} samples from {len(self.scene_ids)} scenes")
        logger.info(f"Configuration: numTimes={self.cfg.numTimes}")

    def _resolve_data_root(self, roots: Any) -> Path:
        """Resolve the dataset root from legacy or nested NuScenes configs."""
        if isinstance(roots, (str, Path)):
            return Path(roots)

        if isinstance(roots, dict):
            for key in ("data_root", "root", "path"):
                value = roots.get(key)
                if value is not None:
                    return Path(value)

            nested_roots = roots.get("roots")
            if nested_roots is not None:
                return self._resolve_data_root(nested_roots)

        if isinstance(roots, (list, tuple)):
            for item in roots:
                if isinstance(item, (str, Path)):
                    return Path(item)
                if isinstance(item, dict):
                    for key in ("data_root", "root", "path"):
                        value = item.get(key)
                        if value is not None:
                            return Path(value)

        raise ValueError(f"Unable to resolve NuScenes data root from roots={roots!r}")

    def _resolve_split_file_path(self, roots: Any, split_file_path: Optional[Path]) -> Path:
        """Resolve the split file path from either the top level or nested roots config."""
        if split_file_path is not None:
            return Path(split_file_path)

        if isinstance(roots, dict):
            for key in ("split_file_path", "split_file", "split"):
                value = roots.get(key)
                if value is not None:
                    return Path(value)

            nested_roots = roots.get("roots")
            if nested_roots is not None:
                return self._resolve_split_file_path(nested_roots, None)

        if isinstance(roots, (list, tuple)):
            for item in roots:
                if isinstance(item, dict):
                    for key in ("split_file_path", "split_file", "split"):
                        value = item.get(key)
                        if value is not None:
                            return Path(value)

        raise ValueError(f"Unable to resolve NuScenes split file path from roots={roots!r}")

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
                start_indices = [0, 1]
            
            for i in start_indices:
                # We store indices into the 'timesteps' list, not the timestep value itself
                self.samples.append({
                    'scene_id': scene_id,
                    'start_idx': i,
                    # 'timesteps': timesteps[i : i + required_frames]
                    'timesteps': timesteps[i : i + required_frames]
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
        # file_path = scene_path / "extrinsics" / f"{timestep:03d}_{cam_id}.txt"
        file_path = scene_path / "cam2ego_extrinsics" / f"{cam_id}.txt"
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
        # if self.cfg.input_image_shape[0] == self.TARGET_HEIGHT and self.cfg.input_image_shape[1] == self.TARGET_WIDTH:
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.BICUBIC)
            
        return self.to_tensor(image)

    def _load_mask(self, scene_path: Path, timestep: int, cam_id: int) -> torch.Tensor:
        file_path = scene_path / "fine_dynamic_masks" / "all" / f"{timestep:03d}_{cam_id}.png"
        if not file_path.exists():
            return torch.ones((1, self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)

        image = Image.open(file_path).convert('L') # Grayscale
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.NEAREST)
        return self.mask_to_tensor(image) # (1, H, W)


    def _read_car_cam_mask(self, cam_id: int) -> torch.Tensor:
        """Read a camera-specific car mask used to filter invalid pixels."""
        cam_name = self.CAM_ID_TO_NAME[cam_id]
        file_path = Path(__file__).resolve().parents[2] / "config" / "nuscenes_mask" / f"{cam_name}_mask.png"
        if not file_path.exists():
            return torch.ones((1, self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)

        image = Image.open(file_path).convert('L')
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.NEAREST)
        return self.mask_to_tensor(image)
    
    
    def _read_ego_pose(self, scene_path: Path, timestep: int) -> np.ndarray:
        """Read 4x4 ego pose (ego-to-global transform) for a given timestep."""
        file_path = scene_path / "ego_pose" / f"{timestep:03d}.txt"
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
            if data.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error reading ego_pose {file_path}: {e}")
            raise

    def _load_depth_map(self, scene_path: Path, timestep: int, cam_id: int) -> np.ndarray | None:
        file_path = scene_path / "depth_map" / f"{timestep:03d}_{cam_id}.npz"
        if not file_path.exists():
            return None

        data = np.load(file_path, allow_pickle=True)
        if "depth" in data:
            depth = data["depth"]
        else:
            depth = data[data.files[0]]

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth

    def _sparse_lidar_map_downsampler(
        self,
        lidar_depth_map: torch.Tensor,
        downscale_factor: float | tuple[float, float],
    ) -> torch.Tensor:
        """Downsample sparse lidar depth with area pooling and valid-count normalization."""
        raw_avg = F.interpolate(
            lidar_depth_map.unsqueeze(0).unsqueeze(0),
            scale_factor=downscale_factor,
            mode="area",
        ).squeeze(0).squeeze(0)
        raw_mask = F.interpolate(
            (lidar_depth_map > 1e-3).float().unsqueeze(0).unsqueeze(0),
            scale_factor=downscale_factor,
            mode="area",
        ).squeeze(0).squeeze(0)

        downsampled = torch.zeros_like(raw_avg)
        valid = raw_mask > 0
        downsampled[valid] = raw_avg[valid] / raw_mask[valid]
        return downsampled

    def _resize_depth_sparse_area(self, depth: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
        """Resize sparse depth maps with nearest upsampling and area-based downsampling."""
        if depth is None:
            return None

        depth_np = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_np.ndim != 2:
            raise ValueError(f"Expected sparse depth to be 2D after squeeze, got {depth_np.shape}")

        out_h, out_w = int(shape[0]), int(shape[1])
        in_h, in_w = depth_np.shape
        depth_t = torch.from_numpy(depth_np)[None, None]

        if out_h >= in_h and out_w >= in_w:
            resized = F.interpolate(depth_t, size=(out_h, out_w), mode="nearest")
            return resized[0, 0].cpu().numpy().astype(np.float32)

        scale_h = out_h / float(in_h)
        scale_w = out_w / float(in_w)
        resized = self._sparse_lidar_map_downsampler(
            depth_t.squeeze(0).squeeze(0),
            downscale_factor=(scale_h, scale_w),
        )
        return resized.cpu().numpy().astype(np.float32)
    
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
            use_front_group = (sample_info['start_idx'] % 2 == 0)

        # cam_ids = self.CAM_GROUP_FRONT if use_front_group else self.CAM_GROUP_BACK
        
        cam_ids = self.CAM_GROUP_BACK # 强制使用后视摄像头组进行训练和验证
        
        # We need to collect data for:
        # 3 cameras * numTimes frames
        # Order: Frame 1 (Cam A, B, C), Frame 2 (Cam A, B, C)... 
        
        images = []
        extrinsics = []
        intrinsics = []
        depth_maps = []
        depth_valid_masks = []
        car_cam_masks = []
        dynamic_masks = []
        # depthmaps_omnivggt = []
        # masks_omnivggt = []
        # depth_indices = [] # 暂时没用，没有使用到depth map
        # camera_indices = []
        
        # # 先别用 list，直接准备 per-view 的 tensor（稍后知道 V 也行）
        # # 这里在确定 timesteps / cam_ids 后就能确定 V
        # V = len(timesteps) * len(cam_ids)  # = 3 * numTimes

        # # camera_indices: 每个时间戳三路相机固定为 0,1,2（按 cam_ids 的顺序）
        # camera_indices = torch.tensor(
        #     [i for _ in timesteps for i in range(len(cam_ids))],
        #     dtype=torch.int64,
        # )  # shape: (V,)

        # # depth_indices: 你不使用 depth，就给占位 -1
        # depth_indices = torch.full((V,), -1, dtype=torch.int64)  # shape: (V,)

            
        
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
                    # mask_tensor = self._load_mask(scene_path, ts, cid)
                    # masks.append(mask_tensor)
                    
                    # Load sparse depth GT if available
                    depth_np = self._load_depth_map(scene_path, ts, cid)
                    depth_np = self._resize_depth_sparse_area(depth_np, (self.TARGET_HEIGHT, self.TARGET_WIDTH))
                    if depth_np is None:
                        depth_maps.append(torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32))
                        depth_valid_masks.append(torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.bool))
                    else:
                        depth_tensor = torch.from_numpy(depth_np).to(torch.float32)
                        depth_maps.append(depth_tensor)
                        depth_valid_masks.append(depth_tensor > 1e-3)
                                            
                    # Load Extrinsics
                    ext = self._read_extrinsics(scene_path, ts, cid)
                    extrinsics.append(ext)
                    
                    # Intrinsics
                    intr = cam_intrinsics_map[cid].copy()
                    intrinsics.append(intr)

                    # Load camera-specific mask
                    car_cam_masks.append(self._read_car_cam_mask(cid))

                    # Load dynamic mask (1=static, 0=dynamic)
                    dynamic_masks.append(self._load_mask(scene_path, ts, cid))
                    
                    ### add for omni-vggt
                    # depthmap = np.zeros((final_height, new_width), dtype=np.float32)
                    # mask = np.zeros_like(depthmap, dtype=bool)
                    
                    # depthmap_tensor_omnivggt = torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)
                    # mask_tensor_omnivggt = torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.bool)
                    # ### debug print shape
                    # # print(f"Depthmap shape: {depthmap_tensor_omnivggt.shape}, Mask shape: {mask_tensor_omnivggt.shape}")
                    # # Depthmap shape: torch.Size([294, 518]), Mask shape: torch.Size([294, 518])
                    # depthmaps_omnivggt.append(depthmap_tensor_omnivggt)
                    # masks_omnivggt.append(mask_tensor_omnivggt)

            # Stack everything
            images = torch.stack(images) # (9, 3, H, W)
            extrinsics_raw = torch.from_numpy(np.stack(extrinsics)) # (9, 4, 4) cam2ego
            intrinsics = torch.from_numpy(np.stack(intrinsics)) # (9, 3, 3)
            depth_maps = torch.stack(depth_maps)   # (9, H, W)
            depth_valid_masks = torch.stack(depth_valid_masks)   # (9, H, W)
            car_cam_masks = torch.stack(car_cam_masks) # (9, 1, H, W)
            dynamic_masks = torch.stack(dynamic_masks) # (9, 1, H, W)

            n_cams = len(cam_ids)

            # --- Load ego poses (ego2world) and compute cam_T_cam for projection loss ---
            # Original order: [0..2]=ts0, [3..5]=ts1, [6..8]=ts2
            # Context = ts1 (position 1), aux = ts0 (position 0) + ts2 (position 2)
            ego_poses = {}
            for ts in timesteps:
                ego_poses[ts] = torch.from_numpy(self._read_ego_pose(scene_path, ts))

            # cam_T_cam: ref_cam -> src_cam
            # = e2c_src @ inv(ego_pose_src) @ ego_pose_ref @ c2e_ref
            ref_ts = timesteps[1]  # ts1 is the reference
            ego_ref = ego_poses[ref_ts].float()
            c2e_map = {}
            for cid in cam_ids:
                c2e_np = self._read_extrinsics(scene_path, timesteps[0], cid)
                c2e_map[cid] = torch.from_numpy(c2e_np).float()

            cam_T_cam_aux = []
            aux_ts_list = [timesteps[0], timesteps[2]]  # ts0 (forward), ts2 (backward)
            for aux_ts in aux_ts_list:
                for cid in cam_ids:
                    c2e_ref = c2e_map[cid]
                    c2e_src = c2e_map[cid]
                    e2c_src = torch.inverse(c2e_src)
                    cam_T = e2c_src @ torch.inverse(ego_poses[aux_ts].float()) @ ego_ref @ c2e_ref
                    cam_T_cam_aux.append(cam_T)
            cam_T_cam_aux = torch.stack(cam_T_cam_aux)  # (6, 4, 4)

            # Determine original image size for normalization
            original_h, original_w = self.cfg.original_image_shape[0], self.cfg.original_image_shape[1]

            # Normalize Intrinsics and Resize Adjustment
            normalized_intrinsics = intrinsics.clone()
            s_x = float(self.TARGET_WIDTH) / original_w
            s_y = float(self.TARGET_HEIGHT) / original_h

            normalized_intrinsics[:, 0, 0] *= s_x # fx
            normalized_intrinsics[:, 1, 1] *= s_y # fy
            normalized_intrinsics[:, 0, 2] *= s_x # cx
            normalized_intrinsics[:, 1, 2] *= s_y # cy

            # --- View Splitting: context = ts1, aux = ts0 + ts2 ---
            # Original order: [0..2]=ts0, [3..5]=ts1, [6..8]=ts2
            context_indices = torch.arange(n_cams, 2 * n_cams)  # ts1 views (position 1)
            target_indices = context_indices
            aux_indices = torch.cat([torch.arange(0, n_cams), torch.arange(2 * n_cams, 3 * n_cams)])  # ts0, ts2

            # --- Coordinate Normalization ---
            scale = 1.0
            if self.cfg.make_baseline_1 and len(context_indices) > 1:
                ctx_ext = extrinsics_raw[context_indices]
                dist = (ctx_ext[0, :3, 3] - ctx_ext[-1, :3, 3]).norm()
                scale = dist
                if scale < 1e-6: scale = 1.0
                extrinsics_raw[:, :3, 3] /= scale

            if self.cfg.rescale_to_1cube:
                max_pos = torch.max(torch.abs(extrinsics_raw[:, :3, 3]))
                if max_pos > 0:
                    extrinsics_raw[:, :3, 3] /= max_pos
                    scale *= max_pos

            # --- Construct Output ---
            extrinsics = closed_form_inverse_se3(extrinsics_raw)[:, :3, :]

            def build_subset(indices):
                return {
                    "extrinsics": extrinsics[indices],
                    "intrinsics": normalized_intrinsics[indices],
                    "image": images[indices],
                    "depth": depth_maps[indices],
                    "depth_valid_mask": depth_valid_masks[indices],
                    "car_cam_mask": car_cam_masks[indices],
                    "dynamic_mask": dynamic_masks[indices],  # (V, 1, H, W), white(1)=dynamic, black(0)=static
                    "near": self.get_bound("near", len(indices)) / scale,
                    "far": self.get_bound("far", len(indices)) / scale,
                    "index": indices,
                }

            scene_id = scene_id + f"_ts{timesteps[1]:03d}_grp{'F' if use_front_group else 'B'}"
            example = {
                "context": build_subset(context_indices),
                "target": build_subset(target_indices),
                "scene": f"nuscenes_{scene_id}",
                # Auxiliary data for projection loss (ts0 + ts2)
                "aux": {
                    "image": images[aux_indices],
                    "intrinsics": normalized_intrinsics[aux_indices],
                    "car_cam_mask": car_cam_masks[aux_indices],
                    "dynamic_mask": dynamic_masks[aux_indices],  # (6, 1, H, W), white(1)=dynamic, black(0)=static
                    "cam_T_cam": cam_T_cam_aux,  # (6, 4, 4)
                },
            }

            # --- Augmentation ---
            # augment
            # if self.stage == "train" and self.cfg.augment:
            #     example = apply_augmentation_shim(example)

            # Placeholder valid masks
            context_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool()
            target_valid_mask = torch.ones_like(example["target"]["image"])[:, 0].bool()
            example["context"]["valid_mask"] = context_valid_mask * 0
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