from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List, Any
import os
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
import logging

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
import torch.nn.functional as F
from ..model.encoder.vggt.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


@dataclass
class DatasetLyft1224Cfg(DatasetCfgCommon):
    """Configuration for Lyft 1224 dataset loader"""
    name: str
    roots_train: Any
    roots_val: Any
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
    split_file_path: Optional[str] = None
    numTimes: int = 1


@dataclass
class DatasetLyft1224CfgWrapper:
    lyft1224: DatasetLyft1224Cfg


class DatasetLyft1224(Dataset):
    """
    Lyft dataset loader.
    Structure:
        root/{scene_id}/
            images/{timestep:03d}_{cam_id}.jpg
            cam2ego_extrinsics/{cam_id}.txt
            intrinsics/{cam_id}.txt
            fine_dynamic_masks/all/{timestep:03d}_{cam_id}.png
        ego_car_masks/{cam_id}.jpg   (shared across scenes)
    """

    cfg: DatasetLyft1224Cfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor

    near: float = 0.1
    far: float = 100.0

    TARGET_HEIGHT = 252
    TARGET_WIDTH = 448

    CAM_GROUP_BACK = [5, 4, 3]

    def __init__(
        self,
        cfg: DatasetLyft1224Cfg,
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

        # Resolve data root based on stage
        roots = cfg.roots_train if stage == "train" else cfg.roots_val
        self.data_root = self._resolve_data_root(roots)

        # Resolve split file
        split_name = cfg.split_file_path
        if split_name is None:
            split_name = "train1224.txt" if stage == "train" else "lyft_val1224.txt"
        self.split_file_path = self.data_root / split_name

        self.scene_ids = self._load_split_file(self.split_file_path)

        # ego_car_masks directory (shared across scenes in the dataset root)
        self.ego_car_masks_dir = self.data_root / "ego_car_masks"

        self.samples = []
        self._build_scene_index()

        logger.info(f"Lyft1224 Dataset: {self.stage}: loaded {len(self.samples)} samples from {len(self.scene_ids)} scenes")
        logger.info(f"Configuration: numTimes={self.cfg.numTimes}")

    def _resolve_data_root(self, roots: Any) -> Path:
        """Resolve the dataset root from config."""
        if isinstance(roots, (str, Path)):
            return Path(roots)
        if isinstance(roots, dict):
            for key in ("data_root", "root", "path"):
                value = roots.get(key)
                if value is not None:
                    return Path(value)
            nested = roots.get("roots")
            if nested is not None:
                return self._resolve_data_root(nested)
        if isinstance(roots, (list, tuple)):
            for item in roots:
                if isinstance(item, (str, Path)):
                    return Path(item)
                if isinstance(item, dict):
                    for key in ("data_root", "root", "path"):
                        value = item.get(key)
                        if value is not None:
                            return Path(value)
        raise ValueError(f"Unable to resolve Lyft data root from roots={roots!r}")

    def _load_split_file(self, split_path: Path) -> List[str]:
        """Load scene IDs from text file."""
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with open(split_path, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        return scene_ids

    def _get_available_timesteps(self, scene_path: Path) -> List[int]:
        """Find all unique timesteps in a scene based on image files."""
        image_dir = scene_path / "images"
        if not image_dir.exists():
            return []
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
        """Build the list of valid samples."""
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

            intrinsics_path = scene_path / "intrinsics"
            if not intrinsics_path.exists():
                logger.warning(f"Intrinsics missing for {scene_id}")
                continue

            max_start_idx = num_timesteps - required_frames
            if self.stage == "train":
                start_indices = range(0, max_start_idx + 1)
            else:
                start_indices = [0, 10]

            for i in start_indices:
                self.samples.append({
                    'scene_id': scene_id,
                    'start_idx': i,
                    'timesteps': timesteps[i: i + required_frames]
                })

            valid_scenes += 1

        if valid_scenes == 0:
            logger.error("No valid scenes found! Check paths and split file.")

    def _read_intrinsics(self, scene_path: Path, cam_id: int) -> np.ndarray:
        """Read intrinsic txt file."""
        file_path = scene_path / "intrinsics" / f"{cam_id}.txt"
        try:
            with open(file_path, 'r') as f:
                v = [float(line.strip()) for line in f if line.strip()]
            intr = np.eye(3, dtype=np.float32)
            if len(v) == 9:
                intr[0, 0] = v[0]
                intr[1, 1] = v[1]
                intr[0, 2] = v[2]
                intr[1, 2] = v[3]
            else:
                intr = np.array(v).reshape(3, 3)
            return intr
        except Exception as e:
            logger.error(f"Error reading intrinsics {file_path}: {e}")
            raise

    def _read_extrinsics(self, scene_path: Path, timestep: int, cam_id: int) -> np.ndarray:
        """Read extrinsic txt file (cam2ego)."""
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
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.BICUBIC)
        return self.to_tensor(image)

    def _load_mask(self, scene_path: Path, timestep: int, cam_id: int) -> torch.Tensor:
        file_path = scene_path / "fine_dynamic_masks" / "all" / f"{timestep:03d}_{cam_id}.png"
        if not file_path.exists():
            return torch.ones((0, self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)
        image = Image.open(file_path).convert('L')
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.NEAREST)
        return self.mask_to_tensor(image)

    def _read_car_cam_mask(self, cam_id: int) -> torch.Tensor:
        """Read camera-specific car mask from the shared ego_car_masks directory."""
        file_path = self.ego_car_masks_dir / f"{cam_id}.jpg"
        if not file_path.exists():
            return torch.ones((1, self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32)
        image = Image.open(file_path).convert('L')
        image = image.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.NEAREST)
        return self.mask_to_tensor(image)

    def _read_ego_pose(self, scene_path: Path, timestep: int) -> np.ndarray:
        """Read 4x4 ego pose (ego-to-global transform)."""
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
        # Filter out depth values beyond 110m
        depth[depth > 110.0] = 0.0
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
        sample_info = self.samples[index]
        scene_id = sample_info['scene_id']
        timesteps = sample_info['timesteps']
        scene_path = self.data_root / scene_id

        cam_ids = self.CAM_GROUP_BACK

        images = []
        extrinsics = []
        intrinsics = []
        depth_maps = []
        depth_valid_masks = []
        car_cam_masks = []
        dynamic_masks = []

        # Pre-read intrinsics (constant per cam)
        cam_intrinsics_map = {}
        for cid in cam_ids:
            cam_intrinsics_map[cid] = self._read_intrinsics(scene_path, cid)

        try:
            for ts in timesteps:
                for cid in cam_ids:
                    img_tensor = self._load_image(scene_path, ts, cid)
                    images.append(img_tensor)

                    depth_np = self._load_depth_map(scene_path, ts, cid)
                    depth_np = self._resize_depth_sparse_area(depth_np, (self.TARGET_HEIGHT, self.TARGET_WIDTH))
                    if depth_np is None:
                        depth_maps.append(torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.float32))
                        depth_valid_masks.append(torch.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH), dtype=torch.bool))
                    else:
                        depth_tensor = torch.from_numpy(depth_np).to(torch.float32)
                        depth_maps.append(depth_tensor)
                        depth_valid_masks.append(depth_tensor > 1e-3)

                    ext = self._read_extrinsics(scene_path, ts, cid)
                    extrinsics.append(ext)

                    intr = cam_intrinsics_map[cid].copy()
                    intrinsics.append(intr)

                    car_cam_masks.append(self._read_car_cam_mask(cid))
                    dynamic_masks.append(self._load_mask(scene_path, ts, cid))

            # Stack everything
            images = torch.stack(images)
            extrinsics_raw = torch.from_numpy(np.stack(extrinsics))
            intrinsics = torch.from_numpy(np.stack(intrinsics))
            depth_maps = torch.stack(depth_maps)
            depth_valid_masks = torch.stack(depth_valid_masks)
            car_cam_masks = torch.stack(car_cam_masks)
            dynamic_masks = torch.stack(dynamic_masks)

            n_cams = len(cam_ids)

            # Load ego poses and compute cam_T_cam for projection loss
            ego_poses = {}
            for ts in timesteps:
                ego_poses[ts] = torch.from_numpy(self._read_ego_pose(scene_path, ts))

            ref_ts = timesteps[1]
            ego_ref = ego_poses[ref_ts].float()
            c2e_map = {}
            for cid in cam_ids:
                c2e_np = self._read_extrinsics(scene_path, timesteps[0], cid)
                c2e_map[cid] = torch.from_numpy(c2e_np).float()

            cam_T_cam_aux = []
            aux_ts_list = [timesteps[0], timesteps[2]]
            for aux_ts in aux_ts_list:
                for cid in cam_ids:
                    c2e_ref = c2e_map[cid]
                    c2e_src = c2e_map[cid]
                    e2c_src = torch.inverse(c2e_src)
                    cam_T = e2c_src @ torch.inverse(ego_poses[aux_ts].float()) @ ego_ref @ c2e_ref
                    cam_T_cam_aux.append(cam_T)
            cam_T_cam_aux = torch.stack(cam_T_cam_aux)

            # Normalize intrinsics
            original_h, original_w = self.cfg.original_image_shape[0], self.cfg.original_image_shape[1]
            normalized_intrinsics = intrinsics.clone()
            s_x = float(self.TARGET_WIDTH) / original_w
            s_y = float(self.TARGET_HEIGHT) / original_h
            normalized_intrinsics[:, 0, 0] *= s_x
            normalized_intrinsics[:, 1, 1] *= s_y
            normalized_intrinsics[:, 0, 2] *= s_x
            normalized_intrinsics[:, 1, 2] *= s_y

            # View splitting: context = ts1, aux = ts0 + ts2
            context_indices = torch.arange(n_cams, 2 * n_cams)
            target_indices = context_indices
            aux_indices = torch.cat([torch.arange(0, n_cams), torch.arange(2 * n_cams, 3 * n_cams)])

            # Coordinate normalization
            scale = 1.0
            if self.cfg.make_baseline_1 and len(context_indices) > 1:
                ctx_ext = extrinsics_raw[context_indices]
                dist = (ctx_ext[0, :3, 3] - ctx_ext[-1, :3, 3]).norm()
                scale = dist
                if scale < 1e-6:
                    scale = 1.0
                extrinsics_raw[:, :3, 3] /= scale

            if self.cfg.rescale_to_1cube:
                max_pos = torch.max(torch.abs(extrinsics_raw[:, :3, 3]))
                if max_pos > 0:
                    extrinsics_raw[:, :3, 3] /= max_pos
                    scale *= max_pos

            extrinsics = closed_form_inverse_se3(extrinsics_raw)[:, :3, :]

            def build_subset(indices):
                return {
                    "extrinsics": extrinsics[indices],
                    "intrinsics": normalized_intrinsics[indices],
                    "image": images[indices],
                    "depth": depth_maps[indices],
                    "depth_valid_mask": depth_valid_masks[indices],
                    "car_cam_mask": car_cam_masks[indices],
                    "dynamic_mask": dynamic_masks[indices],
                    "near": self.get_bound("near", len(indices)) / scale,
                    "far": self.get_bound("far", len(indices)) / scale,
                    "index": indices,
                }

            scene_id = scene_id + f"_ts{timesteps[1]:03d}_grpB"
            example = {
                "context": build_subset(context_indices),
                "target": build_subset(target_indices),
                "scene": f"lyft1224_{scene_id}",
                "aux": {
                    "image": images[aux_indices],
                    "intrinsics": normalized_intrinsics[aux_indices],
                    "car_cam_mask": car_cam_masks[aux_indices],
                    "dynamic_mask": dynamic_masks[aux_indices],
                    "cam_T_cam": cam_T_cam_aux,
                },
            }

            # Placeholder valid masks
            context_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool()
            target_valid_mask = torch.ones_like(example["target"]["image"])[:, 0].bool()
            example["context"]["valid_mask"] = context_valid_mask * 0
            example["target"]["valid_mask"] = target_valid_mask * 0

            return example

        except Exception as e:
            logger.error(f"Error loading sample {scene_id} at {timesteps[0]}: {e}")
            raise e

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> torch.Tensor:
        """Get near/far bounds for views"""
        from einops import repeat
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
