### 统计拼接成功率，并且仅统计拼接成功的样本的各项指标
import os
import sys
import torch
import numpy as np
import re
import math
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=0 python batchCalcuMetricsSuccessNuScenes260202.py

# ================= 项目路径设置 =================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.image import process_image_nuscenes

# ================= 配置区域 =================
# [核心修改] 批处理大小
BATCH_SIZE = 64  # 建议设大一点 (16-64)

RENDER_H = 294
TARGET_W = 518     # 裁剪目标宽度
WIDE_W = 1036       # 渲染宽度

# 拼接失败判定阈值
WHITE_THRESHOLD = 0.9  # 像素值大于此值(归一化到0-1)认为是白色
FAILURE_RATIO = 0.5    # 某一侧白色像素占比超过50%则判定失败

# 功能开关
EVAL_FUSION = False        # True: 计算Fusion指标; False: 仅计算Original指标
SELECTED_CAMERAS = [0, 5] # 仅计算这些相机的指标 (空列表 [] 代表计算所有相机 0-5)

# 路径配置
DATASET_ROOT = "datasets/nuscenes/processed_10Hz/trainval2"
VAL_LIST_PATH = "nuScenes_Val.txt"
SAVE_ROOT = "renders_val_omni/260203SingleFramesVol0002Epoch5Iter30000/render_only"
RESULT_TXT_PATH = os.path.join(SAVE_ROOT, f"260205stitching_metrics_report_cams_{'_'.join(map(str, SELECTED_CAMERAS)) if SELECTED_CAMERAS else 'all'}.txt")

# ================= 统计类 =================

class MetricTracker:
    """统计图像质量指标（仅针对拼接成功的图片）"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr = []
        self.ssim = []
        self.lpips = []
    
    def update(self, psnr, ssim, lpips):
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.lpips.append(lpips)
    
    def get_avg(self):
        return {
            'psnr': np.mean(self.psnr) if self.psnr else 0.0,
            'ssim': np.mean(self.ssim) if self.ssim else 0.0,
            'lpips': np.mean(self.lpips) if self.lpips else 0.0,
            'count': len(self.psnr)
        }

class FailureTracker:
    """统计拼接失败的情况"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_count = 0
        self.failure_count = 0
        self.left_failures = 0
        self.right_failures = 0
        self.both_failures = 0
    
    def update(self, is_failure, left_fail, right_fail):
        self.total_count += 1
        if is_failure:
            self.failure_count += 1
        if left_fail:
            self.left_failures += 1
        if right_fail:
            self.right_failures += 1
        if left_fail and right_fail:
            self.both_failures += 1
    
    def get_stats(self):
        if self.total_count == 0:
            return {
                'total': 0,
                'failures': 0,
                'success': 0,
                'failure_rate': 0.0,
                'success_rate': 0.0,
                'left_fail_rate': 0.0,
                'right_fail_rate': 0.0,
                'both_fail_rate': 0.0
            }
        
        success_count = self.total_count - self.failure_count
        return {
            'total': self.total_count,
            'failures': self.failure_count,
            'success': success_count,
            'failure_rate': self.failure_count / self.total_count * 100,
            'success_rate': success_count / self.total_count * 100,
            'left_fail_rate': self.left_failures / self.total_count * 100,
            'right_fail_rate': self.right_failures / self.total_count * 100,
            'both_fail_rate': self.both_failures / self.total_count * 100
        }

class CombinedTracker:
    """组合统计器：同时追踪拼接成功率和图像质量指标"""
    def __init__(self):
        self.failure_tracker = FailureTracker()
        self.metric_tracker = MetricTracker()
    
    def update_failure(self, is_failure, left_fail, right_fail):
        self.failure_tracker.update(is_failure, left_fail, right_fail)
    
    def update_metrics(self, psnr, ssim, lpips):
        self.metric_tracker.update(psnr, ssim, lpips)
    
    def get_summary(self):
        failure_stats = self.failure_tracker.get_stats()
        metric_avg = self.metric_tracker.get_avg()
        return {**failure_stats, **metric_avg}

def format_metrics(m):
    """格式化指标输出"""
    avg = m.get_avg()
    return f"P={avg['psnr']:.2f} | S={avg['ssim']:.4f} | L={avg['lpips']:.4f} | N={avg['count']}"

def format_combined(tracker):
    """格式化组合统计输出"""
    summary = tracker.get_summary()
    return (f"Total={summary['total']:4d} | Success={summary['success']:4d}({summary['success_rate']:5.2f}%) | "
            f"Fail={summary['failures']:4d}({summary['failure_rate']:5.2f}%) | "
            f"P={summary['psnr']:.2f} | S={summary['ssim']:.4f} | L={summary['lpips']:.4f}")

# ================= 工具函数 =================

def load_scene_list(txt_path):
    if not os.path.exists(txt_path): 
        return []
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    return scenes

def get_scene_frames(scene_path):
    img_dir = os.path.join(scene_path, "images")
    if not os.path.exists(img_dir): 
        return []
    frame_ids = []
    pattern = re.compile(r"^(.*)_0\.(jpg|png|jpeg)$", re.IGNORECASE)
    for f in sorted(os.listdir(img_dir)):
        match = pattern.match(f)
        if match: 
            frame_ids.append(match.group(1))
    return sorted(frame_ids)

def get_image_path(scene_dir, frame_id, cam_idx):
    img_dir = os.path.join(scene_dir, "images")
    for ext in ['.jpg', '.png', '.jpeg']:
        p = os.path.join(img_dir, f"{frame_id}_{cam_idx}{ext}")
        if os.path.exists(p):
            return p
    return None

def load_tensor_from_path(path):
    """加载图像并转为tensor [C, H, W]，范围 [0, 1]"""
    if not os.path.exists(path): 
        return None
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img)

def should_process_cam(cam_idx):
    if not SELECTED_CAMERAS: 
        return True
    return cam_idx in SELECTED_CAMERAS

def check_white_region_batch(img_batch, left_region, right_region):
    """
    批量检查图像的左右拼接区域白色像素比例
    Args:
        img_batch: [B, C, H, W] tensor, 范围 [0, 1]
        left_region: tuple (start_x, end_x) 左侧区域
        right_region: tuple (start_x, end_x) 右侧区域
    Returns:
        failures: [B] boolean tensor, 是否失败
        left_fails: [B] boolean tensor, 左侧是否失败
        right_fails: [B] boolean tensor, 右侧是否失败
    """
    # 提取左右区域
    left_crop = img_batch[:, :, :, left_region[0]:left_region[1]]
    right_crop = img_batch[:, :, :, right_region[0]:right_region[1]]
    
    # 检查是否为白色：所有通道都大于阈值
    left_is_white = (left_crop > WHITE_THRESHOLD).all(dim=1)
    right_is_white = (right_crop > WHITE_THRESHOLD).all(dim=1)
    
    # 计算白色像素比例
    left_white_ratio = left_is_white.float().mean(dim=(1, 2))
    right_white_ratio = right_is_white.float().mean(dim=(1, 2))
    
    # 判断是否失败
    left_fails = left_white_ratio > FAILURE_RATIO
    right_fails = right_white_ratio > FAILURE_RATIO
    failures = left_fails | right_fails
    
    return failures, left_fails, right_fails

# ================= 主流程 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing on {device}...")
    print(f"White threshold: {WHITE_THRESHOLD}, Failure ratio: {FAILURE_RATIO}")
    
    # 初始化LPIPS
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none').to(device)
    lpips_fn.eval()
    
    scenes = load_scene_list(VAL_LIST_PATH)
    if not os.path.exists(SAVE_ROOT):
        print(f"Error: Render dir {SAVE_ROOT} does not exist.")
        return

    # 计算拼接区域坐标
    start_x = (WIDE_W - TARGET_W) // 2  # 224
    left_region = (0, start_x)
    right_region = (start_x + TARGET_W, WIDE_W)
    
    print(f"Center region: [{start_x}, {start_x + TARGET_W})")
    print(f"Left stitching region: {left_region}")
    print(f"Right stitching region: {right_region}")

    # === 全局统计器 ===
    def create_stats_dict():
        d = {
            'total': {'orig': CombinedTracker()},
            'front_grp': {'orig': CombinedTracker()},
            'back_grp': {'orig': CombinedTracker()},
            'cam0_5_avg': {'orig': CombinedTracker()}
        }
        if EVAL_FUSION:
            d['total']['fusion'] = CombinedTracker()
            d['front_grp']['fusion'] = CombinedTracker()
            d['back_grp']['fusion'] = CombinedTracker()
            d['cam0_5_avg']['fusion'] = CombinedTracker()
        
        for i in range(6):
            d[i] = {'orig': CombinedTracker()}
            if EVAL_FUSION:
                d[i]['fusion'] = CombinedTracker()
        return d

    global_stats = create_stats_dict()
    scene_stats_map = {}
    
    # 1. 扫描并收集所有任务
    all_tasks = []
    print("Scanning scenes to collect evaluation tasks...")
    
    for scene_id in tqdm(scenes, desc="Scanning"):
        scene_dir = os.path.join(DATASET_ROOT, scene_id)
        render_scene_dir = os.path.join(SAVE_ROOT, scene_id)
        
        # 初始化该场景的统计器
        scene_stats_map[scene_id] = create_stats_dict()
        
        frames = get_scene_frames(scene_dir)
        
        for frame_id in frames:
            for cam_idx in range(6):
                if not should_process_cam(cam_idx):
                    continue
                
                grp_key = 'front_grp' if cam_idx in [0, 1, 2] else 'back_grp'
                is_fusion_cam = (cam_idx == 0 or cam_idx == 5)
                
                gt_path = get_image_path(scene_dir, frame_id, cam_idx)
                if gt_path is None:
                    continue
                
                wide_path = os.path.join(render_scene_dir, f"{frame_id}_{cam_idx}_wide.jpg")
                if not os.path.exists(wide_path):
                    continue

                task = {
                    'scene_id': scene_id,
                    'frame_id': frame_id,
                    'cam_idx': cam_idx,
                    'grp_key': grp_key,
                    'gt_path': gt_path,
                    'wide_path': wide_path,
                    'render_dir': render_scene_dir,
                    'is_fusion_cam': is_fusion_cam
                }
                all_tasks.append(task)

    # 2. Batch 处理
    total_tasks = len(all_tasks)
    num_batches = math.ceil(total_tasks / BATCH_SIZE)
    print(f"Total tasks: {total_tasks}. Batch size: {BATCH_SIZE}. Batches: {num_batches}")

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_tasks = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        # ===== 阶段1: 加载wide图并检查拼接 =====
        wide_list = []
        valid_indices = []
        
        for idx, task in enumerate(batch_tasks):
            wide_tensor = load_tensor_from_path(task['wide_path'])
            if wide_tensor is None or wide_tensor.shape[2] != WIDE_W:
                continue
            wide_list.append(wide_tensor)
            valid_indices.append(idx)

        if not valid_indices:
            continue

        # 转GPU
        wide_batch = torch.stack(wide_list).to(device)
        
        # 检查拼接失败
        with torch.no_grad():
            failures, left_fails, right_fails = check_white_region_batch(
                wide_batch, left_region, right_region
            )
        
        # ===== 阶段2: 筛选成功的图片，准备计算指标 =====
        success_mask = ~failures  # 成功的图片mask
        success_indices = [idx for idx, is_success in enumerate(success_mask) if is_success]
        
        # 先更新所有的失败统计
        for b in range(len(valid_indices)):
            task_idx = valid_indices[b]
            task = batch_tasks[task_idx]
            
            is_failure = failures[b].item()
            left_fail = left_fails[b].item()
            right_fail = right_fails[b].item()
            
            # 更新失败统计
            def update_failure_stats(stats_dict, cam_idx, grp_key):
                stats_dict['total']['orig'].update_failure(is_failure, left_fail, right_fail)
                stats_dict[grp_key]['orig'].update_failure(is_failure, left_fail, right_fail)
                stats_dict[cam_idx]['orig'].update_failure(is_failure, left_fail, right_fail)
                if cam_idx in [0, 5]:
                    stats_dict['cam0_5_avg']['orig'].update_failure(is_failure, left_fail, right_fail)
                
                if EVAL_FUSION:
                    stats_dict['total']['fusion'].update_failure(is_failure, left_fail, right_fail)
                    stats_dict[grp_key]['fusion'].update_failure(is_failure, left_fail, right_fail)
                    stats_dict[cam_idx]['fusion'].update_failure(is_failure, left_fail, right_fail)
                    if cam_idx in [0, 5]:
                        stats_dict['cam0_5_avg']['fusion'].update_failure(is_failure, left_fail, right_fail)
            
            update_failure_stats(global_stats, task['cam_idx'], task['grp_key'])
            update_failure_stats(scene_stats_map[task['scene_id']], task['cam_idx'], task['grp_key'])
        
        # 如果没有成功的图片，跳过指标计算
        if len(success_indices) == 0:
            continue
        
        # ===== 阶段3: 只对成功的图片计算指标 =====
        gt_list = []
        pred_orig_list = []
        pred_fusion_list = []
        success_tasks = []
        
        for b in success_indices:
            task_idx = valid_indices[b]
            task = batch_tasks[task_idx]
            
            # 加载GT
            gt_tensor_raw = process_image_nuscenes(task['gt_path'])
            gt_img = (gt_tensor_raw + 1) * 0.5
            
            # 裁剪中心区域（从已加载的wide_batch中获取）
            pred_crop = wide_batch[b:b+1, :, :, start_x : start_x + TARGET_W]
            
            gt_list.append(gt_img)
            pred_orig_list.append(pred_crop.squeeze(0))
            
            # Fusion图
            if EVAL_FUSION:
                fusion_path = os.path.join(task['render_dir'], 
                                          f"{task['frame_id']}_{task['cam_idx']}_wide_fusion.jpg")
                if task['is_fusion_cam'] and os.path.exists(fusion_path):
                    render_wide_f = load_tensor_from_path(fusion_path)
                    if render_wide_f is not None:
                        pred_crop_f = render_wide_f[:, :, start_x : start_x + TARGET_W]
                    else:
                        pred_crop_f = pred_crop.squeeze(0)
                else:
                    pred_crop_f = pred_crop.squeeze(0)
                pred_fusion_list.append(pred_crop_f)
            
            success_tasks.append(task)
        
        if not gt_list:
            continue
        
        # 转GPU Batch
        gt_batch = torch.stack(gt_list).to(device)
        pred_orig_batch = torch.stack(pred_orig_list).to(device)
        if EVAL_FUSION:
            pred_fusion_batch = torch.stack(pred_fusion_list).to(device)
        
        # 批量计算指标
        with torch.no_grad():
            # Original
            # [Fix] 添加 .view(-1) 确保即使只有一张图也是 1D tensor，避免 0-dim 索引报错
            lpips_orig = lpips_fn(pred_orig_batch, gt_batch).view(-1)
            
            # Fusion
            if EVAL_FUSION:
                lpips_fusion = lpips_fn(pred_fusion_batch, gt_batch).view(-1)
        
        # 更新指标统计
        for b in range(len(success_tasks)):
            task = success_tasks[b]
            cam_idx = task['cam_idx']
            grp_key = task['grp_key']
            
            # Original metrics
            p_orig = peak_signal_noise_ratio(pred_orig_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
            s_orig = structural_similarity_index_measure(pred_orig_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
            l_orig = lpips_orig[b].item()
            
            def update_metrics(stats_dict, cam, grp, p, s, l, mode='orig'):
                stats_dict['total'][mode].update_metrics(p, s, l)
                stats_dict[grp][mode].update_metrics(p, s, l)
                stats_dict[cam][mode].update_metrics(p, s, l)
                if cam in [0, 5]:
                    stats_dict['cam0_5_avg'][mode].update_metrics(p, s, l)
            
            update_metrics(global_stats, cam_idx, grp_key, p_orig, s_orig, l_orig, 'orig')
            update_metrics(scene_stats_map[task['scene_id']], cam_idx, grp_key, p_orig, s_orig, l_orig, 'orig')
            
            # Fusion metrics
            if EVAL_FUSION:
                p_fusion = peak_signal_noise_ratio(pred_fusion_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                s_fusion = structural_similarity_index_measure(pred_fusion_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                l_fusion = lpips_fusion[b].item()
                
                update_metrics(global_stats, cam_idx, grp_key, p_fusion, s_fusion, l_fusion, 'fusion')
                update_metrics(scene_stats_map[task['scene_id']], cam_idx, grp_key, p_fusion, s_fusion, l_fusion, 'fusion')

    # ================= 写入报告 =================
    print("Generating report...")
    
    with open(RESULT_TXT_PATH, 'w') as f:
        f.write("Combined Stitching & Metrics Evaluation Report\n")
        f.write(f"Source: {SAVE_ROOT}\n")
        f.write(f"Fusion Evaluated: {EVAL_FUSION}\n")
        f.write(f"Selected Cameras: {SELECTED_CAMERAS if SELECTED_CAMERAS else 'All'}\n")
        f.write(f"White Threshold: {WHITE_THRESHOLD}, Failure Ratio: {FAILURE_RATIO}\n")
        f.write(f"Left Region: {left_region}, Right Region: {right_region}\n")
        f.write("="*120 + "\n")
        f.write("Note: Metrics (P/S/L) are calculated ONLY for successfully stitched images\n")
        f.write("="*120 + "\n\n")
        
        def write_block(title, stats_dict, mode='orig'):
            f.write(f">>> [{title}] <<<\n")
            f.write(f"Total Average                : {format_combined(stats_dict['total'][mode])}\n")
            f.write(f"Cam 0 & 5 Average            : {format_combined(stats_dict['cam0_5_avg'][mode])}\n")
            f.write(f"Group Front Avg (0,1,2)      : {format_combined(stats_dict['front_grp'][mode])}\n")
            f.write(f"Group Back Avg  (5,4,3)      : {format_combined(stats_dict['back_grp'][mode])}\n")
            f.write("-" * 100 + "\n")
            f.write("Individual Cameras:\n")
            for c in range(6):
                if should_process_cam(c):
                    f.write(f"  Cam {c}: {format_combined(stats_dict[c][mode])}\n")
            f.write("\n")

        write_block("Original Renders", global_stats, 'orig')
        if EVAL_FUSION:
            write_block("Fusion Renders", global_stats, 'fusion')

        f.write("="*120 + "\n")
        f.write("Per Scene Breakdown:\n\n")
        
        for scene_id in scenes:
            if scene_id not in scene_stats_map:
                continue
            stats = scene_stats_map[scene_id]
            
            # 检查是否有数据
            if stats['total']['orig'].failure_tracker.total_count == 0:
                continue
            
            f.write(f"Scene {scene_id}:\n")
            f.write(f"  [Original]\n")
            f.write(f"    Total       : {format_combined(stats['total']['orig'])}\n")
            f.write(f"    Cam 0+5     : {format_combined(stats['cam0_5_avg']['orig'])}\n")
            f.write(f"    Front Group : {format_combined(stats['front_grp']['orig'])}\n")
            for c in [0, 1, 2]:
                if should_process_cam(c):
                    f.write(f"      Cam {c}: {format_combined(stats[c]['orig'])}\n")
            f.write(f"    Back Group  : {format_combined(stats['back_grp']['orig'])}\n")
            for c in [5, 4, 3]:
                if should_process_cam(c):
                    f.write(f"      Cam {c}: {format_combined(stats[c]['orig'])}\n")
            
            if EVAL_FUSION:
                f.write(f"  [Fusion]\n")
                f.write(f"    Total       : {format_combined(stats['total']['fusion'])}\n")
                f.write(f"    Cam 0+5     : {format_combined(stats['cam0_5_avg']['fusion'])}\n")
                f.write(f"    Front Group : {format_combined(stats['front_grp']['fusion'])}\n")
                for c in [0, 1, 2]:
                    if should_process_cam(c):
                        f.write(f"      Cam {c}: {format_combined(stats[c]['fusion'])}\n")
                f.write(f"    Back Group  : {format_combined(stats['back_grp']['fusion'])}\n")
                for c in [5, 4, 3]:
                    if should_process_cam(c):
                        f.write(f"      Cam {c}: {format_combined(stats[c]['fusion'])}\n")
            
            f.write("\n")
            
    print(f"\nEvaluation finished. Results saved to {RESULT_TXT_PATH}")
    print("\nGlobal Summary (Original):")
    summary = global_stats['total']['orig'].get_summary()
    print(f"  Total: {summary['total']} | Success: {summary['success']}({summary['success_rate']:.2f}%) | "
          f"Fail: {summary['failures']}({summary['failure_rate']:.2f}%)")
    print(f"  Metrics (Success only): P={summary['psnr']:.2f} | S={summary['ssim']:.4f} | L={summary['lpips']:.4f}")

if __name__ == "__main__":
    main()