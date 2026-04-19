import os
import sys
import torch
import re
import math
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as tf
from tqdm import tqdm

# ================= Configuration =================
# Set GPU ID if needed, e.g., os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Model and Geometry Utils
# Assuming these paths exist based on your provided file context
from src.model.model.anysplat import AnySplat 
from src.model.encoder.vggt.utils.geometry import closed_form_inverse_se3
# CUDA_VISIBLE_DEVICES=4 python bf16_BatchInferDWSplatNuScenes260408.py
# --- Batch Settings ---
BATCH_SIZE = 8

# --- Image Dimensions (Must match training/loader logic) ---
# Based on nuScenes_dataset_loader.py
TARGET_HEIGHT = 294 # 294 252
TARGET_WIDTH = 518 # 518 448
RENDER_H = 294  # 294 252     # Height for final rendering (if different from input)
WIDE_W = 1036 # 1036 896       # Width for wide-angle rendering
RENDER_WIDTH = True # If True, render at RENDER_H x WIDE_W; if False, render at TARGET_HEIGHT x TARGET_WIDTH
# --- Fusion Settings ---
ENABLE_FUSION = False
BLEND_EDGE_WIDTH = 100
FUSION_METHOD = 'two_band' # 'simple' or 'two_band'

# --- Paths ---
PRETRAINED_PATH = "outputs/exp_OmniVGGT_nuScenes_omnivggt_finetune/2026-04-17_21-49-57_work1v2_vol0.002_epoch7/checkpoints/epoch_5-step_40000/" # 260203SingleFramesVol0002Epoch3Iter20000 260203SingleFramesVol0002Epoch5Iter30000
DATASET_ROOT = "datasets/nuscenes/processed_10Hz/trainval2" # UPDATE THIS 
VAL_LIST_PATH = "datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val.txt" # UPDATE THIS
FLAG_bf16 = True # Whether to use bfloat16 for inference (requires compatible GPU and PyTorch version)
# Output path generation
SAVE_ROOT_BASE = f"./renders_val{'' if RENDER_WIDTH else '_ogwidth'}_work1v2_omni_e5s4w_vol0.002_{TARGET_WIDTH}px{'_bf16' if FLAG_bf16 else ''}"
if ENABLE_FUSION:
    folder_suffix = f"fusion_{FUSION_METHOD}_{BLEND_EDGE_WIDTH}px"
else:
    folder_suffix = f"render_only{'_bf16' if FLAG_bf16 else ''}"
print("SAVE_ROOT_BASE:", SAVE_ROOT_BASE)
SAVE_ROOT = os.path.join(SAVE_ROOT_BASE, PRETRAINED_PATH.split('/')[-2], folder_suffix)

# ================= Helper Functions: Data Loading =================

def load_scene_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"Error: Val list not found at {txt_path}")
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

def read_intrinsics(scene_path, cam_id, target_h, target_w):
    """Read and normalize intrinsics (scaling to target size + 0-1 normalization)"""
    file_path = os.path.join(scene_path, "intrinsics", f"{cam_id}.txt")
    with open(file_path, 'r') as f:
        v = [float(line.strip()) for line in f if line.strip()]
    
    intr = np.eye(3, dtype=np.float32)
    if len(v) == 9:
        intr[0, 0], intr[1, 1] = v[0], v[1]
        intr[0, 2], intr[1, 2] = v[2], v[3]
    else:
        intr = np.array(v).reshape(3, 3)

    # Assume original NuScenes size for scaling factors
    original_w, original_h = 1600.0, 900.0
    
    # Scale to target resolution
    s_x = float(target_w) / original_w
    s_y = float(target_h) / original_h
    intr[0, 0] *= s_x; intr[0, 2] *= s_x
    intr[1, 1] *= s_y; intr[1, 2] *= s_y

    # Normalize to 0-1 (for model input)
    norm_intr = intr.copy()
    norm_intr[0, 0] /= float(target_w)
    norm_intr[1, 1] /= float(target_h)
    norm_intr[0, 2] /= float(target_w)
    norm_intr[1, 2] /= float(target_h)

    return torch.from_numpy(norm_intr), torch.from_numpy(intr) # Return both normalized and original intrinsics as tensors

def read_extrinsics(scene_path, timestep, cam_id):
    """Read 4x4 cam2world extrinsics"""
    file_path = os.path.join(scene_path, "cam2ego_extrinsics", f"{cam_id}.txt")
    data = np.loadtxt(file_path, dtype=np.float32)
    return torch.from_numpy(data) # [4, 4]

def load_image_tensor(scene_path, timestep, cam_id, target_h, target_w):
    """Load image and resize"""
    file_path = os.path.join(scene_path, "images", f"{timestep}_{cam_id}.jpg")
    image = Image.open(file_path).convert('RGB')
    image = image.resize((target_w, target_h), Image.BICUBIC)
    return tf.ToTensor()(image) # [3, H, W] in [0, 1]

# ================= Fusion Utils =================

def create_edge_blend_mask(h, w, edge_width, device, strategy='smoothstep'):
    mask = torch.ones((1, h, w), device=device)
    if edge_width <= 0: return mask
    t = torch.linspace(0, 1, edge_width, device=device)
    weight = 3 * t**2 - 2 * t**3 if strategy == 'smoothstep' else t
    mask[:, :, :edge_width] = weight.view(1, 1, -1)
    mask[:, :, -edge_width:] = torch.flip(weight, dims=[0]).view(1, 1, -1)
    return mask

def blend_images(render_img, gt_img, mask):
    if render_img.shape != gt_img.shape:
        gt_resized = torch.nn.functional.interpolate(
            gt_img.unsqueeze(0), size=render_img.shape[-2:], mode='bilinear'
        ).squeeze(0)
    else:
        gt_resized = gt_img
    return gt_resized * mask + render_img * (1 - mask)

def two_band_blending(render_img, gt_img, mask, blur_sigma=5):
    import torchvision.transforms as T
    blurrer = T.GaussianBlur(kernel_size=21, sigma=blur_sigma)
    render_low = blurrer(render_img)
    gt_low = blurrer(gt_img)
    render_high = render_img - render_low
    gt_high = gt_img - gt_low
    result_low = gt_low * mask + render_low * (1 - mask)
    result_high = gt_high * mask + render_high * (1 - mask)
    return result_low + result_high

# ================= Main =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {PRETRAINED_PATH}...")
    # Load model
    model = AnySplat.from_pretrained(PRETRAINED_PATH).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # Fusion Mask
    fusion_mask = None
    if ENABLE_FUSION:
        fusion_mask = create_edge_blend_mask(RENDER_H, TARGET_WIDTH, BLEND_EDGE_WIDTH, device)

    # 1. Collect Tasks
    scenes = load_scene_list(VAL_LIST_PATH)
    all_tasks = []
    
    print("Scanning scenes...")
    for scene_id in tqdm(scenes, desc="Scanning"):
        scene_dir = os.path.join(DATASET_ROOT, scene_id)
        frames = get_scene_frames(scene_dir)
        save_dir = os.path.join(SAVE_ROOT, scene_id)
        os.makedirs(save_dir, exist_ok=True)
        
        for frame_id in frames:
            for group_type in ['front', 'back']: # 0,1,2 or 3,4,5
                cam_indices = [0, 1, 2] if group_type == 'front' else [5, 4, 3]
                
                # Check if files exist
                valid = True
                for c in cam_indices:
                    if not os.path.exists(os.path.join(scene_dir, "images", f"{frame_id}_{c}.jpg")):
                        valid = False; break
                
                if valid:
                    all_tasks.append({
                        'scene_id': scene_id,
                        'frame_id': frame_id,
                        'cam_indices': cam_indices,
                        'scene_dir': scene_dir,
                        'save_dir': save_dir
                    })

    # 2. Process Batches
    num_batches = math.ceil(len(all_tasks) / BATCH_SIZE)
    print(f"Processing {len(all_tasks)} tasks in {num_batches} batches.")

    for i in tqdm(range(num_batches), desc="Inference"):
        batch_tasks = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        # Prepare Batch Data Containers
        b_images = []
        b_extrinsics = []
        b_context_extrinsics = []
        b_intrinsics = []
        b_norm_intrinsics = []
        valid_batch_tasks = []

        # Load Data for Batch
        for task in batch_tasks:
            try:
                scene_path = task['scene_dir']
                fid = task['frame_id']
                cams = task['cam_indices']
                
                # Per-view lists
                v_imgs, v_exts, v_ints, norm_v_ints = [], [], [], []
                
                for cid in cams:
                    # Load Image
                    img = load_image_tensor(scene_path, fid, cid, TARGET_HEIGHT, TARGET_WIDTH)
                    v_imgs.append(img)
                    # Load Extrinsics
                    ext = read_extrinsics(scene_path, fid, cid)
                    v_exts.append(ext)
                    # Load Intrinsics
                    norm_intr, intr = read_intrinsics(scene_path, cid, TARGET_HEIGHT, TARGET_WIDTH)
                    v_ints.append(intr)
                    norm_v_ints.append(norm_intr)
                # Stack views
                v_imgs = torch.stack(v_imgs) # [3, 3, H, W]
                v_exts = torch.stack(v_exts) # [3, 4, 4]
                v_ints = torch.stack(v_ints) # [3, 3, 3]
                norm_v_ints = torch.stack(norm_v_ints) # [3, 3, 3]

                # --- Coordinate Normalization (Relative Pose) ---
                # Match loader logic: Make 1st context view (idx 0) the origin
                # c2w_0 = v_exts[0]
                # w2c_0 = torch.inverse(c2w_0)
                # # Apply transformation: T_new = T_0^-1 * T_old
                # v_exts = torch.matmul(w2c_0.unsqueeze(0), v_exts)
                
                # Apply SE3 Inverse for model (converts 4x4 -> 3x4 effectively in many utils)
                # Using the imported util
                v_exts_final = closed_form_inverse_se3(v_exts) # get w2c
                # v_exts_final = v_exts_final[:, :3, :] # Ensure 3x4 if util returned 4x4, usually it returns [B, V, 3, 4]

                # Append to Batch Lists
                b_images.append(v_imgs)
                b_extrinsics.append(v_exts) # Keep 4x4 for decoder
                b_context_extrinsics.append(v_exts_final[:, :3, :]) # 3x4 for encoder
                b_intrinsics.append(v_ints)
                b_norm_intrinsics.append(norm_v_ints)

                # Cache for fusion
                if ENABLE_FUSION:
                    task['input_images_cpu'] = v_imgs.clone()
                
                valid_batch_tasks.append(task)

            except Exception as e:
                print(f"Failed to load task {task['scene_id']} {task['frame_id']}: {e}")
                continue

        if not b_images:
            continue

        # Stack into Batch Tensors
        # Input to model expects: [B, V, C, H, W] for images
        batch_input = {
            "context": {
                "image": torch.stack(b_images).to(device),              # [B, 3, 3, H, W]
                "extrinsics": torch.stack(b_context_extrinsics).to(device),     # [B, 3, 3, 4] for encoder
                "intrinsics": torch.stack(b_intrinsics).to(device),     # [B, 3, 3, 3]
            },
            # "target" is usually not needed for pure inference if we manually decode, 
            # or we can alias context if the model code requires it.
            "target": {} 
        }

        # Save 4x4 extrinsics for decoder
        # b_extrinsics_original = torch.stack(b_extrinsics).to(device) # [B, 3, 4, 4]

        # Normalize Images to [-1, 1] for model input (Loader does this implicitly or explicitly?)
        # Dataset loader returns [0,1] from ToTensor. 
        # The prompt's encoder code: image = (batch["context"]["image"] + 1) / 2
        # WAIT! If the model code does `(image + 1) / 2`, it implies the INPUT to the model 
        # was expected to be [-1, 1].
        # PyTorch ToTensor gives [0, 1].
        # So we must convert [0, 1] -> [-1, 1] BEFORE passing to model.
        batch_input["context"]["image"] = (batch_input["context"]["image"] * 2.0) - 1.0

        current_bs = len(valid_batch_tasks)
        views_per_group = 3

        with torch.no_grad():
            # 1. Run Encoder
            # The prompt says: encoder_output = self.encoder(batch)
            # We assume model.encoder returns Gaussians
            if FLAG_bf16:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16): # dtype=torch.bfloat16
                    # 遍历batch_input，确保都是bfloat16类型
                    # for key in batch_input:
                    #     if isinstance(batch_input[key], torch.Tensor) and batch_input[key].dtype == torch.float32:
                    #         batch_input[key] = batch_input[key].to(torch.bfloat16)
                    gaussians, _ = model.inference(batch_input)
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    gaussians, _ = model.inference(batch_input)                
            
            # 2. Run Decoder / Rendering
            # We need to construct render intrinsics. 
            # Usually we render at WIDE_W (higher res) or TARGET_WIDTH.
            # batch_render_nuscenes.py modified intrinsics for WIDE_W.
            
            # Clone intrinsics from batch (these are 0-1 normalized)
            # render_intrinsics = pred_context_pose['intrinsic'].clone()
            # render_intrinsics = pred_context_pose['intrinsic'].clone()
            # render_intrinsics = batch_input["context"]["intrinsics"].clone()
            render_intrinsics = torch.stack(b_norm_intrinsics).to(device).clone() # [B, 3, 3, 3], we will adjust fx for wide rendering
            render_extrinsics = torch.stack(b_extrinsics).to(device)
            # If rendering at WIDE_W, we need to adjust FOV or Aspect? 
            # batch_render_nuscenes.py did: new_intrinsics[..., 0, 0] *= (TARGET_W / WIDE_W)
            # NOTE: If we want a wider FOV output, we usually scale the focal length DOWN (zoom out) 
            # or keep focal length and increase image plane size. 
            # If we simply increase resolution (W -> WIDE_W) but keep intrinsics normalized, 
            # we render the SAME view at higher res.
            # To render a "Wide" view from a crop-trained model, usually we adjust focal length.
            if RENDER_WIDTH:
                width_scale = TARGET_WIDTH / WIDE_W
                render_intrinsics[..., 0, 0] *= width_scale # Scale normalized fx for wide FOV.
                # Keep center-cropped wide render aligned with normal render when cx != 0.5:
                # cx_wide = crop_offset_norm + width_scale * cx_normal,
                # where crop_offset_norm = (1 - width_scale) / 2.
                crop_offset_norm = (1.0 - width_scale) * 0.5
                render_intrinsics[..., 0, 2] = (
                    crop_offset_norm + width_scale * render_intrinsics[..., 0, 2]
                )
            # render_intrinsics[..., 0, 2] *= width_scale # Scale cx? 
            # Usually cx stays at 0.5 (center) in normalized coords.
            # If cx was 0.5 * TARGET_W, now it is 0.5 * WIDE_W. Normalized value is still ~0.5.
            # However, batch_render_nuscenes logic: `new_intrinsics[..., 0, 0] *= width_scale`. 
            # We will strictly follow the previous logic.
            
            # Define Near/Far
            t_near = torch.ones(current_bs, views_per_group, device=device) * 0.01
            t_far = torch.ones(current_bs, views_per_group, device=device) * 100.0

            # Forward Decoder
            # Note: Decoder needs EXTRINSICS. We use the 4x4 ones.
            # need c2w ext and norm intr
            outputs = model.decoder.forward(
                gaussians,
                render_extrinsics,
                render_intrinsics.float(),
                t_near,
                t_far,
                (RENDER_H, WIDE_W) if RENDER_WIDTH else (RENDER_H, TARGET_WIDTH)
            )
            
            # Output: [B, V, C, H, W]
            rendered_batch = outputs.color

            # 3. Save & Fusion
            for b in range(current_bs):
                task = valid_batch_tasks[b]
                group_render = rendered_batch[b] # [3, C, H, W]
                
                if ENABLE_FUSION:
                    # GT in [0, 1]
                    group_input_gt = task['input_images_cpu'].to(device)

                for v in range(views_per_group):
                    cam_idx = task['cam_indices'][v]
                    pred_img = group_render[v] # [3, H, W]
                    
                    # A. Save Wide Render
                    wide_name = f"{task['frame_id']}_{cam_idx}{'_wide' if RENDER_WIDTH else ''}.jpg"
                    save_image(pred_img, os.path.join(task['save_dir'], wide_name))
                    
                    # B. Fusion (Only for first view in group usually, per old script)
                    if ENABLE_FUSION and v == 0:
                        gt_img = group_input_gt[v] # [0, 1]
                        
                        # Center Crop from Wide Render
                        start_x = (WIDE_W - TARGET_WIDTH) // 2
                        pred_crop = pred_img[:, :, start_x : start_x + TARGET_WIDTH]
                        
                        # Fusion
                        if FUSION_METHOD == 'simple':
                            fusion_crop = blend_images(pred_crop, gt_img, fusion_mask)
                        elif FUSION_METHOD == 'two_band':
                            fusion_crop = two_band_blending(pred_crop, gt_img, fusion_mask)
                        else:
                            fusion_crop = pred_crop

                        # Paste back
                        pred_img_fusion = pred_img.clone()
                        pred_img_fusion[:, :, start_x : start_x + TARGET_WIDTH] = fusion_crop
                        
                        wide_fusion_name = f"{task['frame_id']}_{cam_idx}_wide_fusion.jpg"
                        save_image(pred_img_fusion, os.path.join(task['save_dir'], wide_fusion_name))

if __name__ == "__main__":
    main()