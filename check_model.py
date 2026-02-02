import torch
from safetensors.torch import load_file
import os

def inspect_weights(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print(f"正在加载: {file_path}")
    # 加载 safetensors 文件
    state_dict = load_file(file_path)
    
    # 获取所有的键
    keys = list(state_dict.keys())
    keys.sort()

    # 统计二级模块 (例如 'encoder.aggregator', 'encoder.backbone' 等)
    sub_modules = {}
    for k in keys:
        parts = k.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
        else:
            prefix = parts[0]
            
        if prefix not in sub_modules:
            sub_modules[prefix] = []
        sub_modules[prefix].append(k)

    print("\n" + "="*50)
    print(f"总计权重键数: {len(keys)}")
    print("="*50)

    print("\n模块级概览 (前二级):")
    for mod_name, mod_keys in sub_modules.items():
        print(f"- {mod_name:30} ({len(mod_keys):4} 个键)")

    print("\n每个子模块的第一个键名示例:")
    print("-" * 50)
    for mod_name, mod_keys in sub_modules.items():
        print(f"[{mod_name}]: {mod_keys[0]}")

if __name__ == "__main__":
    # 根据你的目录结构，文件路径应该是：
    target_file = "./anysplat_hfog_1108/model.safetensors"
    inspect_weights(target_file)
    
# 正在加载: ./anysplat_hfog_1108/model.safetensors

# ==================================================
# 总计权重键数: 1405
# ==================================================

# 模块级概览 (前二级):
# - encoder.aggregator             (1210 个键)
# - encoder.camera_head            (  69 个键)
# - encoder.depth_head             (  62 个键)
# - encoder.gaussian_param_head    (  64 个键)

# 每个子模块的第一个键名示例:
# --------------------------------------------------
# [encoder.aggregator]: encoder.aggregator.camera_token
# [encoder.camera_head]: encoder.camera_head.embed_pose.bias
# [encoder.depth_head]: encoder.depth_head.norm.bias
# [encoder.gaussian_param_head]: encoder.gaussian_param_head.input_merger.0.bias