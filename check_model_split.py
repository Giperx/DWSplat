import torch
from safetensors.torch import load_file
import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ("true", "1", "yes", "y", "on"):
        return True
    if val in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("布尔值应为 true/false")


def load_state_dict_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".safetensors":
        return load_file(file_path)

    # 兼容 Lightning 的 .ckpt 以及普通 .pt/.pth 文件
    checkpoint = torch.load(file_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        # 某些权重文件直接就是 state_dict
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "无法从该文件中解析 state_dict。"
        "若是 Lightning ckpt，请确认其中包含 'state_dict' 字段。"
    )


def inspect_weights(file_path, output_dir=None, split=False):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print(f"正在加载: {file_path}")
    state_dict = load_state_dict_from_file(file_path)
    
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

    if not split:
        print("\n当前为仅检查模式（split=False），不执行拆分保存。")
        return

    if output_dir is None:
        # 默认输出到权重文件同级目录下、与权重同名(去后缀)的文件夹
        file_dir = os.path.dirname(file_path)
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(file_dir, file_stem)
    os.makedirs(output_dir, exist_ok=True)

    print("\n开始按二级模块拆分并保存 .pt:")
    print("-" * 50)

    for mod_name, mod_keys in sub_modules.items():
        # 例如 'encoder.depth_head' -> 'depth_head.pt'
        leaf_name = mod_name.split('.')[-1]
        save_path = os.path.join(output_dir, f"{leaf_name}.pt")

        module_state_dict = {}
        for k in mod_keys:
            # 去掉前缀，便于后续 load_state_dict 到对应子模块
            # 例如 'encoder.depth_head.norm.bias' -> 'norm.bias'
            new_key = k[len(mod_name) + 1:] if k.startswith(mod_name + ".") else k
            module_state_dict[new_key] = state_dict[k]

        torch.save(module_state_dict, save_path)
        print(f"已保存: {save_path}  (键数: {len(module_state_dict)})")

    print("\n拆分完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查并按二级模块拆分权重（支持 safetensors / ckpt）")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="权重文件路径，支持 .safetensors / .ckpt / .pt / .pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="拆分后的 pt 输出目录（仅在 split=True 时生效）",
    )
    parser.add_argument(
        "--split",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="是否执行拆分保存。支持 '--split' 或 '--split true/false'",
    )
    args = parser.parse_args()

    inspect_weights(args.file, output_dir=args.output_dir, split=args.split)
    
# python check_model_split.py --split true  --file /path/to/model.ckpt  --output-dir /path/to/split_pt