from PIL import Image

def crop_center_image(input_path, output_path):
    """
    从 1036x294 图像中，裁剪中间 518x294 区域并保存
    """
    # 打开图像
    img = Image.open(input_path)
    width, height = img.size  # (1036, 294)

    # 目标尺寸
    target_w = 518
    target_h = 294

    # 计算居中裁剪的坐标（左右裁，高度不动）
    left = (width - target_w) // 2
    top = 0
    right = left + target_w
    bottom = target_h

    # 裁剪
    cropped_img = img.crop((left, top, right, bottom))
    
    # 保存
    cropped_img.save(output_path)
    print(f"裁剪完成！新尺寸: {cropped_img.size}")
    print(f"保存路径: {output_path}")

# ====================== 使用 ======================
if __name__ == "__main__":
    # 在这里改你的输入图片路径和输出路径
    INPUT_IMG = "renders_val_work1v2_omni_e5s4w_vol0.002_518px/epoch_5-step_40000/render_only_bf16/002/000_0_wide.jpg"    # 输入 1036x294
    OUTPUT_IMG = "./output.jpg"  # 输出 518x294

    crop_center_image(INPUT_IMG, OUTPUT_IMG)