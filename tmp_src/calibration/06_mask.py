import os
import numpy as np
from PIL import Image
from pathlib import Path

def apply_diff_as_mask(before_diff_image, diff_image, threshold):
    diff_image_array = np.array(diff_image)
    mask = diff_image_array > threshold  # 根据阈值生成二值 Mask

    before_diff_image_array = np.array(before_diff_image)
    masked_image = np.copy(before_diff_image_array)
    masked_image[~mask] = 0  # 非 Mask 区域置为 0

    return masked_image, mask

# 路径列表
mask_path_list = [
    r"1_06_05_C1_0213.png",
    r"1_06_05_E1_0130_O.png",
    r"1_11_01_A1_0172_O.png",
]

pic_path_list = [
    r"calibration\data\20241121_reference\1_06_05_C1_0213\result\normalized_resized_aligned_diff_image.png",
    r"calibration\data\20241121_reference\1_06_05_E1_0130\result\normalized_resized_aligned_diff_image_O.png",
    r"calibration\data\20241121_reference\1_11_01_A1_0172\result\normalized_resized_aligned_diff_image_O.png",
]

output_path_list = [
    r"calibration\data\20241121_reference\1_06_05_C1_0213\result\1_06_05_C1_0213.png",
    r"calibration\data\20241121_reference\1_06_05_E1_0130\result\1_06_05_E1_0130_O.png",
    r"calibration\data\20241121_reference\1_11_01_A1_0172\result\1_11_01_A1_0172_O.png",
]
thresholds = [0.36, 0.16, 0.08, 0.02]



for mask_path, pic_path, output_path in zip(mask_path_list, pic_path_list, output_path_list):
    for threshold in thresholds:
        # 加载 Mask 和原图
        hist_path = Path(r"C:\Users\huang\work\bevel-ml\trunk\data\output\20241118\rd4ad_resnet18_512_cc_70epoch\version_0\anomaly_regions_hist_abstract")
        updated_mask_path=hist_path / f"{threshold}"/mask_path
        mask_image = Image.open(updated_mask_path).convert("L")  # 确保 Mask 是灰度图
        original_image = Image.open(pic_path)

        # 应用 Mask
        masked_image, _ = apply_diff_as_mask(original_image, mask_image, threshold=1)# 如果 Mask 已经是二值图像（0 或 255），threshhold无效

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 在文件名前加前缀
        output_path = Path(output_path) 
        updated_output_path = output_path.parent / f"{threshold}_{output_path.name}"
        # 保存结果
        Image.fromarray(masked_image).save(updated_output_path)
        print(f"保存 Mask 应用结果到: {output_path}")
