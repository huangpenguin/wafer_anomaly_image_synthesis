import numpy as np
from PIL import Image
from pathlib import Path

def enhance_image(image, scale_factor):
    """
    增强图像的对比度，将灰度值扩大指定倍数，超出范围的部分截断为255。
    
    参数:
        image (numpy.ndarray): 输入的灰度图像。
        scale_factor (float): 放大的倍数。
    
    返回:
        numpy.ndarray: 增强后的灰度图像。
    """
    enhanced_image = image.astype(np.float32) * scale_factor  # 按倍数放大
    enhanced_image = np.clip(enhanced_image, 0, 255)  # 超过255的部分截断
    return enhanced_image.astype(np.uint8)  # 转换回 uint8 类型

def get_all_image_paths(directory, extensions=("png", "jpg", "jpeg", "bmp")):
    """获取目录下指定扩展名的所有图像文件路径"""
    return [p for ext in extensions for p in Path(directory).rglob(f"*.{ext}")]

def get_global_min_max(image_paths: list) -> tuple:
    """
    获取所有图像的全局最小值和最大值。
    
    Args:
        image_paths (list): 图像文件路径列表。
    
    Returns:
        tuple: (global_min, global_max)，全局最小值和最大值。
    """
    global_min, global_max = float('inf'), float('-inf')
    
    # 遍历所有图片，计算最小值和最大值
    for path in image_paths:
        img = np.array(Image.open(path).convert("L"))  # 转为灰度图
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())
    
    return global_min, global_max

def normalize_with_global_min_max(image: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """
    使用全局最小值和最大值对图像进行归一化。
    
    Args:
        image (np.ndarray): 输入图像。
        global_min (float): 全局最小值。
        global_max (float): 全局最大值。
    
    Returns:
        np.ndarray: 归一化后的图像。
    """
    normalized_image = ((image - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    return normalized_image


image_paths = get_all_image_paths(r"calibration\data\20241203_reference_preprocessed")
global_min, global_max = get_global_min_max(image_paths)

scale_factor = 1.5

processed_images = []
for path in image_paths:
    img = np.array(Image.open(path).convert("L"))  # 转为灰度图
    #normalized_img = normalize_with_global_min_max(img, global_min, global_max)
    enhanced_img = enhance_image(img, scale_factor)  # 增强图像
    processed_images.append(enhanced_img)
    

    # 构造目标目录和新文件名
    new_dir = path.parent / "result_scale"
    new_dir.mkdir(parents=True, exist_ok=True)  # 使用 Path.mkdir 创建目录
    new_filename = f"scaled_{path.name}"
    new_path = new_dir / new_filename

    # 保存归一化后的图像
    Image.fromarray(enhanced_img).save(new_path)
    print(f"保存归一化后的图像到: {new_path}")




