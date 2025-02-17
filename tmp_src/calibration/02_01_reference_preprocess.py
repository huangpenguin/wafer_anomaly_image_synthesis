import os
import numpy as np
from PIL import Image, ImageChops, ImageDraw
from pathlib import Path
from torchvision import transforms
########################################################################################


# def split_images_in_folder(folder_path: Path):
#     """
#     Split all images in the folder into two halves and append 'I' and 'O' to the original file names.

#     :param folder_path: Path to the folder containing images.
#     """
#     if not folder_path.is_dir():
#         print(f"{folder_path} is not a valid folder path.")
#         return

#     for image_file in folder_path.glob('*'):
#         #if 'C' in image_file.stem.upper(): 
#         if set('COI') & set(image_file.stem.upper()):
#             print(f"Skipping {image_file} as it contains 'C'.")
#             continue
#         if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif']:
#             with Image.open(image_file) as img:
#                 width, height = img.size
                
#                 left_image = img.crop((0, 0, width // 2, height))
#                 right_image = img.crop((width // 2, 0, width, height))

#                 left_image_file = image_file.with_stem(image_file.stem + '_I')
#                 right_image_file = image_file.with_stem(image_file.stem + '_O')

#                 left_image.save(left_image_file)
#                 right_image.save(right_image_file)

#                 print(f"Saved: {left_image_file} and {right_image_file}")
                
# def resize_image(image_path:str,size:list):
#     """
#     pic change to special size 
    
#     Args:
#         image_path (str or Path): 
#         output_path (str or Path):
        
#     output: 
#         output path of resized pics     
#     """
        
#     output_path = Path(f"{image_path.parent}/resized_{image_path.stem}{image_path.suffix}")
    
#     #create instance
#     transform = transforms.Compose([
#         transforms.Resize(size), 
#         #transforms.ToTensor(), # (C,H,W)        
#         #transforms.ToPILImage()      
#     ])
    
#     try:
    
#         img = Image.open(image_path).convert("RGB")  
  
#         resized_img = transform(img)
#         #resized_img=transform().Resize((512, 512))(img)

#         resized_img.save(output_path)
#         print(f"Resized image saved to {output_path}")
#     except Exception as e:
#         print(f"Error resizing image: {e}")
        
#     return output_path
##################################################################################
thresholds = [0.36, 0.16, 0.08, 0.02]# from trunk code 

mask_name_list = [
    r"1_06_05_C1_0213.png",
    r"1_06_05_E1_0130_O.png",
    r"1_08_01_A1_0076_O.png",
    r"1_11_01_A1_0172_O.png",
    r"1_15_01_A1_0109_O.png"
]

pic_path_list = [
    r"calibration/data/20241203_reference_preprocessed/1_06_05_C1_0213.tif",
    r"calibration\data\20241203_reference_preprocessed\1_06_05_E1_0130_O.tif",
    r"calibration\data\20241203_reference_preprocessed\1_08_01_A1_0076_O.tif",
    r"calibration\data\20241203_reference_preprocessed\1_11_01_A1_0172_O.tif",
    r"calibration\data\20241203_reference_preprocessed\1_15_01_A1_0109_O.tif"
]

output_path_list = [
    r"calibration/data/20241203_reference_preprocessed/pre_1_06_05_C1_0213.tif",
    r"calibration\data\20241203_reference_preprocessed\pre_1_06_05_E1_0130_O.tif",
    r"calibration\data\20241203_reference_preprocessed\pre_1_08_01_A1_0076_O.tif",
    r"calibration\data\20241203_reference_preprocessed\pre_1_11_01_A1_0172_O.tif",
    r"calibration\data\20241203_reference_preprocessed\pre_1_15_01_A1_0109_O.tif"
]
#split_images_in_folder(Path(pic_path_list[0]).parent)
####################################################################################
for mask_name, pic_path, output_path in zip(mask_name_list, pic_path_list, output_path_list):
    for threshold in thresholds:
        hist_path = Path(r"C:\Users\huang\work\bevel-ml\trunk\data\output\20241118\rd4ad_resnet18_512_cc_70epoch\version_0\anomaly_regions_hist_abstract")
        updated_mask_path=hist_path / f"{threshold}"/mask_name
        pic_path=resize_image(Path(pic_path),(512, 512))#TODO:OPEN操作重合
        mask_image = Image.open(updated_mask_path).convert("L")  # 确保 Mask 是灰度图
        original_image = Image.open(pic_path)
        
        masked_image_mean,masked_image_mode=apply_mask(mask_image, original_image, mask_name)#array

        #masked_image_mean,masked_image_mode = Image.fromarray(masked_image_mean), Image.fromarray(masked_image_mode)
        masked_image_mean_diff = ImageChops.difference(Image.fromarray(masked_image_mean),original_image)
        masked_image_mode_diff = ImageChops.difference(original_image,Image.fromarray(masked_image_mode))
        

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        output_path = Path(output_path) 
        updated_output_mean_path = output_path.parent / f"{threshold}_{output_path.stem}_mean.png"
        updated_output_mode_path = output_path.parent / f"{threshold}_{output_path.stem}_mode.png"
        updated_output_mean_diff_path = output_path.parent / f"{threshold}_{output_path.stem}_mean_diff.png"
        updated_output_mode_diff_path = output_path.parent / f"{threshold}_{output_path.stem}_mode_diff.png"
        # 保存结果
        Image.fromarray(masked_image_mean).save(updated_output_mean_path)
        Image.fromarray(masked_image_mode).save(updated_output_mode_path)
        masked_image_mean_diff.save(updated_output_mean_diff_path)
        masked_image_mode_diff.save(updated_output_mode_diff_path)
        print(f"保存 Mask 应用结果到: {output_path}")