from PIL import Image
import src.pic_process as pic_pro
import src.pic_preprocess as pic_prepro
from pathlib import Path
import numpy as np

# output_path = Path(f"{image_path.parent}/resized_{image_path.stem}{image_path.suffix}")

def generate_pics(base_image_file,back_image_file,anomaly_scores_df,threshold=None,scale_factor=1):
    '''
    Process:TODO???
    
    
    '''
    
    ##########################################################################################################
    mask_image_file=base_image_file
    #mode=Path(base_image_file).stem[-1]
    base_image_folder = Path(r"C:\Users\huang\work\dataset\calibration\data\20241217\anomaly")
    mask_image_folder = Path(r"C:\Users\huang\work\dataset\calibration\data\20241220\mask")
    back_image_folder = Path(r"C:\Users\huang\work\dataset\calibration\data\20241217\normal")
    output_root = Path(r"C:\Users\huang\work\dataset\calibration\data\20241220\output")
    output_folder = output_root / f"{Path(base_image_file).stem}-{Path(back_image_file).stem}"
    aligned_data_path=Path(r"C:\Users\huang\work\dataset\calibration\data\merged_reference_data.csv")#位置校准数据
    is_vertical="C" in base_image_file
    if threshold is None:    
        threshold=pic_prepro.get_best_threshold(anomaly_scores_df,Path(mask_image_file).stem,is_vertical)
    #########################################################################################################
    
    ###
    # anomaly pic_preprocess
    ###
    base_image_path = base_image_folder / base_image_file
    base_image=Image.open(base_image_path).convert('L')#uint8
    base_image=pic_prepro.resize_image(img=base_image,size=(512,512),save=False)
    base_array=np.array(base_image,dtype=np.int16)
    base_image_stem=base_image_path.stem
   
    ###
    # normal pic_preprocess
    ###
    back_image_path = back_image_folder / back_image_file
    back_image=Image.open(back_image_path).convert('L')
    back_image=pic_prepro.resize_image(img=back_image,size=(512,512),save=False)
    back_array=np.array(back_image,dtype=np.int16)
    back_image_stem=back_image_path.stem
    
    ###
    # mask pic_preprocess
    ###
    pic_prepro.make_anomaly_mask_by_name(
        anomaly_scores_df,
        file_stem=Path(mask_image_file).stem,
        threshold=threshold,
        output_folder=mask_image_folder,) 

    mask_image_path = mask_image_folder / f"threshold_{threshold}" / (Path(mask_image_file).stem+".tif")
    mask_image = Image.open(mask_image_path)
    mask_array=np.array(mask_image,dtype=np.int16)
    mask_image_stem=Path(mask_image_file).stem
      
    base_image_dis_from_edge=pic_pro.get_distance(base_image_stem,aligned_data_path)
    back_image_dis_from_edge=pic_pro.get_distance(back_image_stem,aligned_data_path)  
    
    reference_image_array=pic_pro.generate_reference_array(base_array,mask_array,is_vertical,output_type="mean")
    
    diff_image_mean_array = pic_pro.calculate_image_difference(base_array, reference_image_array)
    
    masked_diff_image_array=pic_pro.apply_mask_to_image(diff_image_mean_array,mask_array)
    
    aligned_base_image, aligned_reference_image=pic_pro.align_image_arrays(back_array,back_image_dis_from_edge,masked_diff_image_array,base_image_dis_from_edge,is_vertical,False)
    
    aligned_reference_image = pic_pro.scale_image(aligned_reference_image, scale_factor)
    
    result_array = pic_pro.apply_difference_to_image(aligned_base_image, aligned_reference_image)
    
    result_array=pic_pro.from_int16_to_uint8(result_array)

    if not output_folder.exists():
        output_folder.mkdir(parents=True,exist_ok=True)
    output_filename = f"(result)_scale{scale_factor}_mask{threshold}.png"
    
    Image.fromarray(result_array).save(output_folder / output_filename)
    base_image.save(output_folder / f"(anomaly){base_image_stem}.png")
    mask_image.save(output_folder / f"(mask){threshold}_{mask_image_stem}.png")
    back_image.save(output_folder / f"(target){back_image_stem}.png")
    
    print(f"Finish making pics with {base_image_file} and {back_image_file}")