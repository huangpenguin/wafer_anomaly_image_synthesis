import pandas as pd
import src.main_process as main_process
from pathlib import Path

###
# prepare data
###

# import src.pic_preprocess as pic_prepro
# from pathlib import Path
# path_a=Path(r"C:\Users\huang\work\bevel-ml\trunk\data\input\20241212_test")
# pic_prepro.split_images_in_folder(path_a,path_a)
#TODO make split_image_info.csv~~~~~~~~~~~~~~~~~~~~~~~~

###
# train selected files
###

# Paths and configuration
base_paths = {
    "base_image_folder": Path(r"calibration\data\20241217\anomaly"),
    "mask_image_folder": Path(r"calibration\data\20250117\mask"),
    "back_image_folder": Path(r"calibration\data\20241217\normal"),
    "output_root": Path(r"calibration\data\20250127_test\output"),
    "aligned_data_path": Path(r"calibration\data\merged_reference_data_updated.csv"),
}
#information for all anomaly pics
calibration_path=Path(r'calibration\data\20241217\pair_add_hige.csv')

anomaly_scores_path=Path(r"C:\Users\huang\work\bevel-ml\trunk\data\output\20241216\rd4ad_resnet18_512_cc_70epoch\version_1\anomalymaps.csv")

# pic_prepro.make_anomaly_masks_in_folder(
#     anomaly_scores_df,
#     thresholds=[0.02,0.08,0.16,0.32],
#     output_folder=mask_image_folder,) 
df = pd.read_csv(calibration_path,header=None)
base_image_files = df.iloc[:, 0].values   
back_image_files = df.iloc[:, 1].values
thresholds = df.iloc[:, 2:5].values.tolist()
column_names = df.iloc[:, 5].values
scale_factors=df.iloc[:, 6].values.tolist()
anomaly_scores_df = pd.read_csv(anomaly_scores_path, header=None, index_col=0)

scale_factors=main_process.convert_to_2d_list(scale_factors)

# base_image_file="1_12_02_A1_0102_O.tif"
# back_image_file="2_02_01_A1_0094_O.tif"
# main_process.generate_pics(base_paths,base_image_file,back_image_file,anomaly_scores_df,threshold=0.02,scale_factor=-1,column_name="Dis_cut")

for base_image_file, back_image_file,threshold_list,column_name,scale_factor_list in zip(base_image_files, back_image_files,thresholds,column_names,scale_factors):
    for threshold in threshold_list:
        for scale_factor in scale_factor_list:
            main_process.generate_pics(base_paths,base_image_file,back_image_file,anomaly_scores_df,threshold=threshold,scale_factor=float(scale_factor),column_name=column_name)   

print("done")