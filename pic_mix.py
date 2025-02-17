import pandas as pd
import src.main_process as main_process
from pathlib import Path
import src.csv_analyser as analy
# Paths and configuration
paths = {
    "base_image_folder": Path(r"D:\H23057-J_とめ研社内共有\画像データ\アノテーション済\20250207_bareSi\ベアSi\input_data"),
    "mask_image_folder": Path(r"C:\Users\huang\work\bevel-ml\trunk\data\output\20250207_new_with_watermark\rd4ad_resnet18_512_cc\version_0\mask"),
    "back_image_folder": Path(r"D:\H23057-J_とめ研社内共有\画像データ\アノテーション済\20250120\パターン付き\input_data"),
    "output_root": Path(r"data\data_final_test\output"),
    "aligned_data_path": Path(r"data\data_final_test\merged_reference_data_updated.csv"),
}

calibration_path=Path(r'data\data_final_test\pair_add_hige.csv')#information for all anomaly pics
#anomaly_scores_path=Path(r"C:\Users\huang\work\bevel-ml\trunk\data\output\20241216\rd4ad_resnet18_512_cc_70epoch\version_1\anomalymaps.csv")
df = pd.read_csv(calibration_path,header=None)
base_image_files = df.iloc[:, 0].values   
back_image_files = df.iloc[:, 1].values
#anomaly_scores_df = pd.read_csv(anomaly_scores_path, header=None, index_col=0)#todo change to accept pics

#thresholds = df.iloc[:, 2:5].values.tolist()
#column_names = df.iloc[:, 5].values
#scale_factors=df.iloc[:, 6].values.tolist()
#scale_factors=analy.convert_to_2d_list(scale_factors)

for base_image_file,back_image_file in zip(base_image_files,back_image_files):
    column_name,scale_factors=analy.filename_analy(base_image_file,back_image_file)
    main_process.generate_pics(paths,base_image_file,back_image_file,None,
                               *scale_factors,threshold=0.08,column_name=column_name)   

# base_image_file="1_06_03_C1_0004.tif"
# back_image_file="2_01_01_C1_0001.tif"
# column_name,scale_factors=analy.filename_analy(base_image_file,back_image_file)
# main_process.generate_pics(paths,base_image_file,back_image_file,None,
#                                *scale_factors,threshold=0.08,column_name=column_name) 
print("done")
