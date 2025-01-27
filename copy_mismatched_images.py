import pandas as pd
import shutil
from pathlib import Path

def copy_mismatched_images(csv_path, source_folder, target_folder,output_csv):

    df = pd.read_csv(csv_path)
    mismatched = df[df['actual_class'] != df['predicted_class']]

    #image_ids = mismatched['image_id'].tolist()

    # Path(target_folder).mkdir(parents=True, exist_ok=True)

    # for image_id in image_ids:
    #     source_path = Path(source_folder) / f"{image_id}.tif" 
    #     target_path = Path(target_folder) / f"{image_id}.tif"
    #     if source_path.exists():
    #         shutil.copy(source_path, target_path)
    #         print(f"Copied: {source_path} -> {target_path}")
    #     else:
    #         print(f"File not found: {source_path}")
    mismatched[['image_id']].to_csv(output_csv, index=False)
    print("File copying completed.")


csv_file = r"C:\Users\huang\work\bevel-ml\trunk\data\output\20250123\resnet50_cc_epoch40\version_0\prediction.csv"
source_dir = "data/images"
target_dir = "data/mismatched_images"
output_csv="mismatched_images.csv"

copy_mismatched_images(csv_file, source_dir, target_dir,output_csv)
