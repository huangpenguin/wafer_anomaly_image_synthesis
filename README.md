### Since the data used here comes from a semiconductor manufacturing company, only the data structure is described, and specific data details are omitted.

### Data Description
base_image_file and back_image_file

base_image_file refers to the original anomalous image used for synthesis.
back_image_file refers to the normal background image.
anomaly_scores_df

This is a CSV file containing anomaly scores for 512x512 images.
The anomaly detection model used is RD4AD, which was trained on normal images and outputs anomaly scores for each pixel.
The data format is structured as (pic_nums, [512,512]), where each row represents the anomaly scores of a single image.

