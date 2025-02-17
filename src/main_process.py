from pathlib import Path
from PIL import Image
import numpy as np
import src.pic_process as pic_pro
import src.pic_preprocess as pic_prepro
import pandas as pd


def preprocess_image(image_folder, image_file, size=(512, 512), mode="L"):
    """
    Preprocesses an image by loading, resizing, and converting to a numpy array.

    Args:
        image_folder (Path): Folder containing the image.
        image_file (str): Name of the image file.
        size (tuple): Target size for resizing (default: (512, 512)).
        mode (str): Image mode for conversion (default: 'L' for grayscale).

    Returns:
        tuple: A tuple containing:
            - Image object
            - numpy array representation of the image
            - Image stem (filename without extension)
    """
    image_path = image_folder / image_file
    image = Image.open(image_path).convert(mode)
    image = pic_prepro.resize_image(img=image, size=size, save=False)
    image_array = np.array(image, dtype=np.int16)
    return image, image_array, image_path.stem


def process_mask_image(
    mask_image_file: str,
    threshold: float,
    mask_image_folder: Path,
    anomaly_scores_df=None
) -> dict:
    """Processes a mask image by generating it and loading it as a NumPy array.

    If an anomaly scores DataFrame is provided, this function first generates a mask
    image using the provided threshold and saves it into the output folder. It then loads
    the mask image from the output folder, converts it to a NumPy array, and returns the
    image object, the array, and the file stem.

    Args:
        mask_image_file (str): The name of the mask image file.
        threshold (float): The threshold value for generating the mask.
        mask_image_folder (Path): The folder where the generated mask image is saved.
        anomaly_scores_df (DataFrame, optional): A DataFrame containing anomaly scores. Defaults to None.

    Returns:
        dict: A dictionary with the following keys:
            - "mask_image": The PIL Image object of the mask.
            - "mask_array": The NumPy array representation of the mask.
            - "mask_stem": The file stem (filename without extension) of the mask image.
    """
    if anomaly_scores_df is not None:
        pic_prepro.make_anomaly_mask_by_name(
            anomaly_scores_df,
            file_stem=Path(mask_image_file).stem,
            threshold=threshold,
            output_folder=mask_image_folder,
        )

    # mask_image_path = mask_image_folder / (Path(mask_image_file).stem + ".png")
    # mask_image = Image.open(mask_image_path)
    # mask_array = np.array(mask_image, dtype=np.int16)
    # mask_stem = Path(mask_image_file).stem

    mask_image = pic_prepro.find_and_open_image(mask_image_folder, mask_image_file)
    mask_array = np.array(mask_image, dtype=np.int16)
    mask_stem = Path(mask_image_file).stem

    return {
        "mask_image": mask_image,
        "mask_array": mask_array,
        "mask_stem": mask_stem,
    }

def generate_pics(
    base_paths:dict,
    base_image_file:str,
    back_image_file:str,
    anomaly_scores_df:pd.DataFrame=None,
    *scale_factors:tuple,
    threshold=None,
    column_name="Dis_edge",
)->None:#todo fix description of func
    """
    Generates processed images using anomaly scores and saves them to the output folder.

    Args:
        base_paths(dic):dictionaries of all paths used in this function.
        base_image_file (str): Name of the base image file."Anomaly image"
        back_image_file (str): Name of the background image file."Background image"
        anomaly_scores_df (DataFrame): DataFrame containing anomaly scores.
        threshold (float, optional): Threshold value for anomaly detection (default: None).
        scale_factor (tuple): Scaling factors for the reference image.
        column_name (str): Name to determine which distance to use.

    Returns:
        None
    """
    if not scale_factors:
            scale_factors = (1.0,)
    else:
        scale_factors = tuple(map(float, scale_factors))

    is_vertical = "C" in base_image_file
    base_image_folder = base_paths["base_image_folder"]
    mask_image_folder = base_paths["mask_image_folder"]
    back_image_folder = base_paths["back_image_folder"]
    output_folder = (
        base_paths["output_root"]
        / f"{Path(base_image_file).stem}-{Path(back_image_file).stem}"
    )
    aligned_data_path = base_paths["aligned_data_path"]  # calibration data
    output_folder.mkdir(parents=True, exist_ok=True)

    # Preprocess images
    base_image, base_array, base_image_stem = preprocess_image(
        base_image_folder, base_image_file
    )
    back_image, back_array, back_image_stem = preprocess_image(
        back_image_folder, back_image_file
    )

    # self-Determine threshold
    if threshold is None:
        threshold = pic_prepro.get_best_threshold(
            anomaly_scores_df, Path(base_image_file).stem, is_vertical
        )
    

    result = process_mask_image(
            base_image_file, threshold, mask_image_folder,anomaly_scores_df)
    
    mask_image_from_anomaly_scores= result["mask_image"]
    mask_array_from_anomaly_scores= result["mask_array"]
    mask_image_stem_from_anomaly_scores= result["mask_stem"]
    #########################################################################
    # Align and process images(cut distance/edge distance)
    base_image_dis_from_edge = pic_pro.get_distance(
        base_image_stem, aligned_data_path, column_name
    )
    (base_image_dis_from_edge, base_array_lined) = (
        pic_pro.correct_distance_and_draw_line(
            base_image_dis_from_edge, is_vertical, base_array
        )
    )
    back_image_dis_from_edge = pic_pro.get_distance(
        back_image_stem, aligned_data_path, column_name
    )#todo make sure the column name is correct
    (back_image_dis_from_edge, back_array_lined) = (
        pic_pro.correct_distance_and_draw_line(
            back_image_dis_from_edge, is_vertical, back_array
        )
    )
    reference_image_array = pic_pro.generate_reference_array(
        base_array, mask_array_from_anomaly_scores, is_vertical, output_type="mean"
    )    
    diff_array = pic_pro.calculate_image_difference(base_array, reference_image_array)
    masked_diff_array = pic_pro.apply_mask_to_image(
        diff_array, mask_array_from_anomaly_scores
    )
    aligned_back_image, aligned_diff_image = pic_pro.align_image_arrays(
        back_array,
        back_image_dis_from_edge,
        masked_diff_array,
        base_image_dis_from_edge,
        is_vertical,
        False,
    )  # shift masked_diff_array


    # Scale and save results
    for scale_factor in scale_factors:
        aligned_diff_image = pic_pro.scale_image(aligned_diff_image, scale_factor)
        result_array = pic_pro.apply_difference_to_image(
            aligned_back_image, aligned_diff_image
        )
        result_array = pic_pro.from_int16_to_uint8(result_array)

        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        output_filename = f"(result)_scale{scale_factor}_mask{threshold}.png"
        Image.fromarray(result_array).save(output_folder / output_filename)


    Image.fromarray(base_array_lined).save(output_folder / f"(anomaly){base_image_stem}.png")
    Image.fromarray(back_array_lined).save(output_folder / f"(target){back_image_stem}.png")
    #base_image.save(output_folder / f"(anomaly){base_image_stem}.png")
    #back_image.save(output_folder / f"(target){back_image_stem}.png")

    mask_image_from_anomaly_scores.save(
        output_folder / f"(mask_from_anomaly_scores){threshold}_{mask_image_stem_from_anomaly_scores}.png"
    )

    #print(f"Finish making pics with {base_image_file} and {back_image_file}")


