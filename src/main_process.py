from pathlib import Path
from PIL import Image
import numpy as np
import src.pic_process as pic_pro
import src.pic_preprocess as pic_prepro


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
    anomaly_scores_df, mask_image_file, threshold, mask_image_folder
):
    """
    Processes a mask image by generating it and loading it as a numpy array.

    Args:
        anomaly_scores_df (DataFrame): DataFrame containing anomaly scores.
        mask_image_file (str): Name of the mask image file.
        threshold (float): Threshold value for generating the mask.
        mask_image_folder (Path): Folder to save the generated mask.

    Returns:
        tuple: A tuple containing:
            - Mask Image object
            - numpy array representation of the mask
            - Mask image stem (filename without extension)
    """
    pic_prepro.make_anomaly_mask_by_name(
        anomaly_scores_df,
        file_stem=Path(mask_image_file).stem,
        threshold=threshold,
        output_folder=mask_image_folder,
    )
    mask_image_path = (
        mask_image_folder
        / f"threshold_{threshold}"
        / (Path(mask_image_file).stem + ".tif")
    )
    mask_image = Image.open(mask_image_path)
    mask_array = np.array(mask_image, dtype=np.int16)
    return mask_image, mask_array, Path(mask_image_file).stem


def generate_pics(
    base_paths,
    base_image_file,
    back_image_file,
    anomaly_scores_df,
    threshold=None,
    scale_factor=1,
    column_name="Dis_edge",
):
    """
    Generates processed images using anomaly scores and saves them to the output folder.

    Args:
        base_paths(dic):dictionaries of all paths used in this function.
        base_image_file (str): Name of the base image file."Anomaly image"
        back_image_file (str): Name of the background image file."Background image"
        anomaly_scores_df (DataFrame): DataFrame containing anomaly scores.
        threshold (float, optional): Threshold value for anomaly detection (default: None).
        scale_factor (float, optional): Scaling factor for the reference image (default: 1).
        column_name (str): Name to determine which distance to use.

    Returns:
        None
    """
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

    # Determine threshold
    if threshold is None:
        threshold = pic_prepro.get_best_threshold(
            anomaly_scores_df, Path(base_image_file).stem, is_vertical
        )

    # Preprocess images
    base_image, base_array, base_image_stem = preprocess_image(
        base_image_folder, base_image_file
    )
    back_image, back_array, back_image_stem = preprocess_image(
        back_image_folder, back_image_file
    )
    (
        mask_image_from_anomaly_scores,
        mask_array_from_anomaly_scores,
        mask_image_stem_from_anomaly_scores,
    ) = process_mask_image(
        anomaly_scores_df, base_image_file, threshold, mask_image_folder
    )

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
    aligned_diff_image = pic_pro.scale_image(aligned_diff_image, scale_factor)
    result_array = pic_pro.apply_difference_to_image(
        aligned_back_image, aligned_diff_image
    )
    result_array = pic_pro.from_int16_to_uint8(result_array)

    # Save results
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

def convert_to_2d_list(data):
    """
    Converts a list of strings containing numbers or colon-separated numbers into a 2D list of integers.

    Args:
        data (list of str): A list where each element is either a single number or a colon-separated string of numbers.

    Returns:
        list of list of int: A 2D list where each inner list contains integers parsed from the input strings.
    
    Example:
        data = ['1', '1', '1', '1', '1', '1:-1', '1:-1']
        result = convert_to_2d_list(data)
        print(result)  # Output: [[1], [1], [1], [1], [1], [1, -1], [1, -1]]
    """
    result = []
    for item in data:
        if ':' in item:
            values = item.split(':')
            values = [int(val) for val in values]
            result.append(values)
        else:
            result.append([int(item)])
    
    return result
