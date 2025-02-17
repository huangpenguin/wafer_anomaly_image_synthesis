from PIL import Image
import numpy as np
from pathlib import Path
from scipy.stats import mode
import pandas as pd
from typing import Tuple, Union
import cv2


def align_image_arrays(
    base_image: np.ndarray,
    base_image_distance: float,
    reference_image: np.ndarray,
    reference_image_distance: float,
    is_vertical: bool = True,
    return_gap_mask: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Aligns the reference image with the base image based on specified distances.

    This function shifts the reference image either vertically or horizontally to align it with the base image.

    Args:
        base_image (np.ndarray): The base image used as a reference.
        base_image_distance (float): Distance from a specific edge of the base image.
        reference_image (np.ndarray): The image to be aligned.
        reference_image_distance (float): Distance from the same edge of the reference image.
        is_vertical (bool): If True, aligns vertically; if False, horizontally.
        return_gap_mask (bool): If True, returns a mask indicating gap areas.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            - If `return_gap_mask` is False:
                A tuple of two aligned image arrays:
                (aligned_base_image, shifted_reference_image).
            - If `return_gap_mask` is True:
                A tuple of two aligned image arrays and a gap mask:
                (aligned_base_image, shifted_reference_image, gap_mask).
    """
    # Convert images to numpy arrays
    base_image_array = np.array(base_image)
    reference_image_array = np.array(reference_image)

    # Calculate offsets
    offset = round(reference_image_distance - base_image_distance)
    horizontal_offset = 0
    vertical_offset = 0
    if is_vertical:
        vertical_offset = offset
    else:
        horizontal_offset = offset

    # Initialize blank canvas for the shifted reference image
    shifted_reference_array = np.zeros_like(base_image_array)

    # Calculate valid region to copy from the reference image
    ref_x_start = max(0, horizontal_offset)
    ref_y_start = max(0, vertical_offset)
    ref_x_end = min(
        base_image_array.shape[1], reference_image_array.shape[1] + horizontal_offset
    )
    ref_y_end = min(
        base_image_array.shape[0], reference_image_array.shape[0] + vertical_offset
    )

    # Calculate the corresponding region in the reference image
    ref_src_x_start = max(0, -horizontal_offset)
    ref_src_y_start = max(0, -vertical_offset)

    # Copy data from the reference image to the shifted canvas
    shifted_reference_array[ref_y_start:ref_y_end, ref_x_start:ref_x_end] = (
        reference_image_array[
            ref_src_y_start : ref_y_end - ref_y_start + ref_src_y_start,
            ref_src_x_start : ref_x_end - ref_x_start + ref_src_x_start,
        ]
    )

    # Create a gap mask if needed
    gap_mask = None
    if return_gap_mask:
        gap_mask = np.zeros_like(base_image_array, dtype=bool)
        if is_vertical:
            if vertical_offset > 0:
                gap_mask[:vertical_offset, :] = True
            elif vertical_offset < 0:
                gap_mask[vertical_offset:, :] = True
        else:
            if horizontal_offset > 0:
                gap_mask[:, :horizontal_offset] = True
            elif horizontal_offset < 0:
                gap_mask[:, horizontal_offset:] = True

    if return_gap_mask:
        return base_image_array, shifted_reference_array, gap_mask
    else:
        return base_image_array, shifted_reference_array


def save_image_with_replace(image: Image.Image, output_file: Path):
    """
    Saves an image to the specified output file, overwriting it if it already exists.

    Args:
        image (Image.Image): The image object to save.
        output_file (Path): The path where the image will be saved.

    Raises:
        IOError: If the image cannot be saved to the specified path.
        OSError: If an existing file cannot be deleted.

    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        try:
            output_file.unlink()
            print(f"Deleted existing file: {output_file}")
        except OSError as e:
            raise OSError(f"Failed to delete the existing file: {output_file}") from e

    # Save the image to the output file
    try:
        image.save(output_file)
        print(f"Saved image to: {output_file}")
    except IOError as e:
        raise IOError(f"Failed to save the image to {output_file}") from e

def generate_reference_array(
    original_array: np.ndarray,
    mask_array: np.ndarray,
    is_vertical: bool = True,
    output_type: str = "mean",
) -> np.ndarray:
    """
    Generates a reference image by applying the given mask to the original image.
    The reference image is computed based on the mean or mode of the masked regions.

    Args:
        original_array (np.ndarray): The original image array from which reference values are extracted.
        mask_array (np.ndarray): A binary mask array where 255 represents the selected area and 0 represents the excluded area.
        is_vertical (bool): Indicates whether to process vertically (True) or horizontally (False).
        output_type (str): Specifies the type of output ("mean" or "mode"). Default is "mean".

    Returns:
        np.ndarray: A reference image where masked regions are replaced by the specified aggregation (mean or mode).
    """
    
    # Validate inputs
    if original_array.shape != mask_array.shape:
        raise ValueError("original_array and mask_array must have the same shape.")   
    if output_type not in {"mean", "mode"}:
        raise ValueError("Invalid output_type. Choose 'mean' or 'mode'.")
    
    mask = mask_array
    original = original_array
    result_image = original.copy()

    if is_vertical:
        for i in range(mask.shape[0]):  # Process each row
            row_mask = mask[i, :] == 0  # Identify excluded areas(anomaly) in the row 
            row_values = original[i, row_mask] if np.any(row_mask) else original[i, :]# if there is any anomaly, use row_mask to aviod involving anomaly pixels into the generation of reference pics.
            
            if output_type == "mean":
                value = np.mean(row_values)
            elif output_type == "mode":
                value = mode(row_values, axis=None).mode[0]
            else:
                raise ValueError("Invalid output_type. Choose 'mean' or 'mode'.")
            result_image[i, :] = value

    else:
        for j in range(mask.shape[1]):  # Process each column
            col_mask = mask[:, j] == 0  # Identify excluded areas in the column
            col_values = original[col_mask, j] if np.any(col_mask) else original[:, j]
            if output_type == "mean":
                value = np.mean(col_values)
            elif output_type == "mode":
                value = mode(col_values, axis=None).mode[0]
            else:
                raise ValueError("Invalid output_type. Choose 'mean' or 'mode'.")
            result_image[:, j] = value

    return result_image


def apply_mask_to_image(
    image_array: np.ndarray, mask_array: np.ndarray, mask_value: int = 255
) -> np.ndarray:
    """
    Applies a given mask to an image, setting the masked area to a specific color.

    Args:
        image_array (np.ndarray): The image to apply the mask to, should be a 2D or 3D array.
        mask_array (np.ndarray): The 2-D binary mask (1 for masked areas, 0 for unmasked areas).
        mask_value (int): The threshold value to determine masked areas (default is 255, for white).

    Returns:
        np.ndarray: The image with the mask applied, where masked areas are set to 0 (black).
    """
    if image_array.ndim not in {2, 3}:
        raise ValueError("image_array must be a 2D (grayscale) or 3D (color) array.")
    if mask_array.ndim != 2:
        raise ValueError("mask_array must be a 2D array.")
    if image_array.shape[:2] != mask_array.shape:
        raise ValueError("The spatial dimensions of image_array and mask_array must match.")
    
    mask = mask_array >= mask_value
    if image_array.ndim == 3:  # For color images
        masked_array = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            masked_array[..., channel] = image_array[..., channel] * mask
    else:  # For grayscale images
        masked_array = image_array * mask

    return masked_array

def calculate_image_difference(
    base_image: np.ndarray, reference_image: np.ndarray
) -> np.ndarray:
    """
    Calculates the pixel-wise difference between two images (base - reference).
    The difference is computed in int16 to handle overflow/underflow issues.

    Args:
        base_image (numpy.ndarray): The base image
        reference_image (numpy.ndarray): The reference image

    Returns:
        numpy.ndarray: The difference image
    """
    # Ensure input images are numpy arrays
    base = np.array(base_image, dtype=np.int16)
    reference = np.array(reference_image, dtype=np.int16)

    # Compute the pixel-wise difference
    diff_image = base - reference

    return diff_image


def apply_difference_to_image(back_image_array: np.array, diff_image_array: np.array) -> np.array:
    """
    Applies a difference image to a background image and clamps the result
    to the valid uint8 range [0, 255].

    Args:
        back_image (numpy.ndarray): The background image
        diff_image (numpy.ndarray): The difference image

    Returns:
        numpy.ndarray
    """
    # Ensure input images are numpy arrays
    back = np.array(back_image_array, dtype=np.int16)
    diff = np.array(diff_image_array, dtype=np.int16)

    # Add the difference image to the background
    result = back + diff

    # Clamp values to the range [0, 255]
    result_clamped = np.clip(result, 0, 255)

    return result_clamped


def get_distance(get_file_name: str, csv_path: Path,column_name:str="data"):
    """
    Retrieves alignment distance and orientation from a CSV file based on the given file name.

    Args:
        get_file_name (str): The name of the file to search for in the alignment data.
        csv_path (Path): Path to the CSV file containing alignment data.

    Returns:
        Tuple[float, bool]: A tuple containing the distance (float) and orientation (bool, True for vertical).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the required columns are not found in the CSV file.
        IndexError: If the file name does not match any row in the CSV.
    """
    try:
        alignment_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")
    
    required_columns = ["File", "Area", column_name]
    for col in required_columns:
        if col not in alignment_data.columns:
            raise KeyError(f"Missing required column '{col}' in the CSV file.")
        
    if "C" not in get_file_name:
        get_file_name = get_file_name[:-2]

    alignment_data["File_Area"] = (
        alignment_data["File"].astype(str) + "_" + alignment_data["Area"].astype(str)
    )

    # try:
    #     row = alignment_data[alignment_data["File_Area"] == get_file_name].iloc[0]
    # except IndexError:
    #     print(f"No alignment data found for {get_file_name}. Skipping this folder.")
    #     raise

    # Attempt to match get_file_name
    filtered_data = alignment_data[alignment_data["File_Area"] == get_file_name]
    if filtered_data.empty:
        raise ValueError(
            f"No alignment data found for file name: {get_file_name}. "
            "Please check if the file name is correct and matches the data in the CSV."
        )

    # Retrieve the first matching row
    row = filtered_data.iloc[0]
    distance = row[column_name]
    return distance

def correct_distance_and_draw_line(distance, is_vertical, image_array):
    original_size_vertical = (328, 248)  
    original_size_horizontal = (248, 560)  
    target_size = (512, 512)  
    
    image_array = image_array.astype(np.uint8)
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    if is_vertical:
        scale_factor = target_size[0] / original_size_vertical[0]  # 512 / 328
        corrected_distance = distance * scale_factor
        cv2.line(image_array, (0, round(corrected_distance)), (target_size[0], round(corrected_distance)), (0, 255, 255), 2)
        
    else:
        scale_factor = target_size[1] / original_size_horizontal[1]  # 512 / 560
        corrected_distance = distance * scale_factor
        cv2.line(image_array, (target_size[0]-round(corrected_distance), 0), (target_size[0]-round(corrected_distance), target_size[1]), (0, 255, 255), 2)

    return corrected_distance, image_array

def from_int16_to_uint8(result_array):
    
    clipped_result = np.clip(result_array, 0, 255)
    uint8_result_array = clipped_result.astype(np.uint8)

    return uint8_result_array


def scale_image(image_array: np.ndarray, scale_factor: float, clip: bool = False) -> np.ndarray:
    """
    Scales the input image by a given factor. Optionally clips the values to the range [0, 255].

    Args:
        image_array (np.ndarray): The input image array (2D or 3D).
        scale_factor (float): The factor by which to scale the image.
        clip (bool): Whether to clip the scaled values to the range [0, 255].

    Returns:
        np.ndarray: The scaled image, optionally clipped, and of type int16.
    """
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array.")
    if image_array.ndim not in {2, 3}:
        raise ValueError("image_array must be a 2D (grayscale) or 3D (color) array.")
    if not isinstance(scale_factor, (int, float)):
        raise ValueError("scale_factor must be an int or float.")

    # Scale the image
    scaled_image = image_array * scale_factor

    # Optionally clip the values
    if clip:
        scaled_image = np.clip(scaled_image, 0, 255)

    # Return as int16
    return scaled_image.astype(np.int16)
