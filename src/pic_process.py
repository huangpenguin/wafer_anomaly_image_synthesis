from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Tuple
from scipy.stats import mode
import pandas as pd 

def align_images(
    base_image: Image.Image,
    base_image_distance: float,
    reference_image: Image.Image,
    reference_image_distance: float,
    is_vertical: bool = True,
    return_gap_mask: bool = False
) -> tuple:
    """
    Aligns the reference image with the base image based on specified distances.

    This function shifts the reference image either vertically or horizontally to align it with the base image.

    Args:
        base_image (Image.Image): The base image used as a reference.
        base_image_distance (float): Distance from a specific edge of the base image.
        reference_image (Image.Image): The image to be aligned.
        reference_image_distance (float): Distance from the same edge of the reference image.
        is_vertical (bool): If True, aligns vertically; if False, horizontally.
        return_gap_mask (bool): If True, returns a mask indicating gap areas.

    Returns:
        tuple: 
            - If `return_gap_mask` is False: Pair of aligned images
            - If `return_gap_mask` is True:  Pair of aligned images and gap_mask between two pics)
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
    ref_x_end = min(base_image_array.shape[1], reference_image_array.shape[1] + horizontal_offset)
    ref_y_end = min(base_image_array.shape[0], reference_image_array.shape[0] + vertical_offset)

    # Calculate the corresponding region in the reference image
    ref_src_x_start = max(0, -horizontal_offset)
    ref_src_y_start = max(0, -vertical_offset)

    # Copy data from the reference image to the shifted canvas
    shifted_reference_array[ref_y_start:ref_y_end, ref_x_start:ref_x_end] = \
        reference_image_array[ref_src_y_start:ref_y_end-ref_y_start+ref_src_y_start,
                              ref_src_x_start:ref_x_end-ref_x_start+ref_src_x_start]

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

def split_images_in_folder(folder_path: Path):
    """
    Split all images in the folder into two halves and append 'I' and 'O' to the original file names.

    :param folder_path: Path to the folder containing images.
    """
    if not folder_path.is_dir():
        print(f"{folder_path} is not a valid folder path.")
        return

    for image_file in folder_path.glob('*'):
        if "C" not in image_file.stem and "O" not in image_file.stem:
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif']:           
                with Image.open(image_file) as img:
                    width, height = img.size
                    
                    left_image = img.crop((0, 0, width // 2, height))
                    right_image = img.crop((width // 2, 0, width, height))

                    left_image_file = image_file.with_stem(image_file.stem + '_I')
                    right_image_file = image_file.with_stem(image_file.stem + '_O')

                    left_image.save(left_image_file)
                    right_image.save(right_image_file)

                    print(f"Saved: {left_image_file} and {right_image_file}")

def resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Resize an input image to the specified size.
    #resized_img=transform.Resize((512, 512))(img)

    Args:
        img (Image.Image): A PIL Image object to be resized.
        size (Tuple[int, int]): The target size for resizing, specified as (width, height).

    Returns:
        Image.Image: A resized PIL Image object.

    Raises:
        ValueError: If the input is not a valid PIL Image or size is not a tuple of two integers.
        RuntimeError: If the resizing process encounters an error.
    """
    # Validate inputs
    if not isinstance(img, Image.Image):
        raise ValueError("The input 'img' must be a PIL Image object.")
    if not (isinstance(size, tuple) and len(size) == 2 and all(isinstance(dim, int) for dim in size)):
        raise ValueError("The 'size' must be a tuple of two integers, e.g., (width, height).")
    
    # Define the transform for resizing
    transform = transforms.Resize(size)

    try:
        # Apply the resize transformation
        resized_img = transform(img)
        print("Resized image successfully.")
    except Exception as e:
        print(f"Error resizing image: {e}")
        raise RuntimeError("Failed to resize the image.") from e

    return resized_img

def save_image_with_overwrite(image: Image.Image, output_file: Path):
    """
    Saves the given image to the specified output file.
    If the file already exists, it deletes the existing file before saving.

    :param diff_image: The image object to save.
    :param output_file: The path of the output file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    #display(image)
    
    if output_file.exists():
        output_file.unlink() 
        print(f"Deleted existing file: {output_file}")

    image.save(output_file) 
    print(f"Saved image to: {output_file}")
    print(f"####################################################################")

def generate_reference_images(original_image:Image.Image,mask_image:Image.Image,mask_image_name:str)->Tuple[np.array,np.array]:
    """
    Generates reference images by applying the given mask to the original image. 
    The reference images are computed based on the mean and mode of the masked regions.

    Args:
        original_image (image): The original image from which reference values are extracted.
        mask_image (image): A binary mask image where 255 represents the selected area and 0 represents the excluded area.
        mask_image_name (str): A string identifier that determines the processing approach 
                               (e.g., "C" for horizontal processing, "A"/"E" for vertical processing).

    Returns:
        tuple: A tuple containing two reference images:
            - The first image with masked regions replaced by the mean of the corresponding rows or columns.
            - The second image with masked regions replaced by the mode of the corresponding rows or columns.
    """
    mask = np.array(mask_image)
    original = np.array(original_image)

    result_image_mean = original.copy()
    result_image_mode = original.copy()

    if "C" in mask_image_name:  # Horizontal processing
        for i in range(mask.shape[0]):  # Process each row
            row_mask = mask[i, :] == 0  # Identify excluded areas in the row
            row_values = original[i, row_mask] if np.any(row_mask) else original[i, :]
            mean_value = np.mean(row_values)
            mode_value = mode(row_values, axis=None).mode
            result_image_mean[i, :] = mean_value
            result_image_mode[i, :] = mode_value

    if "A" in mask_image_name or "E" in mask_image_name:  # Vertical processing
        for j in range(mask.shape[1]):  # Process each column
            col_mask = mask[:, j] == 0  # Identify excluded areas in the column
            col_values = original[col_mask, j] if np.any(col_mask) else original[:, j]
            mean_value = np.mean(col_values)
            mode_value = mode(col_values, axis=None).mode
            result_image_mean[:, j] = mean_value
            result_image_mode[:, j] = mode_value

    return result_image_mean, result_image_mode

def apply_mask_to_image(image_array: np.ndarray, mask_array: np.ndarray, mask_value: int = 255) -> np.ndarray:
    """
    Applies a given mask to an image, setting the masked area to a specific color.

    Args:
        image_array (np.ndarray): The image to apply the mask to, should be a 2D or 3D array.
        mask_array (np.ndarray): The 2-D binary mask (1 for masked areas, 0 for unmasked areas).
        mask_value (int): The threshold value to determine masked areas (default is 255, for white).

    Returns:
        np.ndarray: The image with the mask applied, where masked areas are set to 0 (black).
    """
    # Create a copy of the original image array
    masked_array = np.copy(image_array)

    # Create a boolean mask based on the threshold value
    mask = mask_array >= mask_value

    # Set unmasked areas to 0 (black)
    masked_array[~mask] = 0 

    return masked_array

def calculate_image_difference(base_image: np.array, reference_image: np.array)->np.array:
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

def apply_difference_to_image(back_image:np.array, diff_image:np.array)-> np.array:
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
    back = np.array(back_image, dtype=np.int16)
    diff = np.array(diff_image, dtype=np.int16)

    # Add the difference image to the background
    result = back + diff

    # Clamp values to the range [0, 255]
    result_clamped = np.clip(result, 0, 255)
    
    return result_clamped

def get_distance(get_file_name,csv_path):
    """
    todo
    """
    alignment_data = pd.read_csv(csv_path)
    
    if "C" not in get_file_name:#TODO
        get_file_name = get_file_name[:-2]

    alignment_data["File_Area"] = alignment_data["File"].astype(str) + "_" + alignment_data["Area"].astype(str)
    
    try:
        row = alignment_data[alignment_data["File_Area"] == get_file_name].iloc[0]
    except IndexError:
        print(f"No alignment data found for {get_file_name}. Skipping this folder.")
        
    area = row["Area"]
    distance = row["Data"]
    is_vertical = area.startswith("C")
    
    return distance
    
def from_int16_to_uint8(result_array):
    
    clipped_result = np.clip(result_array, 0, 255)
    uint8_result_array = clipped_result.astype(np.uint8)
    
    return uint8_result_array

def scale_image(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scales the input image by a given factor and clips the values to the range [0, 255].

    Args:
        image_array (np.ndarray): The input image array (2D or 3D).
        scale_factor (float): The factor by which to scale the image.

    Returns:
        np.ndarray: The scaled image, clipped to the range [0, 255] and of type uint8.
    """
    # Scale the image
    scaled_image = image_array * scale_factor
    
    # Clip the values to the range [0, 255]
    #clipped_image = np.clip(scaled_image, 0, 255)
    
    # Convert to uint8
    return scaled_image.astype(np.int16)