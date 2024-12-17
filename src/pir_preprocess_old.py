from pathlib import Path
from PIL import Image
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from typing import Tuple

# def split_images_in_folder(input_folder: Path,output_folder:Path):
#     """
#     Splits all images in a folder into two halves and saves them with '_I' and '_O' appended to the original file names.

#     Args:
#         folder_path (Path): Path to the folder containing images to be split.

#     Returns:
#         None

#     Raises:
#         ValueError: If the provided path is not a valid folder.

#     """
#     if not input_folder.is_dir():
#         print(f"{input_folder} is not a valid folder path.")
#         return

#     for image_file in input_folder.glob('*'):
#         if ("C" not in image_file.stem) and ("O" not in image_file.stem) and ("I" not in image_file.stem):
#             if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif']:
#                 with Image.open(image_file) as img:
#                     width, height = img.size

#                     left_image = img.crop((0, 0, width // 2, height))
#                     right_image = img.crop((width // 2, 0, width, height))

#                     left_image_path = image_file.with_stem(image_file.stem + '_I')
#                     right_image_path = image_file.with_stem(image_file.stem + '_O')

#                     left_image.save(left_image_path)
#                     right_image.save(right_image_path)

def split_images_in_folder(input_folder: Path, output_folder: Path):
    """
    Splits all images in a folder into two halves and saves them with '_I' and '_O' appended to the original file names.
    Copies files containing 'C' in their names to the output folder without splitting.

    Args:
        input_folder (Path): Path to the folder containing images to be processed.
        output_folder (Path): Path to the folder where processed images will be saved.

    Returns:
        None

    Raises:
        ValueError: If the provided input folder path is not valid.
    """
    if not input_folder.is_dir():
        raise ValueError(f"{input_folder} is not a valid folder path.")

    # Ensure the output folder exists
    split_folder = output_folder / "splitted"
    split_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob('*'):
        if image_file.is_file():
            if ("C" not in image_file.stem) and ("O" not in image_file.stem) and ("I" not in image_file.stem):
                split_image(image_file, split_folder)
            elif "C" in image_file.stem:
                # Copy the file directly
                shutil.copy(image_file, split_folder)
                print(f"Copied file: {image_file} to {split_folder}")








def split_image(image_path: Path, output_folder: Path) -> None:
    """
    Splits a single image into two halves and saves them with '_I' and '_O' appended to the original file name.

    Args:
        image_path (Path): Path to the input image to be split.
        output_folder (Path): Path to the folder where split images will be saved.

    Returns:
        None
    """
    if not image_path.is_file() or image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif']:
        print(f"Invalid image file: {image_path}")
        return

    with Image.open(image_path) as img:
        width, height = img.size

        # Split the image into two halves
        left_image = img.crop((0, 0, width // 2, height))
        right_image = img.crop((width // 2, 0, width, height))

        # Ensure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)

        # Prepare output file paths
        left_image_path = output_folder / f"{image_path.stem}_I{image_path.suffix}"
        right_image_path = output_folder / f"{image_path.stem}_O{image_path.suffix}"

        # Save split images
        left_image.save(left_image_path)
        right_image.save(right_image_path)

        print(f"Saved: {left_image_path} and {right_image_path}")

        return left_image,right_image

def make_anomaly_masks_(
    anomaly_scores_path: Path,
    dest_path: Path,
    thresholds: list[float],
):
    """
    Generate binary anomaly region images from anomaly scores using thresholds 
    and save them with index-based file names.

    Args:
        anomaly_scores_path (Path): Path to the CSV file containing anomaly scores.
        dest_path (Path): Directory to save the output images.
        thresholds (list[float]): List of thresholds for binary mask generation.

    Returns:
        None
    """
    # Load anomaly scores
    anomaly_scores_df = pd.read_csv(str(anomaly_scores_path), header=None, index_col=0)
    
    # Ensure output directory exists
    #dest_path.mkdir(parents=True, exist_ok=True)
    
    # Reshape the scores
    anomaly_scores = anomaly_scores_df.values.reshape(len(anomaly_scores_df), 1, 512, 512)
    
    # Iterate through thresholds
    for threshold in thresholds:
        threshold_dir = dest_path / f"threshold_{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate binary anomaly regions
        for idx, anomaly_map in zip(anomaly_scores_df.index, anomaly_scores):
            binary_anomaly_region = (anomaly_map[0] > threshold).astype(np.uint8) * 255
            
            # Apply morphology operations
            kernel = np.ones((5, 5), np.uint8)
            binary_anomaly_region = cv2.morphologyEx(binary_anomaly_region, cv2.MORPH_OPEN, kernel)
            binary_anomaly_region = cv2.morphologyEx(binary_anomaly_region, cv2.MORPH_CLOSE, kernel)
            
            # Save the binary mask image
            output_path = threshold_dir / f"{idx}.png"
            cv2.imwrite(str(output_path), binary_anomaly_region)


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
