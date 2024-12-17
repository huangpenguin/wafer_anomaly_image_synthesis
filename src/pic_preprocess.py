from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


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

    split_folder = output_folder / "splitted"
    split_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob("*.tif"):
        if image_file.is_file():
            with Image.open(image_file) as img:
                if (
                    ("A" in image_file.stem or "E" in image_file.stem)
                    and ("O" not in image_file.stem)
                    and ("I" not in image_file.stem)
                ):
                    split_image(
                        img,
                        image_name=image_file.stem,
                        save=True,
                        output_folder=split_folder,
                    )
                elif "C" in image_file.stem:
                    save_image(
                        img,
                        output_path=split_folder / image_file.name,
                    )


def split_image(
    image: Image.Image, image_name: str, save: bool = True, output_folder: Path = None
) -> Tuple[Image.Image, Image.Image]:
    """
    Splits an image into two halves.

    Args:
        image (Image.Image): The image to split.
        image_name: Image name without suffix.
        save (bool): Whether to save the split images or return them directly.
        output_folder (Path): Directory to save the split images if `save` is True.

    Returns:
        Tuple[Image.Image, Image.Image]: Left and right halves of the split image (if `save` is False).
    """
    width, height = image.size

    left_image = image.crop((0, 0, width // 2, height))
    right_image = image.crop((width // 2, 0, width, height))

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    if save:  # TODO test
        left_image_path = output_folder / f"{image_name}_I.tif"
        right_image_path = output_folder / f"{image_name}_O.tif"
        save_image(left_image, left_image_path)
        save_image(right_image, right_image_path)
    elif not save:
        return left_image, right_image


def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save a PIL Image object to the specified output path.

    Args:
        image (Image.Image): The image to save.
        output_path (Path): Path where the image will be saved.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")


def make_anomaly_masks_in_folder(
    anomaly_scores_df: pd.DataFrame,
    thresholds: list[float],
    output_folder: Path = None,
):
    """
    Generate binary anomaly masks from a DataFrame of anomaly scores for specified thresholds
    and save the results as images in the output folder.

    Args:
        anomaly_scores_df (pd.DataFrame): A DataFrame where each row corresponds to a flattened anomaly map (512x512).
        The index of the DataFrame is used as the file name for saving masks.
        thresholds (list[float]): A list of threshold values to apply for generating binary masks.
        output_folder (Path): The root directory where the binary masks will be saved.
                              Subdirectories will be created for each threshold.

    Returns:
        None

    """
    anomaly_scores = anomaly_scores_df.values.reshape(
        len(anomaly_scores_df), 1, 512, 512
    )
    # Iterate through thresholds
    for threshold in thresholds:
        threshold_dir = output_folder / f"threshold_{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)

        # Generate binary anomaly regions
        for idx, anomaly_map in zip(anomaly_scores_df.index, anomaly_scores):
            binary_anomaly_region = (anomaly_map[0] > threshold).astype(np.uint8) * 255
            # Apply morphology operations
            kernel = np.ones((5, 5), np.uint8)
            binary_anomaly_region = cv2.morphologyEx(
                binary_anomaly_region, cv2.MORPH_OPEN, kernel
            )
            binary_anomaly_region = cv2.morphologyEx(
                binary_anomaly_region, cv2.MORPH_CLOSE, kernel
            )

            # Save the binary mask image
            output_path = threshold_dir / f"{idx}.tif"
            cv2.imwrite(str(output_path), binary_anomaly_region)

def get_best_threshold(anomaly_scores_df: pd.DataFrame, file_stem: str, is_vertical: bool) -> float:
    """
    Computes the best threshold value from an anomaly map.

    The function retrieves the anomaly map corresponding to the given `file_stem` from the 
    anomaly scores DataFrame, reshapes it into a 512x512 matrix, and computes the best threshold:
    - If `is_vertical` is True, it finds the maximum value among the minimum values of each row.
    - If `is_vertical` is False, it finds the maximum value among the minimum values of each column.

    Args:
        anomaly_scores_df (pd.DataFrame): A DataFrame containing anomaly scores indexed by file_stem.
        file_stem (str): The key to locate the anomaly map within the DataFrame.
        is_vertical (bool): Determines whether to compute the threshold row-wise (True) or column-wise (False).

    Returns:
        float: The computed best threshold value.
    """
    try:
        # Retrieve and reshape the anomaly map
        anomaly_map = anomaly_scores_df.loc[file_stem].values.reshape(512, 512)
    except KeyError:
        raise KeyError(f"File stem '{file_stem}' not found in DataFrame.")
    except ValueError:
        raise ValueError(f"Data for file stem '{file_stem}' cannot be reshaped to (512, 512).")

    if is_vertical:
        # Row-wise: Find the maximum of the minimum values of each row
        threshold = np.max(np.min(anomaly_map, axis=1))
    else:
        # Column-wise: Find the maximum of the minimum values of each column
        threshold = np.max(np.min(anomaly_map, axis=0))
    threshold=np.ceil(threshold*100)/100 
    return threshold #TODO accuracy problem 0.150000000002

def make_anomaly_mask_by_name(
    anomaly_scores_df: pd.DataFrame,
    file_stem:str,
    threshold: float,
    output_folder: Path = None,
):
    """
TODO

    """
    #anomaly_scores = anomaly_scores_df.values.reshape(len(anomaly_scores_df), 1, 512, 512  )

    threshold_dir = output_folder / f"threshold_{threshold}"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    # Generate binary anomaly regions
    anomaly_map=anomaly_scores_df.loc[file_stem].values.reshape(1,512,512)
    binary_anomaly_region = (anomaly_map[0] > threshold).astype(np.uint8) * 255
    # Apply morphology operations
    kernel = np.ones((5, 5), np.uint8)
    binary_anomaly_region = cv2.morphologyEx(
        binary_anomaly_region, cv2.MORPH_OPEN, kernel
    )
    binary_anomaly_region = cv2.morphologyEx(
        binary_anomaly_region, cv2.MORPH_CLOSE, kernel
    )

    # Save the binary mask image
    output_path = threshold_dir / f"{file_stem}.tif"
    cv2.imwrite(str(output_path), binary_anomaly_region)
            
def resize_image(
    img: Image.Image, size: Tuple[int, int], save: bool = True, output_path: Path = None
) -> Image.Image:
    """
    Resize an input image to the specified size.
    #resized_img=transform.Resize((512, 512))(img)

    Args:
        img (Image.Image): A PIL Image object to be resized.
        size (Tuple[int, int]): The target size for resizing, specified as (width, height).
        save (bool): Whether to save the resized image or return it directly.
        output_path (Path): Path to save the resized image if `save` is True.

    Returns:
        Image.Image: The resized image (if `save` is False).

    Raises:
        ValueError: If the input is not a valid PIL Image or size is not a tuple of two integers.
        RuntimeError: If the resizing process encounters an error.
    """
    # Validate inputs
    if not isinstance(img, Image.Image):
        raise ValueError("The input 'img' must be a PIL Image object.")
    if not (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(dim, int) for dim in size)
    ):
        raise ValueError(
            "The 'size' must be a tuple of two integers, e.g., (width, height)."
        )

    # Define the transform for resizing
    transform = transforms.Resize(size)

    try:
        # Apply the resize transformation
        resized_img = transform(img)
        print("Resized image successfully.")
    except Exception as e:
        print(f"Error resizing image: {e}")
        raise RuntimeError("Failed to resize the image.") from e

    if save and output_path:
        save_image(resized_img, output_path)
    elif not save:
        return resized_img
