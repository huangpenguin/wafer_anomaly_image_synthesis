import pandas as pd
from pathlib import Path
import shutil
from dataclasses import dataclass

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

def generate_dataframe_from_image(file_name, defect_class, dataframe=None):
    # if not file_name.endswith(".tif"):
    #     raise ValueError("Input file must be a .tif file")
    # file_base = os.path.splitext(file_name)[0]

    if dataframe is None:
        raise ValueError("No dataframe to save data")
    parts = file_name.split("_")
    if len(parts) < 5:
        raise ValueError("File name format is invalid, unable to parse required fields")

    image_id = file_name
    original_id = "_".join(parts[:-1]) if len(parts) > 5 else file_name
    measurement_id = "_".join(parts[:3])
    bevel_section = parts[3]

    try:
        flame_no = int(parts[4])
    except ValueError:
        flame_no = parts[4]

    split = parts[5] if len(parts) > 5 else ""

    new_row = {
        "image_id": image_id,
        "original_id": original_id,
        "measurement_id": measurement_id,
        "foup_slot": "",  
        "bevel_section": bevel_section,
        "flame_no": flame_no,
        "split": split,
        "making_defect_type": "",  
        "selection_no": "", 
        "defect_class": defect_class
    }

    dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)
    return dataframe
                    
def extract_parts(measurement_id):
    # Split the measurement_id based on underscores
    parts = measurement_id.split('_')
    
    # Extract the relevant parts (200, 01, and A1_0000)
    part1 = parts[0]+parts[1]  # '200'
    part2 = parts[2]  # '01'
    part3 = parts[3] + '_' + parts[4]  # 'A1_0000'
    
    return part1, part2, part3


def find_matching_folder(base_path, folder_name):
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(folder_name):  # Match folder that starts with folder_name
            return folder
    return None

def find_and_copy_images(src_dir, dest_dir, id):

    part1, part2, part3 = extract_parts(id)
    
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)   
    if not dest_path.exists():
        dest_path.mkdir(parents=True) 
    first_folder = find_matching_folder(src_path, part1)   
    if first_folder:
        second_folder = find_matching_folder(first_folder, part2)
        if second_folder:
            raw_folder = second_folder / 'Raw'
            if raw_folder.is_dir():
                tif_file = raw_folder / (part3 + '.tif')
                if tif_file.is_file():
                    destination_file = dest_path / ( id+ '.tif')
                    shutil.copy(tif_file, destination_file)
                    print(f"Copied {tif_file.name} to {destination_file}")
                else:
                    print(f"{tif_file.name} not found in {raw_folder}")
            else:
                print(f"Raw folder not found in {second_folder}")
        else:
            print(f"Second folder matching {part2} not found")
    else:
        print(f"First folder matching {part1} not found")
             
@dataclass
class ImageInfo:
    image_id: str
    original_id: str
    measurement_id: str
    bevel_section: str
    flame_no: str
    split: str

def process_image_name(image, is_vertical=False) -> ImageInfo:
    if not is_vertical:           
        image_id = image
        original_id = image.rsplit('_', 1)[0]
        measurement_id = image.rsplit('_', 3)[0]
        bevel_section = image.split('_')[3]
        flame_no = image.split('_')[4]
        split = image.rsplit('_', 1)[1]
    else:
        image_id = image
        original_id = image
        measurement_id = image.rsplit('_', 2)[0]
        bevel_section = image.split('_')[3]
        flame_no = image.split('_')[4]
        split = ""

    return ImageInfo(image_id, original_id, measurement_id, bevel_section, flame_no, split)

def filename_analy(base_image_file,back_image_file):
    column_name_exception=['1_05','1_06','1_07','1_08','1_09','1_10','1_11','1_12','1_13','1_14','1_15']
    scale_factor_exception=['1_08','1_09','1_10','1_11','1_12','1_13','1_14','1_15']
    base_info = process_image_name(base_image_file)
    back_info = process_image_name(back_image_file)
    if base_info.bevel_section=="A1" and base_info.measurement_id.rsplit('_', 1)[0] in column_name_exception:
        column_name="Dis_cut"
    else:
        column_name="Dis_edge"
    if base_info.bevel_section=="A1" and base_info.measurement_id.rsplit('_', 1)[0] in scale_factor_exception:
        scale_factor=(1,-1)
    else:
        scale_factor=(1,)

    return column_name,scale_factor