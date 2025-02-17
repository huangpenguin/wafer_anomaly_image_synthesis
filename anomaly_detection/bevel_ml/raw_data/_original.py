"""元データ(SCREEN様から提供のフォルダ構造のまま)を扱うモジュール。
"""
import re
from pathlib import Path
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series


def _parse_wafer_info(wafer_folder_string: str) -> (str, str, str):
    datetime_rgx = r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'
    foup_rgx = r'FOUP_[A-Z]\d{2}'
    slot_rgx = r'Slot_?\d{2}'
    making_rgx = r'.*?'
    parameter_rgx = r'#\d{2}_.*?'

    pattern1 = f'^({datetime_rgx})_({foup_rgx}_{slot_rgx})_({making_rgx})_({parameter_rgx}$)'
    m1 = re.match(pattern1, wafer_folder_string)
    if m1:
        datetime = m1.group(1)
        foup_slot = m1.group(2)
        making_defect_type = m1.group(3)
    else:
        pattern2 = f'^({datetime_rgx})_({making_rgx})_({parameter_rgx}$)'
        m2 = re.match(pattern2, wafer_folder_string)
        if m2:
            datetime = m2.group(1)
            foup_slot = 'FOUP_UK_Slot_UK'
            making_defect_type = m2.group(2)
        else:
            raise RuntimeError('Not match!')

    return datetime, foup_slot, making_defect_type


def _parse_frame_info(file_name: str) -> (str, str):
    pattern = r'^([A-Z]\d)_(\d{4})\.tif$'
    m = re.match(pattern, file_name)
    if m:
        bevel_section = m.group(1)
        flame_no = m.group(2)
    else:
        raise RuntimeError('Not match!')

    return bevel_section, flame_no


def _get_measurement_map(datetime_ary: ndarray):
    unique_dts = np.unique(datetime_ary)
    unique_dts = np.sort(unique_dts)
    n_measurements = len(unique_dts)
    measurement_ids = [f'MEAS_{i:02}' for i in range(n_measurements)]
    measurement_map = dict(zip(unique_dts, measurement_ids))
    return measurement_map


def _get_image_ids(parsed_df: DataFrame, measurement_ids: Series) -> ndarray:
    image_ids = measurement_ids.copy()
    image_ids += '_' + parsed_df['foup_slot']
    image_ids += '_' + parsed_df['bevel_section']
    image_ids += '_' + parsed_df['flame_no']
    return image_ids.values


def find_images(image_dir: Path) -> DataFrame:
    if not image_dir.exists():
        raise FileNotFoundError(f'{str(image_dir)} is not exist.')
    file_list = [file for file in image_dir.glob('**/*') if re.search(r'([ACE])1_\d{4}.tif', str(file))]
    file_list = sorted(file_list)

    file_info_list = []
    for file in file_list:
        file_path = str(file)

        # ウエハ情報を抽出
        wafer_folder_string = file.parts[4]  # HACK: 間に合わせの仮実装になっている。フォルダ階層を自動検出することが望ましい。
        datetime, foup_slot, making_defect_type = _parse_wafer_info(wafer_folder_string)

        # フレーム情報を抽出
        file_name = file.parts[6]  # HACK: 間に合わせの仮実装になっている。フォルダ階層を自動検出することが望ましい。
        bevel_section, flame_no = _parse_frame_info(file_name)

        file_info = (datetime, foup_slot, bevel_section, flame_no, making_defect_type, file_path)
        file_info_list.append(file_info)

    columns = ['datetime', 'foup_slot', 'bevel_section', 'flame_no', 'making_defect_type', 'file_path']
    parsed_df = DataFrame(file_info_list, columns=columns)
    return parsed_df


def convert_to_er(parsed_df: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """関係DB形式のテーブルに変換する。
    """
    datetime_ary = parsed_df['datetime'].values
    measurement_map = _get_measurement_map(datetime_ary)
    measurement_ids = parsed_df['datetime'].replace(measurement_map)
    image_ids = _get_image_ids(parsed_df, measurement_ids)

    measurement_df = DataFrame({
        'measurement_id': measurement_map.values(),
        'datetime': measurement_map.keys(),
    })
    original_image_df = DataFrame({
        'image_id': image_ids,
        'measurement_id': measurement_ids,
        'foup_slot': parsed_df['foup_slot'],
        'bevel_section': parsed_df['bevel_section'],
        'flame_no': parsed_df['flame_no'],
        'making_defect_type': parsed_df['making_defect_type'],
        'file_path': parsed_df['file_path'],
    })
    original_directory_df = original_image_df[[
        'measurement_id',
        'foup_slot',
        'making_defect_type',
        'file_path',
    ]]
    original_directory_df = original_directory_df.drop_duplicates(subset=['measurement_id', 'foup_slot'])
    original_directory_df['path'] = original_directory_df['file_path'].replace(r'\\Raw_data.*$', '', regex=True)
    original_directory_df = original_directory_df.drop(columns=['file_path'])

    return measurement_df, original_directory_df, original_image_df
