"""分割データ(モデルへの入力用)を扱うモジュール。
"""
import re
import warnings
from pathlib import Path
import numpy as np
from numpy import ndarray
import skimage.io
import skimage.util
from pandas import DataFrame, Series

SPLIT_CAPTIONS = ['I', 'O']
# DIR_CLASS_MAP = {
#     # 旧クラス番号(2024/03/28以前のアノテーションデータで使用)
#     '01_汚れ': '01_blot',
#     '02_擦り跡': '02_scratch',
#     '03_発塵痕': '03_dust',
#     '04_打痕、傷': '04_dent',
#     '05_異物': '05_foreign_substance',
#     '06_ウォーターマーク': '06_watermark',
#     '07_残膜': '07_residue',
#     '08_ヒゲ': '08_barb',
#     '09_膜境界不良': '09_defective_boundary',
#     '10_コメット': '10_comet',
#     '11_タイガーストライプ': '11_tiger_stripe',
#     '12_正常': '12_normal',
#     '13_ぐにゃあ': '13_gunyaa',
#     '50_分からない': '50_unknown',
# }
DIR_CLASS_MAP = {
    # 新クラス番号(2024/05/20以降のアノテーションデータで使用)
    '01_正常': '01_normal',
    '02_汚れ': '02_blot',
    '03_擦り跡': '03_scratch',
    '04_発塵痕': '04_dust',
    '05_打痕、傷': '05_dent',
    '06_異物': '06_foreign_substance',
    '07_ウォーターマーク': '07_watermark',
    '08_残膜': '08_residue',
    '09_ヒゲ': '09_barb',
    '10_膜境界不良': '10_defective_boundary',
    '11_コメット': '11_comet',
    '12_タイガーストライプ': '12_tiger_stripe',
    '13_ぐにゃあ': '13_gunyaa',
    '50_分からない': '50_unknown',
}
CLASS_DIR_NAMES = list(DIR_CLASS_MAP.keys())
CLASSES = list(DIR_CLASS_MAP.values())


def split_save_images(selected_image_df: DataFrame, output_root: Path | None, output: bool = True) -> DataFrame:
    """selected_image_dfで指定した画像を分割して保存する。

    Args:
        selected_image_df:
        output_root:
        output: 分割後の画像ファイルを出力するかどうかのフラグ
    Returns:
        split_df: 分割後データの情報を格納したDataFrame。
    """

    def save_image(img: ndarray, new_row_: Series, output_dir_):
        if output:
            output_subdir = output_dir_ / f'{new_row_['bevel_section']}_{new_row_['split']}'
            file_name = f'{new_row_["image_id"]}.tif'
            output_file = output_subdir / file_name
            print(f'Saving image... output_file={output_file}')
            output_subdir.mkdir(parents=True, exist_ok=True)
            skimage.io.imsave(output_file, img)

    split_rows = []
    for i, row in selected_image_df.iterrows():
        input_file = row['file_path']
        print(f'Reading image... input_file={input_file}')
        input_img = skimage.io.imread(input_file)
        output_dir = output_root / f'{row['making_defect_type']}' if output else None

        if row['bevel_section'] in ['A1', 'E1']:
            # 画像を2分割する
            blocks = np.hsplit(input_img, 2)
            for split_no, block_img in enumerate(blocks):
                new_row = row.copy()
                new_row = new_row.drop('file_path')
                new_row['split'] = SPLIT_CAPTIONS[split_no]
                new_row['original_id'] = new_row['image_id']
                new_row['image_id'] = new_row['image_id'] + '_' + new_row['split']
                split_rows.append(new_row)

                # ファイルに保存
                save_image(block_img, new_row, output_dir)
        else:
            # 画像を2分割しない
            new_row = row.copy()
            new_row = new_row.drop('file_path')
            new_row['original_id'] = new_row['image_id']
            new_row['split'] = None
            split_rows.append(new_row)

            # ファイルに保存
            save_image(input_img, new_row, output_dir)

    # 分割画像テーブルを作成
    columns = [
        'image_id', 'original_id', 'measurement_id',
        'foup_slot', 'bevel_section', 'flame_no', 'split',
        'making_defect_type', 'selection_no', 'defect_class',
    ]
    split_df = DataFrame(split_rows, columns=columns)
    return split_df


def make_template_class_dirs(parent_dir: Path) -> None:
    """分類先のフォルダを作成する。
    - フォルダ名の書式は、`{クラス番号}_{クラス名}`。
    """
    (parent_dir / f'00_未分類[作業用の一時保存]').mkdir(parents=True, exist_ok=True)
    for dir_name in CLASS_DIR_NAMES:
        (parent_dir / dir_name).mkdir(parents=True, exist_ok=True)


def extract_annotations(annotation_dir: Path) -> DataFrame:
    """アノテーション済フォルダから、アノテーション情報を抽出する。
    """
    if not annotation_dir.exists():
        raise FileNotFoundError(f'{str(annotation_dir)} is not exist.')
    file_name_pattern = r'^(.+?)_([ACE])1_\d{4}(|_I|_O).tif$'
    file_list = [file for file in annotation_dir.glob('**/*') if re.search(file_name_pattern, str(file))]
    file_list = sorted(file_list)

    annotation_rows = []
    for file in file_list:
        image_id = file.stem
        class_dir = file.parent.stem
        if class_dir not in CLASS_DIR_NAMES:
            if file.parent.parent.stem == '50_分からない':
                # '50_分からない'のサブフォルダに分類されているとき
                class_dir = '50_分からない'
            else:
                warnings.warn(f'invalid annotation! class_dir_name={class_dir}')
                continue

        class_label = DIR_CLASS_MAP[class_dir]

        annotation_row = (image_id, class_label)
        annotation_rows.append(annotation_row)

    columns = ['image_id', 'defect_class']
    annotation_df = DataFrame(annotation_rows, columns=columns)
    return annotation_df


def remove_mistake_files(mistake_dir: Path, mistake_image_ids: list[str]):
    """指定フォルダからアノテーション間違いファイルを削除する。
    """
    # 指定フォルダ内の画像ファイルをリストアップする
    if not mistake_dir.exists():
        raise FileNotFoundError(f'{str(mistake_dir)} is not exist.')
    file_name_pattern = r'^(.+?)_([ACE])1_\d{4}(|_I|_O).tif$'
    file_list = [file for file in mistake_dir.glob('**/*') if re.search(file_name_pattern, str(file))]
    file_list = sorted(file_list)

    # 間違いIDのファイルを削除する
    for file in file_list:
        image_id = file.stem
        if image_id in mistake_image_ids:
            file.unlink()
            print(f'{str(file)} is removed.')
