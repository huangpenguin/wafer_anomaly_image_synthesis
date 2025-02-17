"""アノテーション対象データの選択に関するモジュール。
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame


TARGET_SECTIONS = {
    # 'Ref_EBR1mm',  # 差しあたっては使わない
    '側面回込': ['C1'],
    'タイガー': ['A1'],
    'エッジ汚れ': ['A1', 'C1', 'E1'],
    'エッジ傷': ['C1'],
    'エッジ傷+α': ['A1', 'C1', 'E1'],
    'コメット': ['A1'],
    'ヒゲ': ['A1'],
    'EBR境界不良': ['A1'],
    '裏面回り込み': ['A1', 'C1', 'E1'],
    '裏面擦り傷+ウォーターマーク': ['E1'],
}
TARGET_MAKING_TYPES = list(TARGET_SECTIONS.keys())

NOTCH_FLAME_NUMS = ['0358', '0359', '0360', '0361']

MANUAL_SELECT_MAKING_TYPES = ['コメット', '裏面擦り傷+ウォーターマーク']
MANUAL_SELECT_FLAMES = {
    'コメット': {
        # Slot23
        'MEAS_18': [
            '0265', '0266', '0267', '0268', '0269',
            '0342', '0343', '0344', '0345', '0346',
            '0358', '0359', '0360', '0361', '0362',  # ノッチを含む
            '0363', '0364', '0365', '0366', '0367',
            '0372', '0373', '0374', '0375', '0376',
        ],
        # Slot24
        'MEAS_19': [
            '0112', '0113', '0114', '0115', '0116',
            '0128', '0129', '0130', '0131', '0132',
            # '0162', '0163', '0164', '0165', '0166',  # 数調整のため除外
            # '0264', '0265', '0266', '0267', '0268',  # 数調整のため除外
            # '0283', '0284', '0285', '0286', '0287',  # 数調整のため除外
            # '0288', '0289', '0290', '0291', '0292',  # 数調整のため除外
            '0301', '0302', '0303', '0304', '0305',
            '0310', '0311', '0312', '0313', '0314',
            '0324', '0325', '0326', '0327', '0328',
            '0358', '0359', '0360', '0361', '0362',  # ノッチを含む
            '0464', '0465', '0466', '0467', '0468',
        ],
        # Slot25
        'MEAS_20': [
            '0020', '0021', '0022', '0023', '0024',
            '0025', '0026', '0027', '0028', '0029',
            # '0034', '0035', '0036', '0037', '0038',  # 数調整のため除外
            # '0044', '0045', '0046', '0047', '0048',  # 数調整のため除外
            # '0071', '0072', '0073', '0074', '0075',  # 数調整のため除外
            # '0125', '0126', '0127', '0128', '0129',  # 数調整のため除外
            '0235', '0236', '0237', '0238', '0239',
            '0358', '0359', '0360', '0361', '0362',  # ノッチを含む
            '0377', '0378', '0379', '0380', '0381',
            '0382', '0383', '0384', '0385', '0386',
            '0411', '0412', '0413', '0414', '0415',
            '0475', '0476', '0477', '0478', '0479',
        ],
    },
    '裏面擦り傷+ウォーターマーク': {
        'MEAS_26': [
            '0000', '0001', '0002', '0003', '0004',
            '0005', '0006', '0007', '0008', '0009',
            '0116', '0117', '0118', '0119', '0120',  # ノッチを含む
            '0121', '0122', '0123', '0124', '0125',
            '0238', '0239', '0240', '0241', '0242',
            '0356', '0357', '0358', '0359', '0360',
            '0443', '0444', '0445', '0446', '0447',
            '0448', '0448', '0449', '0450', '0460',
            '0461', '0462', '0463', '0464', '0465',
            '0466', '0467', '0468', '0469', '0470',
            '0471', '0472', '0473', '0474', '0475',
            '0476', '0477', '0478', '0479',
        ],
    },
}


def select_annotation_targets(original_image_df: DataFrame, n_samples_per_frame: int) -> DataFrame:
    # アノテーション対象の欠陥種類を絞り込み
    original_image_df = original_image_df.query(f'making_defect_type in {TARGET_MAKING_TYPES}')

    # 欠陥種類別に繰り返す
    making_groups = original_image_df.groupby('making_defect_type')
    selected_dfs = []
    for making_defect_type, making_group in making_groups:
        # 発生箇所のbevel_sectionのみを選択
        selected_sections = TARGET_SECTIONS[making_defect_type]
        making_group = making_group.query(f'bevel_section in {selected_sections}').copy()

        # (測定,フレーム)ペアの識別子を一時的に追加
        making_group['meas_frame'] = making_group['measurement_id'] + '_' + making_group['flame_no']
        meas_frames = making_group['meas_frame'].unique()

        if making_defect_type not in MANUAL_SELECT_MAKING_TYPES:
            """480フレームからランダムサンプリング
            """
            # ノッチ付近のフレームは必ず選択する
            notch_num_rgx = '|'.join(NOTCH_FLAME_NUMS)
            pattern = rf'(.+?)_({notch_num_rgx})'
            near_notch_meas_frames = [m_f for m_f in meas_frames if re.match(pattern, m_f)]
            remain_meas_frames = [m_f for m_f in meas_frames if not re.match(pattern, m_f)]

            # アノテーション対象の(測定,フレーム)をランダムに選択
            n_remain_samples = n_samples_per_frame - len(near_notch_meas_frames)
            selected_meas_flames = np.random.choice(remain_meas_frames, size=n_remain_samples, replace=False).tolist()
            selected_meas_flames += near_notch_meas_frames
            selected_meas_flames = np.sort(selected_meas_flames).tolist()
        else:
            """手動で指定した欠陥を含むフレームを使う
            """
            selected_meas_flames = []
            for meas, flames in MANUAL_SELECT_FLAMES[making_defect_type].items():
                meas_frames_i = [f'{meas}_{flame}' for flame in flames]
                selected_meas_flames.extend(meas_frames_i)

        # (A1,C1,E1)ごとに繰り返す
        section_groups = making_group.groupby('bevel_section')
        for bevel_section, section_group in section_groups:
            section_group = section_group.query(f'meas_frame in {selected_meas_flames}')
            section_group = section_group.drop(columns=['meas_frame'])  # 後は不要

            # 側面の残膜は、'側面回込'と'裏面回り込み'から半分ずつ選ぶ
            if making_defect_type in ['側面回込', '裏面回り込み'] and bevel_section == 'C1':
                n_samples_in_section = int(n_samples_per_frame / 2)
                section_group = section_group.sample(n=n_samples_in_section, replace=False)
            selected_dfs.append(section_group)

    selected_df = pd.concat(selected_dfs, ignore_index=True)
    return selected_df


def select_extra_annotation_targets(
        original_image_df: DataFrame, exclude_image_df: DataFrame,
        target_making_types, n_samples_per_frame: int
) -> DataFrame:
    # アノテーション対象の欠陥種類を絞り込み
    original_image_df = original_image_df.query(f'making_defect_type in {target_making_types}')

    # 欠陥種類別に繰り返す
    making_groups = original_image_df.groupby('making_defect_type')
    selected_dfs = []
    for making_defect_type, making_group in making_groups:
        # 発生箇所のbevel_sectionのみを選択
        selected_sections = TARGET_SECTIONS[making_defect_type]
        making_group = making_group.query(f'bevel_section in {selected_sections}').copy()

        # (測定,フレーム)ペアの識別子を一時的に追加
        making_group['meas_frame'] = making_group['measurement_id'] + '_' + making_group['flame_no']
        meas_frames = making_group['meas_frame'].unique()  # A1,C1,E1の重複を除去

        # 除外する(測定,フレーム)ペアを作成
        exclude_group_df = exclude_image_df.query(f'making_defect_type=="{making_defect_type}"')
        exclude_meas_frame_s = exclude_group_df['measurement_id'] + '_' + exclude_group_df['flame_no']
        exclude_meas_frames = exclude_meas_frame_s.unique()  # A1,C1,E1の重複を除去

        # 選択済の(測定,フレーム)ペアを除外する
        meas_frames = [m_f for m_f in meas_frames if m_f not in exclude_meas_frames]
        # HACK: `select_annotation_targets`との違いはここだけ。select_annotation_targetsと統合する。

        if making_defect_type not in MANUAL_SELECT_MAKING_TYPES:
            """480フレームからランダムサンプリング
            """
            # ノッチ付近のフレームは必ず選択する
            notch_num_rgx = '|'.join(NOTCH_FLAME_NUMS)
            pattern = rf'(.+?)_({notch_num_rgx})'
            near_notch_meas_frames = [m_f for m_f in meas_frames if re.match(pattern, m_f)]
            remain_meas_frames = [m_f for m_f in meas_frames if not re.match(pattern, m_f)]

            # アノテーション対象の(測定,フレーム)をランダムに選択
            n_remain_samples = n_samples_per_frame - len(near_notch_meas_frames)
            selected_meas_flames = np.random.choice(remain_meas_frames, size=n_remain_samples, replace=False).tolist()
            selected_meas_flames += near_notch_meas_frames
            selected_meas_flames = np.sort(selected_meas_flames).tolist()
        else:
            """手動で指定した欠陥を含むフレームを使う
            """
            manual_select_meas_flames = []
            for meas, flames in MANUAL_SELECT_FLAMES[making_defect_type].items():
                meas_frames_i = [f'{meas}_{flame}' for flame in flames]
                manual_select_meas_flames.extend(meas_frames_i)
            selected_meas_flames = manual_select_meas_flames
            # HACK: こっちに変更したい
            # selected_meas_flames = [m_f for m_f in meas_frames if m_f in manual_select_meas_flames]

        # (A1,C1,E1)ごとに繰り返す
        section_groups = making_group.groupby('bevel_section')
        for bevel_section, section_group in section_groups:
            section_group = section_group.query(f'meas_frame in {selected_meas_flames}')
            section_group = section_group.drop(columns=['meas_frame'])  # 後は不要

            # 側面の残膜は、'側面回込'と'裏面回り込み'から半分ずつ選ぶ
            if making_defect_type in ['側面回込', '裏面回り込み'] and bevel_section == 'C1':
                n_samples_in_section = int(n_samples_per_frame / 2)
                section_group = section_group.sample(n=n_samples_in_section, replace=False)
            selected_dfs.append(section_group)

    selected_df = pd.concat(selected_dfs, ignore_index=True)
    return selected_df


def reconstruct_annotation_targets(
        original_image_df: DataFrame, annotation_dir: Path, selection_no: int,
) -> DataFrame:
    """既存のアノテーション前画像データから、画像選択情報を再構築する。
    """
    if not annotation_dir.exists():
        raise FileNotFoundError(f'{str(annotation_dir)} is not exist.')
    file_name_pattern = r'^(.+?)_([ACE])1_\d{4}(|_I|_O).tif$'
    file_list = [file for file in annotation_dir.glob('**/*') if re.search(file_name_pattern, str(file))]
    file_list = sorted(file_list)

    selected_image_ids = []
    for file in file_list:
        image_id = file.stem.rstrip('_I').rstrip('_O')

        selected_image_ids.append(image_id)

    # オリジナル画像テーブルに選択情報を追加
    def in_selected_image_ids(image_id_: str):
        return image_id_ in selected_image_ids
    in_flag = original_image_df['image_id'].apply(in_selected_image_ids)
    if 'selection_no' not in original_image_df.columns:
        original_image_df['selection_no'] = pd.NA
    original_image_df['selection_no'] = original_image_df['selection_no'].mask(in_flag, selection_no)

    return original_image_df
