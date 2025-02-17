from pathlib import Path
import pandas as pd
from pandas import DataFrame

from bevel_ml import db
from bevel_ml import raw_data
import _context as ctx


def create_summary_df(split_image_df: DataFrame, summary_path: Path):
    split_image_df['split'] = split_image_df['split'].fillna('NA')
    summary_df = split_image_df.groupby(['defect_class', 'bevel_section', 'split'], as_index=False).size()
    summary_df['defect_class'] = summary_df['defect_class'].replace(raw_data.CLASSES, raw_data.CLASS_DIR_NAMES)
    summary_df['section_split'] = summary_df['bevel_section'] + '_' + summary_df['split']
    summary_df['section_split'] = summary_df['section_split'].str.replace('_NA', '')
    summary_pivot_df = summary_df.pivot(index='defect_class', columns='section_split', values='size')
    summary_pivot_df = summary_pivot_df.fillna(0)
    summary_pivot_df = summary_pivot_df.astype(int)
    print('------------------------------------------------')
    print(summary_pivot_df)
    print('------------------------------------------------')
    summary_pivot_df.to_csv(summary_path)


def main():
    annotation_dirs = [
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240222_庭瀬',
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240301_庭瀬',
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240304_庭瀬_1回目+2回目',
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240306_庭瀬_1回目+2回目+再分類'
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240308_SCREEN様修正までを統合'
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240312_SCREEN様再振までを統合'
        # ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240520_SCREEN様_欠陥修正',
        ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240520_SCREEN様_欠陥修正_クラス番号変更'
    ]
    db_path = ctx.OUTPUT_ROOT / 'image_info.db'
    split_image_path = ctx.OUTPUT_ROOT / 'split_image_info.csv'
    summary_path = ctx.OUTPUT_ROOT / 'annotation_summary.csv'
    assert db_path.exists()

    # アノテーション情報を抽出
    annotation_dfs = []
    for annotation_dir in annotation_dirs:
        annotation_df_i = raw_data.extract_annotations(annotation_dir)
        annotation_dfs.append(annotation_df_i)
    annotation_df = pd.concat(annotation_dfs)

    # フォルダ内のアノテーション画像に重複がないかチェック
    check_s = annotation_df.groupby('image_id').size()
    check_s.columns = ['size']
    invalid_rows = check_s[check_s >= 2]
    assert len(invalid_rows) == 0

    # 分割テーブルにアノテーション情報を追加
    split_image_df = db.load_df(db_path=db_path, tbl_name='split_image')
    if 'defect_class' in split_image_df.columns:
        split_image_df = split_image_df.drop(columns=['defect_class'])
    split_image_df = pd.merge(split_image_df, annotation_df, how='left', on='image_id')
    assert len(annotation_df) == len(split_image_df)

    # DBを更新、CSVを出力
    db.save_df(split_image_df, db_path=db_path, tbl_name='split_image')
    split_image_df.to_csv(split_image_path, index=False)

    # クラス別のサマリを作成
    create_summary_df(split_image_df, summary_path)


if __name__ == '__main__':
    main()
