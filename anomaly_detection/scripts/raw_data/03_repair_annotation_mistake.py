"""アノテーション間違いの修正用スクリプト。
"""
import pandas as pd
from bevel_ml import raw_data
import _context as ctx


def main():
    # 修正対象フォルダ
    mistake_dir = ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240312_SCREEN様再振までを統合'

    # 再アノテーションした差分フォルダ
    re_annotation_dirs = [
        ctx.SHARED_DATA_DIR / '画像データ/アノテーション済/20240312_SCREEN様_再振 (差分)',
    ]

    # 再アノテーション画像データの情報を抽出
    annotation_dfs = []
    for re_annotation_dir in re_annotation_dirs:
        annotation_df_i = raw_data.extract_annotations(re_annotation_dir)
        annotation_dfs.append(annotation_df_i)
    annotation_df = pd.concat(annotation_dfs)
    mistake_image_ids = annotation_df['image_id'].values

    # アノテーション間違いデータを削除
    raw_data.remove_mistake_files(mistake_dir, mistake_image_ids=mistake_image_ids)

    # ==> この後、修正対象フォルダに、手動で再アノテーションフォルダを上書きする。


if __name__ == '__main__':
    main()
