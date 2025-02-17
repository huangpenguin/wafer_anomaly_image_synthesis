"""選択画像フォルダの情報から、画像情報DBを再構築する。
"""
from bevel_ml import db
from bevel_ml import raw_data
import _context as ctx


def main():
    image_dir = ctx.SHARED_DATA_DIR / '画像データ/オリジナル'
    annotation_dirs = {
        1: ctx.SHARED_DATA_DIR / '画像データ/アノテーション前/20240221_選択データ_1回目',
        2: ctx.SHARED_DATA_DIR / '画像データ/アノテーション前/20240229_選択データ_2回目',
    }
    db_path = ctx.OUTPUT_ROOT / 'image_info.db'

    # オリジナル画像データの情報を抽出、DB保存
    parsed_df = raw_data.find_images(image_dir)
    measurement_df, original_directory_df, original_image_df = raw_data.convert_to_er(parsed_df)
    db.save_df(measurement_df, db_path=db_path, tbl_name='measurement')
    db.save_df(original_directory_df, db_path=db_path, tbl_name='original_directory')

    # アノテーション対象データを選択
    for selection_no, annotation_dir in annotation_dirs.items():
        # オリジナル画像テーブルに情報を追加
        original_image_df = raw_data.reconstruct_annotation_targets(
            original_image_df, annotation_dir, selection_no=selection_no)
        selected_image_df = original_image_df.query(f'selection_no=={selection_no}')

        # オリジナル画像データを分割、DB保存
        split_image_df_i = raw_data.split_save_images(selected_image_df, output_root=None, output=False)
        db.save_df(split_image_df_i, db_path=db_path, tbl_name='split_image', if_exists='append')

    # オリジナル画像テーブルをDB保存
    db.save_df(original_image_df, db_path=db_path, tbl_name='original_image')


if __name__ == '__main__':
    main()
