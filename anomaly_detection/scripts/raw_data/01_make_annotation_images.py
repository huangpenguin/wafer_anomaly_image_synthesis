import shutil
from bevel_ml import db
from bevel_ml import raw_data
import _context as ctx


def main():
    image_dir = ctx.SHARED_DATA_DIR / '画像データ/オリジナル'
    output_root = ctx.OUTPUT_ROOT / f'アノテーション前'
    db_path = output_root / 'image_info.db'

    # 出力先フォルダを削除
    if output_root.exists():
        shutil.rmtree(output_root)

    # オリジナル画像データの情報を抽出
    parsed_df = raw_data.find_images(image_dir)
    measurement_df, original_directory_df, original_image_df = raw_data.convert_to_er(parsed_df)
    db.save_df(measurement_df, db_path=db_path, tbl_name='measurement')
    db.save_df(original_image_df, db_path=db_path, tbl_name='original_image')

    # アノテーション対象データを選択
    selected_image_df = raw_data.select_annotation_targets(original_image_df, n_samples_per_frame=100)
    db.save_df(selected_image_df, db_path=db_path, tbl_name='selected_image')

    selected_summary_s = selected_image_df.groupby('making_defect_type').size()
    print('-------------------')
    print(selected_summary_s)
    print('-------------------')

    # オリジナル画像データを分割、保存
    split_image_df = raw_data.split_save_images(selected_image_df, output_root=output_root)
    db.save_df(split_image_df, db_path=db_path, tbl_name='split_image')

    # テンプレートフォルダを作成
    template_root = ctx.OUTPUT_ROOT / f'アノテーション済'
    raw_data.make_template_class_dirs(parent_dir=template_root)


if __name__ == '__main__':
    main()
