"""追加のアノテーション対象データを作成するスクリプト。
"""
import shutil
import pandas as pd
from bevel_ml import db
from bevel_ml import raw_data
import _context as ctx


def main():
    input_db = ctx.OUTPUT_ROOT / 'annotated_image_info.db'
    output_root = ctx.OUTPUT_ROOT / f'アノテーション前'
    output_db = output_root / 'image_info.db'

    # 出力先フォルダを初期化
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # コピーして出力DBを作成
    shutil.copy(input_db, output_db)

    # DBを読み込み
    original_image_df = db.load_df(db_path=input_db, tbl_name='original_image')
    exclude_image_df = original_image_df.query(f'selection_no==1')  # 1回目アノテーション対象
    split_df = db.load_df(db_path=input_db, tbl_name='split_image')

    # 2回目アノテーション対象データを選択
    # target_making_types = ['エッジ傷', 'エッジ傷+α']
    target_making_types = ['エッジ傷+α']
    selected_image_df = raw_data.select_extra_annotation_targets(
        original_image_df, exclude_image_df=exclude_image_df,
        target_making_types=target_making_types, n_samples_per_frame=200,
    )
    selected_image_ids = selected_image_df['image_id'].values.tolist()

    # オリジナル画像テーブルに選択情報を追加
    def in_selected_image_ids(image_id: str):
        return image_id in selected_image_ids
    selection_no = 2  # 2回目のデータ選択
    in_flag = original_image_df['image_id'].apply(in_selected_image_ids)
    original_image_df['selection_no'] = original_image_df['selection_no'].mask(in_flag, selection_no)

    # オリジナル画像テーブルを更新
    db.save_df(original_image_df, db_path=output_db, tbl_name='original_image')

    selected_summary_s = selected_image_df.groupby('making_defect_type').size()
    print('-------------------')
    print(selected_summary_s)
    print('-------------------')

    # オリジナル画像データを分割、保存、DB更新
    extra_split_df = raw_data.split_save_images(selected_image_df, output_root=output_root)
    split_df = pd.concat([split_df, extra_split_df])
    db.save_df(split_df, db_path=output_db, tbl_name='split_image')

    # テンプレートフォルダを作成
    template_root = ctx.OUTPUT_ROOT / f'アノテーション済'
    raw_data.make_template_class_dirs(parent_dir=template_root)


if __name__ == '__main__':
    main()
