"""画像情報DBを読み込んで表示する例
"""
from bevel_ml import db
import _context as ctx


def main():
    db_path = ctx.OUTPUT_ROOT / 'image_info.db'

    # 元画像テーブルを読み込み
    original_image_df = db.load_df(db_path=db_path, tbl_name='original_image')
    print(original_image_df.head())

    # 分割画像テーブルを読み込み
    split_image_df = db.load_df(db_path=db_path, tbl_name='split_image')
    print(split_image_df.head())


if __name__ == '__main__':
    main()
