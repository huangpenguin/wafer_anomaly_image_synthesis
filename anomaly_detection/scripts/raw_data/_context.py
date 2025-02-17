"""実行環境に関する設定値。
"""
from pathlib import Path
from bevel_ml import env

COMMON_DATA_DIR = env.DATA_DIR
SHARED_DATA_DIR = Path('D:/H23057-J_とめ研社内共有')
INPUT_ROOT = Path(__file__).parent / '_input'  # 今のところ使っていない
OUTPUT_ROOT = Path(__file__).parent / '_output'
