"""
bevel_ml.env
~~~~~~~~~~~~
実行時の環境情報。
"""
from pathlib import Path

# プロジェクトのホームディレクトリ
PROJECT_HOME = Path(__file__).parent.parent.resolve()

# データ入出力用のディレクトリ
DATA_DIR = PROJECT_HOME / 'data'

# ログ出力用のディレクトリ
LOG_DIR = PROJECT_HOME / 'log'

# 設定ファイル用のディレクトリ
CONFIG_DIR = PROJECT_HOME / 'config'

# スクリプトディレクトリ
SCRIPT_DIR = PROJECT_HOME / 'scripts'

# テストディレクトリ
TEST_DIR = PROJECT_HOME / 'tests'


def test():
    print(f'PROJECT_HOME={PROJECT_HOME}')
    print(f'DATA_DIR={DATA_DIR}')
    print(f'LOG_DIR={LOG_DIR}')
    print(f'CONFIG_DIR={CONFIG_DIR}')
    print(f'SCRIPT_DIR={SCRIPT_DIR}')
    print(f'TEST_DIR={TEST_DIR}')


if __name__ == '__main__':
    test()
