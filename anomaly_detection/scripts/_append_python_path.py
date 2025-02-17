"""PYTHONPATHに`bevel_ml`ディレクトリを追加する。
"""
import sys
from pathlib import Path
project_home = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_home))
