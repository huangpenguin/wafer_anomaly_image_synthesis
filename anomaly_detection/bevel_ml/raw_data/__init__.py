"""
bevel_ml.raw_data
~~~~~~~~~~~~~~~~~
元データを扱うモジュール。画像情報の抽出、アノテーション用の画像分割等。
前処理は別モジュールで行う。
"""
from ._original import find_images, convert_to_er
from ._selection import select_annotation_targets, select_extra_annotation_targets, reconstruct_annotation_targets
from ._split import split_save_images, make_template_class_dirs, extract_annotations, remove_mistake_files
from ._split import CLASS_DIR_NAMES, CLASSES
