"""
"""
import _append_python_path
from pprint import pprint
from pandas import DataFrame
import numpy as np
import torch
from bevel_ml import env
from anomalib.metrics.threshold import F1AdaptiveThreshold

from bevel_ml.metrics.threshold import ManualRecallThreshold
# from bevel_ml.metrics.threshold import F1AdaptiveThreshold

IMAGE_DIR = env.DATA_DIR / "input/20240520_SCREEN様_欠陥修正_クラス番号変更"
OUTPUT_DIR = env.TEST_DIR / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_search_sorted():
    # sorted_sequence_1d = torch.tensor([1, 3, 5, 7, 9])
    # sorted_sequence_1d = torch.tensor([1, 3, 5, 7, 9])
    # sorted_sequence_1d = torch.tensor([9, 9, 3, 5, 7])
    # value = 8

    # sorted_sequence_1d = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.9, 0.5, 0.1])  # 未ソート
    sorted_sequence_1d = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8])  # 未ソート
    # sorted_sequence_1d, indices = torch.sort(sorted_sequence_1d)
    value = 0.6
    index = torch.searchsorted(sorted_sequence_1d, value, right=True)
    print()
    print("-----------")
    print(f"index={index}")
    print(f"search_value={sorted_sequence_1d[index]}")
    print("-----------")


def test_manual_recall_threshold():
    # preds = torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3])
    # labels = torch.tensor([0, 0, 0, 1, 1])

    preds = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = torch.tensor([0, 0, 1, 0, 1, 1])
    
    print()
    print("================================================================")
    manual_recall = 0.1
    manual_threshold = ManualRecallThreshold(manual_recall, verbose=True)
    threshold = manual_threshold(preds, labels)
    print(f"manual_recall={manual_recall}, threshold={threshold}\n")

    manual_recall = 0.4
    manual_threshold = ManualRecallThreshold(manual_recall, verbose=True)
    threshold = manual_threshold(preds, labels)
    print(f"manual_recall={manual_recall}, threshold={threshold}\n")

    manual_recall = 0.7
    manual_threshold = ManualRecallThreshold(manual_recall, verbose=True)
    threshold = manual_threshold(preds, labels)
    print(f"manual_recall={manual_recall}, threshold={threshold}\n")

    manual_recall = 0.9
    manual_threshold = ManualRecallThreshold(manual_recall, verbose=True)
    threshold = manual_threshold(preds, labels)
    print(f"manual_recall={manual_recall}, threshold={threshold}\n")

    print("================================================================")


def test_manual_recall_threshold_with_engine():
    from anomalib.engine import Engine
    engine = Engine(
        threshold=ManualRecallThreshold(manual_recall=0.7),
        task="classification",max_epochs=1,
    )
