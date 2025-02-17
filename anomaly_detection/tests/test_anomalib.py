"""
"""
import _append_python_path
import pytest
import torch
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    ToDtype,
    ToImage,
)
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from bevel_ml import env

IMAGE_DIR = env.DATA_DIR / "input/20240520_SCREEN様_欠陥修正_クラス番号変更"
OUTPUT_DIR = env.TEST_DIR / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_folder_module():
    transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Resize((256,)*2)
    ])
    root = env.DATA_DIR / "input/20240501/cv0"
    datamodule = Folder(
        name="wafer",
        root=root,
        normal_dir=(root / "12_normal"),
        abnormal_dir=(root / "11_tiger_stripe"),
        normal_test_dir=(root / "12_normal"),
        normal_split_ratio=0.0,
        test_split_ratio=0.0,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        transform=transform,
        train_batch_size=16,
        eval_batch_size=16,
        task="classification",
        num_workers=0,
        seed=0
    )
    datamodule.setup(stage="validate")
    batch = next(iter(datamodule.val_dataloader()))
    print(batch)
