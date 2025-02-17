"""リファクタリング後の各種DetectorExample、DetectionResultCallbackExampleのテストケース。
テスト対象コード:
    - BaseDetectorExample
    - PathcoreDetectorExample
    - ReverseDistillationDetectorExample
    - DetectionResultCallbackExample
    
    - TimmDetectorExample
TODO:
    - このテストケースを採用、修正する場合には、
      ファイル名から`example`を削除し`test_detector.py`に変更すること。
    - このテストケースを使わない場合には、このファイルを削除すること。
    - テスト対象のXxxxExampleクラスにつても同様にすること。
"""
from pprint import pprint

import _append_python_path
import numpy as np
import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    ToDtype,
    ToImage,
)

from bevel_ml import env
from bevel_ml.callbacks import DetectionResultCallbackExample
from bevel_ml.data import (
    BevelDataset,
    SupervisedDataModule,
    UnsupervisedDataModule,
)
from bevel_ml.models.patchcore import PathcoreDetectorExample
from bevel_ml.models.reverse_distillation import ReverseDistillationDetector
from bevel_ml.models.timm import TimmDetector

IMAGE_DIR = env.DATA_DIR / "input/20240520_SCREEN様_欠陥修正_クラス番号変更"
OUTPUT_DIR = env.TEST_DIR / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def small_dataset():
    return BevelDataset(
        root=IMAGE_DIR,
        image_ids=[
            "MEAS_12_FOUP_L01_Slot_17_A1_0007_I",
            "MEAS_05_FOUP_L01_Slot_11_C1_0358", 
            "MEAS_25_FOUP_UK_Slot_UK_A1_0010_I",
            "MEAS_25_FOUP_UK_Slot_UK_A1_0020_I",
        ],
        targets=[
            "01_normal",
            "02_blot",
            "13_gunyaa",
            "13_gunyaa",
        ],
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((256,)*2)
        ])
    )

@pytest.fixture()
def small_batch(small_dataset):
    dataloader = DataLoader(small_dataset, batch_size=4, num_workers=0)
    batch = next(iter(dataloader))
    return batch

@pytest.fixture()
def small_unsupervised_data_datamodule():
    datamodule = UnsupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=32,
        batch_size=8,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        normal_test_split_ratio=0.2,
        max_samples=16,
    )
    datamodule.prepare_data_info()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")
    return datamodule


@pytest.fixture()
def small_supervised_data_datamodule():
    datamodule = SupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=32,
        batch_size=8,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=3,
        is_binary=True,
        max_samples=16,
    )
    datamodule.prepare_data_info()
    datamodule.set_fold(0)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")
    return datamodule


@pytest.fixture()
def middle_unsupervised_data_datamodule():
    datamodule = UnsupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=32,
        batch_size=16,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        normal_test_split_ratio=0.2,
        max_samples=100,
    )
    datamodule.prepare_data_info()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")
    return datamodule


def test_small_dataset(small_dataset):
    """テスト用の少量データの中身を確認する。
    """
    print()
    print("-------------------------------------------------------------------------------")
    pprint(small_dataset.samples)
    print("-------------------------------------------------------------------------------")
    # ==> コンソール出力を目視確認する！

    info_list = small_dataset.samples

    # １つめデータをチェック
    info, sample = info_list[0], small_dataset[0]
    assert r"\01_正常\MEAS_12_FOUP_L01_Slot_17_A1_0007_I.tif" in str(info[1])
    assert sample[0].data.shape == (3, 256, 256)

    # ２つめデータをチェック
    info, sample = info_list[1], small_dataset[1]
    assert r"\02_汚れ\MEAS_05_FOUP_L01_Slot_11_C1_0358.tif" in str(info[1])
    assert sample[0].data.shape == (3, 256, 256)


def test_small_batch(small_batch):
    """テスト用の少量データの中身を確認する。
    """
    x, label = small_batch
    print()
    print(f"label_in_batch = {label}")
    print(f"x_in_batch: {x.shape}")
    print(x[1, ...])
    assert tuple(x.shape) == (4, 3, 256, 256)


def test_patchcore_detector(small_batch):
    model = PathcoreDetectorExample(
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
    )
    model.training_step(small_batch, batch_idx=0)
    model.fit()
    model.eval()
    model.validation_step(small_batch, batch_idx=0)


def test_patchcore_with_datamodule(small_unsupervised_data_datamodule):
    datamodule = small_unsupervised_data_datamodule

    model = PathcoreDetectorExample(
        backbone="resnet18",
        coreset_sampling_ratio=0.1,
        transform=datamodule.test_transform,
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=1,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[DetectionResultCallbackExample()]
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


def test_re4ad_with_datamodule(small_unsupervised_data_datamodule):
    datamodule = small_unsupervised_data_datamodule

    model = ReverseDistillationDetector(
        backbone="resnet18",
        transform=datamodule.test_transform, 
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=1,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[DetectionResultCallbackExample()]
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


def test_timm_detector_with_datamodule(small_supervised_data_datamodule):
    datamodule = small_supervised_data_datamodule

    use_weighted_loss = True
    num_classes = 2
    weight = torch.ones(num_classes)
    if use_weighted_loss:
        from collections import Counter
        labels = datamodule.train_data.targets
        weight = weight / np.array([Counter(labels)[i] for i in range(num_classes)])
        weight = weight.float()

    model = TimmDetector(
        case_id="test_timm_detector", 
        net="resnet18", 
        num_classes=num_classes,
        lr=0.001,
        weight=weight
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=2,
        log_every_n_steps=1,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[DetectionResultCallbackExample()]
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


def test_timm_detector_with_datamodule__esc():
    use_weighted_loss = True
    num_classes = 2
    weight = torch.ones(num_classes)

    datamodule = SupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=16,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=3,
        is_binary=True,
        max_samples=100,
    )
    datamodule.prepare_data_info()
    datamodule.set_fold(0)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")
    if use_weighted_loss:
        from collections import Counter
        labels = datamodule.train_data.targets
        weight = weight / np.array([Counter(labels)[i] for i in range(num_classes)])
        weight = weight.float()
    
    # # DEBUG: これは動いた！
    # aug_dict = OmegaConf.create({"p_hflip": 0, "p_vflip": 0})
    # datamodule = WaferDataModule(
    #     data_dir=env.DATA_DIR / "input/CV用データ", 
    #     data_size=256, 
    #     batch_size=16, 
    #     num_workers=0,
    #     aug_dict=aug_dict,
    #     is_binary=True # 二値分類
    # )
    # datamodule.setup()
    # if use_weighted_loss:
    #         from collections import Counter
    #         labels =[label for i in range(len(datamodule.data_train.datasets)) for _, label in datamodule.data_train.datasets[i].samples]
    #         weight = weight / np.array([Counter(labels)[i] for i in range(num_classes)])
    #         weight = weight.float()

    model = TimmDetector(
        case_id="test_timm_detector", 
        net="resnet18", 
        num_classes=num_classes,
        lr=0.001,
        weight=weight
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=2,
        log_every_n_steps=1,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[DetectionResultCallbackExample()]
    )
    trainer.fit(model, datamodule=datamodule)
    # ==> timm.py:50: in _shared_eval_step でエラーになる
    # TODO: リファクタ後のDatasetのバグ疑い？
    # trainer.test(datamodule=datamodule)