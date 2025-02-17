import re
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from shutil import copy
from typing import Any, Callable, Tuple, override

import numpy as np
import pandas as pd
import sklearn
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)

from bevel_ml.utils.performance import Timer
from bevel_ml.transforms import ChangeContrast


class BevelDataset(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        image_ids: list[str],
        targets: list[str],
        # HACK: これはRGBで読み込むが、問題ないか？
        loader: Callable[[str], Any] = default_loader,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        with Timer() as self._setup_timer:
            classes, class_to_idx = self._get_class_to_index(targets)
            samples = self._make_dataset(
                self.root, image_ids, targets, class_to_idx=class_to_idx)

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.image_ids = [s[0] for s in samples]
        self.targets = [s[2] for s in samples]  # list of class_idx
        self._load_timer = Timer()
        self._transform_timer = Timer()

    @property
    def setup_time(self) -> float:
        return self._setup_timer.elapsed_time

    @property
    def load_time(self) -> float:
        return self._load_timer.elapsed_time

    @property
    def transform_time(self) -> float:
        return self._transform_timer.elapsed_time

    @property
    def total_time(self) -> float:
        return self.setup_time + self.load_time + self.transform_time

    @property
    def load_time_per_image(self) -> float:
        return self.load_time / len(self)

    @property
    def transform_time_per_image(self) -> float:
        return self.transform_time / len(self)

    @staticmethod
    def _get_class_to_index(targets: list[str]) -> Tuple[list, dict]:
        classes = np.unique(targets)
        classes = np.sort(classes).tolist()
        indices = np.arange(len(classes)).tolist()
        return classes, dict(zip(classes, indices))

    @staticmethod
    def _find_images(directory: Path | str) -> dict[str, Path]:
        """指定フォルダ内の画像ファイルをリストアップする。
        """
        if not isinstance(directory, Path):
            directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f'{str(directory)} is not exist.')
        file_name_pattern = r'^(.+?)_([ACE])1_\d{4}(|_I|_O).tif$'
        file_list = [
            file for file in directory.glob('**/*')
            if re.search(file_name_pattern, str(file))
        ]
        file_list = sorted(file_list)
        id_to_path = [(file.stem, file) for file in file_list]
        return dict(id_to_path)

    @staticmethod
    def _make_dataset(
        directory: str | Path,
        image_ids: list[str],
        targets: list[str],
        class_to_idx: dict[str, int],
    ) -> list[tuple[str, int]]:
        """
        Args:
            directory: 
            image_ids: 
            targets: 
            class_to_idx: 
        Returns:
            samples of a form (image_id, path_to_sample, class)
        """
        id_to_path = BevelDataset._find_images(directory)

        # 指定IDの画像を選択する
        instances = []
        for image_id, target in zip(image_ids, targets):
            found_ids = list(id_to_path.keys())
            if image_id in found_ids:
                file_path = id_to_path[image_id]
                class_idx = class_to_idx[target]
                instances.append((image_id, file_path, class_idx))
            else:
                raise RuntimeError(
                    "Specified image_id does not exists in the folder.")
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_id, sample, target) where target is class_index of the target class.
        """
        with self._load_timer:
            image_id, path, target = self.samples[index]
            sample = self.loader(path)

        with self._transform_timer:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        # HACK: image_idも一緒に返すべきか要検討。IDと画像をペアにして呼び出し側で管理したい。
        # return image_id, sample, target
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BaseDataModule(LightningDataModule, ABC):

    # 確認表示用の主要な列
    PRIMARY_INFO_COLUMNS = [
        "image_id", "foup_slot", "bevel_section", "flame_no", "split",
        "making_defect_type", "defect_class",
    ]
    TARGET_COLUMN = "target"  # 推論対象のクラスラベルの列
    STAGE_COLUMN = "stage"  # ["fit", "validate_test"]を格納する列

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        aug_dict: dict,
        is_binary: bool,
        max_samples: int | None,
        subsample_seed: int | None,
        train_dir: str | None,
        test_dir: str | None
    ):
        super().__init__()

        # Parameters:
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_dict = aug_dict
        self.is_binary = is_binary
        self.max_samples = max_samples
        self.subsample_seed = subsample_seed
        self.train_dir = train_dir
        self.test_dir = test_dir

        # Attributes:
        match is_binary:
            case True:
                self._append_target = self._append_binary_target
            case False:
                self._append_target = self._append_multiclass_target
            case _:
                raise ValueError()
        self.train_transform = self._get_transform(self.aug_dict, is_train=True)
        self.val_transform = self._get_transform(self.aug_dict, is_train=False)
        self.test_transform = self.val_transform
        self.image_info_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.meas_data = None

    @property
    def exclude_defect_classes(self) -> list[str]:
        if self.is_binary:
            return ["50_unknown"]
        else:
            return ["13_gunyaa", "50_unknown"]

    @property
    @abstractmethod
    def view_columns(self) -> list[str]:
        pass

    @property
    def image_info_view(self) -> DataFrame:
        return self.image_info_df[self.view_columns]

    def _get_transform(self, aug_dict: dict, is_train: bool) -> list:
        aug_list = [ToImage()]
        if "change_contrast" in aug_dict and aug_dict["change_contrast"]:
            aug_list.append(ChangeContrast(aug_dict["change_contrast"]))
        aug_list.append(ToDtype(torch.float32, scale=True))
        if "p_hflip" in aug_dict and is_train:
            aug_list.append(RandomHorizontalFlip(aug_dict["p_hflip"]))
        if "p_vflip" in aug_dict and is_train:
            aug_list.append(RandomVerticalFlip(aug_dict["p_vflip"]))
        if "padding" in aug_dict and aug_dict["padding"]:
            aug_list.append(CenterCrop((self.image_size,)*2))
        else:
            aug_list.append(Resize((self.image_size,)*2))
        return Compose(aug_list)

    def _load_image_info_df(self, image_dir: str) -> DataFrame:
        info_file = Path(image_dir) / "split_image_info.csv"
        info_df = pd.read_csv(info_file)
        info_df = info_df.sort_values('image_id', ignore_index=True)
        return info_df

    def _filter_defect_class(self, image_info_df: DataFrame) -> DataFrame:
        """image_info_df から使用しない欠陥クラスのデータを除外する。
        """
        info_df = image_info_df.copy()
        info_df = info_df.query(
            f"defect_class not in {self.exclude_defect_classes}")
        return info_df

    @staticmethod
    def _append_multiclass_target(image_info_df: DataFrame) -> DataFrame:
        """image_info_df に多クラス分類用のクラスラベルを追加する。
        """
        info_df = image_info_df.copy()
        info_df["target"] = info_df["defect_class"]
        return info_df

    @staticmethod
    def _append_binary_target(image_info_df: DataFrame) -> DataFrame:
        """image_info_df に二値分類用のクラスラベルを追加する。
        """
        info_df = image_info_df.copy()
        info_df["target"] = info_df["defect_class"]
        info_df["target"] = info_df["target"].mask(
            info_df["defect_class"] == "01_normal", "01_good")
        info_df["target"] = info_df["target"].mask(
            info_df["defect_class"] != "01_normal", "02_defect")
        return info_df

    @staticmethod
    def _append_stratified_class(image_info_df: DataFrame) -> DataFrame:
        """image_info_df を層化分割したいカラムごとにクラス分けし、そのクラスラベルを新カラムに追加する。
        """
        info_df = image_info_df.copy()

        d1 = pd.DataFrame(
            info_df["defect_class"].unique(), columns=["defect_class"])
        d2 = pd.DataFrame(
            info_df["bevel_section"].unique(), columns=["bevel_section"])
        d3 = pd.DataFrame(info_df["split"].unique(), columns=["split"])
        stratified_df = d1.join(d2, how="cross")
        stratified_df = stratified_df.join(d3, how="cross")
        stratified_df["stratified_class"] = list(
            range(len(stratified_df)))  # 層化サンプリングのためのクラスラベル
        info_df = info_df.merge(stratified_df)

        return info_df

    @abstractmethod
    def _prepare_split(self, stage: str = None) -> None:
        """各stage用のデータ分割を作成し、image_info_df を更新する。
        """
        pass

    @abstractmethod
    def _subsample(self) -> None:
        """image_info_df をサブサンプリングして更新する。
        更新後のサンプル数は max_samples 以下。
        """
        pass

    def prepare_data_info(self) -> None:
        """image_info_df を読み込み、情報を付加する。
        """
        if (self.train_dir is None) and (self.test_dir is None):
            self.train_dir = self.image_dir
            self.test_dir = self.image_dir
            self.image_info_df = self._load_image_info_df(self.image_dir)
            self.image_info_df = self._filter_defect_class(self.image_info_df)
            self.image_info_df = self._append_target(self.image_info_df)
            self._prepare_split()
        else:
            train_image_info_df = pd.DataFrame()
            test_image_info_df = pd.DataFrame()
            if self.train_dir is not None:
                train_image_info_df = self._load_image_info_df(self.train_dir)
                train_image_info_df = self._filter_defect_class(train_image_info_df)
                train_image_info_df = self._append_target(train_image_info_df)
                train_image_info_df = self._set_stage(train_image_info_df,"fit")
            if self.test_dir is not None:
                test_image_info_df = self._load_image_info_df(self.test_dir)
                test_image_info_df = self._filter_defect_class(test_image_info_df)
                test_image_info_df = self._append_target(test_image_info_df)
                test_image_info_df = self._set_stage(test_image_info_df,"validate_test")
            self.image_info_df = pd.concat([train_image_info_df, test_image_info_df]).reset_index(drop=True)
        if self.image_info_df.empty:
            raise RuntimeError(
                "image_info_df is empty.")

        if self.max_samples is not None:
            self._subsample()
            
    def _set_stage(self, image_info_df: DataFrame, stage: str):
        """入力DataFrameに含まれる全データにstageの情報を付与する。
        """
        info_df = image_info_df.copy()
        info_df["stage"] = stage
        return info_df

    @override
    def prepare_data(self) -> None:
        pass  # 何もしない

    @override
    def setup(self, stage: str = None) -> None:
        assert self.image_info_df is not None
        assert "stage" in self.image_info_df.columns

        if stage == "fit":
            selected_df = self.image_info_df.query("stage=='fit'")
            image_ids = selected_df["image_id"].values
            targets = selected_df["target"].values
            self.train_data = BevelDataset(
                root=self.train_dir, image_ids=image_ids, targets=targets, transform=self.train_transform)
            selected_df = self.image_info_df.query("stage=='validate_test'")
            image_ids = selected_df["image_id"].values
            targets = selected_df["target"].values
            self.val_data = BevelDataset(
                root=self.test_dir, image_ids=image_ids, targets=targets, transform=self.val_transform)
        if stage == "validate" and self.val_data is None:
            selected_df = self.image_info_df.query("stage=='validate_test'")
            image_ids = selected_df["image_id"].values
            targets = selected_df["target"].values
            self.val_data = BevelDataset(
                root=self.test_dir, image_ids=image_ids, targets=targets, transform=self.val_transform)
        if stage == "test" or "predict":
            # test predictには指定がない限り validate と同じデータを使う
            selected_df = self.image_info_df.query("stage=='validate_test'")
            image_ids = selected_df["image_id"].values
            targets = selected_df["target"].values
            self.test_data = BevelDataset(
                root=self.test_dir, image_ids=image_ids, targets=targets, transform=self.test_transform)

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_data is not None
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.val_data is not None
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_data is not None
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_data is not None
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def meas_dataloader(self) -> EVAL_DATALOADERS:
        """処理時間測定用のDataLoaderを取得する。
        - データセットの内容はテスト用のものと同じだが、インスタンスは異なる。
        - このDataLoaderを取得するたびに新しい`self.meas_data`がセットされる。
        - num_workers=0
        """
        assert self.image_info_df is not None
        assert "stage" in self.image_info_df.columns

        # ここで毎回データセットを新規作成し、Timerを初期化する。
        selected_df = self.image_info_df.query("stage=='validate_test'")
        image_ids = selected_df["image_id"].values
        targets = selected_df["target"].values
        self.meas_data = BevelDataset(
            root=self.image_dir, image_ids=image_ids, targets=targets, transform=self.test_transform)
        return DataLoader(self.meas_data, batch_size=self.batch_size, num_workers=0)

    def get_weight(self) -> torch.Tensor:
        class_counts = np.unique(self.image_info_df.query("stage=='fit'").target, return_counts=True)[1]
        weight = np.ones_like(class_counts) / class_counts
        return torch.tensor(weight).float()
    
    def save_split_result(self, test_fold: int, save_dir: Path, data_name: str):
        """_prepare_splitの結果から、trainとtestに使用したデータを保存する。
        """
        def _save_files(df: pd.DataFrame, save_dir: Path, id_to_path: dict):
            for image_id, defect_class in df[["image_id", "defect_class"]].values:
                src = id_to_path[image_id]
                dst = save_dir / defect_class / (image_id + ".tif")
                if not dst.parent.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                copy(src, dst)
            
        id_to_path = BevelDataset._find_images(self.image_dir)
        
        test_dir = save_dir / (data_name + "_test")
        # test_df = self.image_info_df.query(f"fold=={test_fold}")
        test_df = self.image_info_df.query("stage=='validate_test'")
        _save_files(test_df, test_dir, id_to_path)
        test_df.to_csv(test_dir/"split_image_info.csv", index=False)
        
        train_dir = save_dir / (data_name + "_train")
        #train_df = self.image_info_df.query(f"fold!={test_fold}")
        train_df = self.image_info_df.query("stage=='fit'")
        _save_files(train_df, train_dir, id_to_path)
        train_df.to_csv(train_dir/"split_image_info.csv", index=False)

class SupervisedDataModule(BaseDataModule):
    """教師有りモデルのためのDataModuleクラス。以下のデータ分割を行う。
    - 交差検定用のデータ分割を行う。
    - 学習データの一部を、モデルのバリデーション用に使う。

    Args:
        image_dir (str): 
        image_size (int): 
        batch_size (int): 
        num_workers (int): 
        aug_dict (dict): 
        split_num (int, optional): 
        split_seed (int, optional): 
        val_is_test (bool, optional)      : val_dataとtest_dataを同じにするか否かのフラグ。Falseには未対応。
        val_split_ratio (float, optional) : val_is_test==Falseのとき使う。 train_splitに占めるval_splitの割合。
        is_binary (bool, optional)        : 二値分類用のクラスラベルを用いるか否かのフラグ
        max_samples (int, optional): 
        subsample_seed (int, optional): 
    """
    FOLD_COLUMN = "fold"  # 交差検定のフォールド番号を格納する列

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        aug_dict: dict,
        split_num: int | None = None,
        split_seed: int | None = None,
        val_is_test: bool = True,
        val_split_ratio: float | None = None,
        is_binary: bool = False,
        max_samples: int | None = None,
        subsample_seed: int | None = None,
        train_dir: str | None = None,
        test_dir: str | None = None
    ):
        super().__init__(
            image_dir, image_size, batch_size, num_workers,
            aug_dict, is_binary, max_samples, subsample_seed,
            train_dir=train_dir, test_dir=test_dir
        )

        # Parameters:
        self.split_num = split_num
        self.split_seed = split_seed
        self.val_is_test = val_is_test
        self.val_split_ratio = val_split_ratio

        # Attributes:
        self.fold: int | None = None

    @property
    @override
    def view_columns(self) -> list[str]:
        view_columns = deepcopy(self.PRIMARY_INFO_COLUMNS)
        if self.TARGET_COLUMN in self.image_info_df.columns:
            view_columns.append(self.TARGET_COLUMN)
        if self.FOLD_COLUMN in self.image_info_df.columns:
            view_columns.append(self.FOLD_COLUMN)
        if self.STAGE_COLUMN in self.image_info_df.columns:
            view_columns.append(self.STAGE_COLUMN)
        return view_columns

    @override
    def _prepare_split(self) -> None:
        if not self.val_is_test:
            raise NotImplementedError()
        assert self.split_num is not None and self.split_num >= 2

        # 層化分割のためのクラスを追加
        info_df = self.image_info_df
        info_df = self._append_stratified_class(info_df)

        # 上記クラスに基づく層化k分割
        X = info_df["image_id"].values
        y = info_df["stratified_class"].values
        skf = StratifiedKFold(n_splits=self.split_num,
                              shuffle=True, random_state=self.split_seed)
        for fold, (_, test_idx) in enumerate(skf.split(X, y)):
            info_df.loc[test_idx, "fold"] = fold

        self.image_info_df = info_df

    @override
    def _subsample(self) -> None:
        info_df = self.image_info_df
        index = info_df.index.values
        stratify = info_df["fold"].values
        n_samples = np.min([len(info_df), self.max_samples])
        selected_index = sklearn.utils.resample(
            index, n_samples=n_samples, stratify=stratify, random_state=self.subsample_seed)
        self.image_info_df = info_df.loc[selected_index, :]

    def set_fold(self, fold: int) -> None:
        if fold not in list(range(self.split_num)):
            raise ValueError()
        assert self.image_info_df is not None

        info_df = self.image_info_df.copy()
        info_df["stage"] = None
        info_df["stage"] = info_df["stage"].mask(info_df["fold"] == fold, "validate_test")
        info_df["stage"] = info_df["stage"].mask(info_df["fold"] != fold, "fit")
        self.image_info_df = info_df


class UnsupervisedDataModule(BaseDataModule):
    """教師無しモデルのためのDataModuleクラス。以下のデータ分割を行う。
    - 良品クラスは、学習データとテストデータに分割する。
    - 欠陥クラスは、全てテストデータとして使う。
    - テストデータの一部を、モデルのバリデーション用に使う。

    Args:
        image_dir (str): 
        image_size (int): 
        batch_size (int): 
        num_workers (int): 
        aug_dict (dict): 
        normal_test_split_ratio (float)  : 良品データ(01_normal)に占めるのtest_data割合
        split_seed (int, optional): 
        val_is_test (bool, optional)     : val_dataとtest_dataを同じにするか否かのフラグ。Falseには未対応。
        val_split_ratio (float, optional): val_is_test==Falseのとき使う。test_dataに占めるval_dataの割合。
        max_samples (int, optional): 
        subsample_seed (int, optional): 
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        aug_dict: dict,
        normal_test_split_ratio: float,
        split_seed: int | None = None,
        val_is_test: bool = True,
        val_split_ratio: float | None = None,
        max_samples: int | None = None,
        subsample_seed: int | None = None,
        train_dir: str | None = None,
        test_dir: str | None = None
    ):
        super().__init__(
            image_dir, image_size, batch_size, num_workers, aug_dict,
            is_binary=True, max_samples=max_samples, subsample_seed=subsample_seed,
            train_dir=train_dir, test_dir=test_dir
        )

        # Parameters:
        self.normal_test_split_ratio = normal_test_split_ratio
        self.split_seed = split_seed
        self.val_split_ratio = val_split_ratio
        self.val_is_test = val_is_test

    @property
    @override
    def view_columns(self) -> list[str]:
        view_columns = deepcopy(self.PRIMARY_INFO_COLUMNS)
        if self.TARGET_COLUMN in self.image_info_df.columns:
            view_columns.append(self.TARGET_COLUMN)
        if self.STAGE_COLUMN in self.image_info_df.columns:
            view_columns.append(self.STAGE_COLUMN)
        return view_columns

    @override
    def _prepare_split(self) -> None:
        assert 0.0 < self.normal_test_split_ratio < 1.0
        if not self.val_is_test:
            raise NotImplementedError()

        # 層化分割のためのクラスを追加
        info_df = self.image_info_df
        info_df = self._append_stratified_class(info_df)

        # 良品データを分割クラスに基づいて層化分割
        good_df = info_df.query("defect_class=='01_normal'").reset_index()

        # HACK: train_test_split を使いたいが、なぜか動かない。
        # index = info_df.index.values
        # y = info_df["stratified_class"].values
        # train_idx, test_idx, y_train, y_test = train_test_split(
        #     index, y, test_size=self.normal_test_split_ratio, random_state=self.split_seed)
        # ==> これは動く。
        # train_idx, test_idx, y_train, y_test = train_test_split(
        #     index, y, test_size=self.normal_test_split_ratio, random_state=self.split_seed, stratify=y)
        # ==> これはなぜかエラー。
        # ==> StratifiedKFoldによる以下の代替実装とした
        X = good_df["image_id"].values
        y = good_df["stratified_class"].values
        n_splits = int(1 / self.normal_test_split_ratio)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.split_seed)
        train_idx, test_idx = next(skf.split(X, y))

        good_df.loc[train_idx, "stage"] = "fit"
        good_df.loc[test_idx, "stage"] = "validate_test"

        # 欠陥データは全てテストデータにする
        defect_df = info_df.query("defect_class!='01_normal'").reset_index()
        defect_df["stage"] = "validate_test"

        info_df = pd.concat([good_df, defect_df], axis=0).sort_values("image_id", ignore_index=True)
        self.image_info_df = info_df

    @override
    def _subsample(self) -> None:
        info_df = self.image_info_df
        index = info_df.index.values
        stratify = info_df["stage"].values
        n_samples = np.min([len(info_df), self.max_samples])
        selected_index = sklearn.utils.resample(
            index, n_samples=n_samples, stratify=stratify, random_state=self.subsample_seed)
        self.image_info_df = info_df.loc[selected_index, :]


def measure_dataloader_time(datamodule: BaseDataModule, verbose: bool = True) -> None:
    """"DataLoaderによるテストデータの読み込み時間を測定する。
    - 測定結果は`datamodule.meas_data`に保存される。
    """
    dataloader = datamodule.meas_dataloader()
    for batch_idx, (x, label) in enumerate(dataloader):
        if verbose and batch_idx % 10 == 0.0:
            print(f"batch_idx={batch_idx}, batch_size={len(label)}, batch_data is loaded.")
