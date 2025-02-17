"""
テストデータの配置方法
1. アノテーション済画像フォルダ `20240520_SCREEN様_欠陥修正_クラス番号変更` を
   `bevel-ml/input` フォルダ配下にコピーして配置する。
"""
import _append_python_path
from pprint import pprint
from pandas import DataFrame
import pytest
from bevel_ml import env
from bevel_ml.data import BaseDataModule, BevelDataset, SupervisedDataModule, UnsupervisedDataModule

IMAGE_DIR = env.DATA_DIR / "input/20240520_SCREEN様_欠陥修正_クラス番号変更"
OUTPUT_DIR = env.TEST_DIR / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_bevel_dataset():
    dataset = BevelDataset(
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
    )
    print()
    print("-------------------------------------------------------------------------------")
    pprint(dataset.samples)
    print("-------------------------------------------------------------------------------")
    info_list = dataset.samples
    # ==> コンソール出力を目視確認する！

    # １つめデータをチェック
    info, sample = info_list[0], dataset[0]
    assert r"\01_正常\MEAS_12_FOUP_L01_Slot_17_A1_0007_I.tif" in str(info[1])
    assert sample[0].size == (560, 248)

    # ２つめデータをチェック
    info, sample = info_list[1], dataset[1]
    assert r"\02_汚れ\MEAS_05_FOUP_L01_Slot_11_C1_0358.tif" in str(info[1])
    assert sample[0].size == (248, 268)


def test_bevel_dataset_with_invalid_ids():
    with pytest.raises(RuntimeError) as e:
        BevelDataset(
            root=IMAGE_DIR,
            image_ids=["MEAS_12_FOUP_L01_Slot_17_A1_0007_I", "NOT_EXIST_IMAGE_ID"],
            targets=["01_normal", "01_normal"],
        )
    assert str(e.value) == "Specified image_id does not exists in the folder."


def _output_image_info_view(datamodule: BaseDataModule, file_name: str=None):
    selected_info_df = datamodule.image_info_view[::100]
    selected_columns = ["image_id", "defect_class", "target"]
    if "fold" in selected_info_df.columns:
        selected_columns += ["fold", "stage"]
    else:
        selected_columns += [ "stage"]
    print()
    print("------------------------------------------------------------------------------------------------")
    print(selected_info_df[selected_columns].to_string())
    print("------------------------------------------------------------------------------------------------")
    if file_name is not None:
        output_file = OUTPUT_DIR / file_name
        datamodule.image_info_view.to_csv(output_file, index=False)


def _output_summary_df(image_info_df: DataFrame, by_column: str) -> None:
    image_info_df = image_info_df.copy()

    image_info_df['split'] = image_info_df['split'].fillna('NA')
    summary_df = image_info_df.groupby([by_column, 'defect_class', 'bevel_section', 'split'], as_index=False).size()
    summary_df['section_split'] = summary_df['bevel_section'] + '_' + summary_df['split']
    summary_df['section_split'] = summary_df['section_split'].str.replace('_NA', '')
    summary_pivot_df = summary_df.pivot(index=[by_column, 'defect_class'], columns='section_split', values='size')
    summary_pivot_df = summary_pivot_df.fillna(0)
    summary_pivot_df = summary_pivot_df.astype(int)

    print("------------------------------------------------------------------")
    print(summary_pivot_df.to_string())
    print("------------------------------------------------------------------")


def _assert_batch(datamodule: BaseDataModule, stage: str) -> None:
    print(f"\nstage='{stage}' -------------------------------------------------")
    datamodule.setup(stage=stage)
    for i, (x, label) in enumerate(datamodule.train_dataloader()):
        print()
        print(f"label_in_batch = {label}")
        print(f"x_in_batch: {x.shape}")
        print(x[1, ...])
        assert tuple(x.shape) == (4, 3, 256, 256)
        if i == 1:
            break


def test_multiclass_datamodule():
    datamodule = SupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=4,  # 少量を目視確認
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=3,
    )
    datamodule.prepare_data_info()
    datamodule.set_fold(0)

    # image_info_viewの中身
    _output_image_info_view(datamodule, file_name="test_multiclass_datamodule.csv")
    # ==> 出力ファイルを目視確認する！

    # データ分割の内訳
    _output_summary_df(datamodule.image_info_df, by_column="fold")
    # ==> コンソール出力を目視確認する！

    # batchの中身
    _assert_batch(datamodule, stage="fit")
    _assert_batch(datamodule, stage="validate")
    _assert_batch(datamodule, stage="test")


def test_binary_datamodule():
    datamodule = SupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=4,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=3,
        is_binary=True,
    )
    datamodule.prepare_data_info()
    datamodule.set_fold(0)

    # image_info_viewの中身
    _output_image_info_view(datamodule, file_name="test_binary_datamodule.csv")
    # ==> 出力ファイルを目視確認する！

    # データ分割の内訳
    _output_summary_df(datamodule.image_info_df, by_column="fold")
    # ==> コンソール出力を目視確認する！

    # batchの中身
    _assert_batch(datamodule, stage="fit")
    _assert_batch(datamodule, stage="validate")
    _assert_batch(datamodule, stage="test")


def test_unsupervised_datamodule():
    datamodule = UnsupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=4,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        normal_test_split_ratio=0.2,
    )
    datamodule.prepare_data_info()

    # image_info_viewの中身
    _output_image_info_view(datamodule, file_name="test_unsupervised_datamodule.csv")
    # ==> 出力ファイルを目視確認する！

    # データ分割の内訳
    _output_summary_df(datamodule.image_info_df, by_column="stage")
    # ==> コンソール出力を目視確認する！

    # batchの中身
    _assert_batch(datamodule, stage="fit")
    _assert_batch(datamodule, stage="validate")
    _assert_batch(datamodule, stage="test")


def test_max_samples():
    # SupervisedDataModuleで、split_num=3、max_samples=100
    datamodule1 = SupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=16,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=3,
        max_samples=100,
    )
    datamodule1.prepare_data_info()
    
    # フォールド毎のデータ数をチェック
    assert len(datamodule1.image_info_view) == 100
    fold_size_s = datamodule1.image_info_view.groupby("fold").size()
    assert fold_size_s[0] >= 33
    assert fold_size_s[1] >= 33
    assert fold_size_s[2] >= 33

    # UnsupervisedDataModuleで、normal_test_split_ratio=0.25、max_samples=100
    datamodule2 = UnsupervisedDataModule(
        image_dir=IMAGE_DIR,
        image_size=256,
        batch_size=16,
        num_workers=0, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        normal_test_split_ratio=0.25,
        max_samples=100,
    )
    datamodule2.prepare_data_info()
    
    # stage毎のデータ数をチェック
    assert len(datamodule2.image_info_view) == 100
    stage_size_s = datamodule2.image_info_view.groupby("stage").size()
    assert stage_size_s["fit"] == 75
    assert stage_size_s["validate_test"] == 25


if __name__ == "__main__":
    """pytestを使わずに実行。
    """
    # test_multiclass_datamodule()
    test_max_samples()
