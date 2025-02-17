"""DataLoaderによるテストデータの読み込み時間を測定するスクリプト。
"""
import _append_python_path
from time import sleep
from bevel_ml import env
from bevel_ml.data import SupervisedDataModule, measure_dataloader_time
from bevel_ml.utils.performance import Timer


def main():
    image_dir = env.DATA_DIR / "input/20240520_SCREEN様_欠陥修正_クラス番号変更"
    datamodule = SupervisedDataModule(
        image_dir=image_dir,
        image_size=256,
        batch_size=16,
        num_workers=4, 
        aug_dict={"p_hflip": 0, "p_vflip": 0},
        split_num=2,
        is_binary=True,
    )
    datamodule.prepare_data_info()
    datamodule.set_fold(0)

    with Timer() as timer:
        measure_dataloader_time(datamodule, verbose=True)
        # measure_dataloader_time(datamodule, verbose=False)
    total_time2 = timer.elapsed_time

    meas_data = datamodule.meas_data
    print("----------------------------------------------------")
    print(f"dataset_size = {len(meas_data)}")
    print()
    print(f"setup_time = {meas_data.setup_time}")
    print(f"load_time = {meas_data.load_time}")
    print(f"transform_time = {meas_data.transform_time}")
    print(f"total_time1 = {meas_data.total_time}")
    print(f"total_time2 = {total_time2}")
    print()
    print(f"load_time_per_image = {meas_data.load_time_per_image}")
    print(f"transform_time_per_image = {meas_data.transform_time_per_image}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
