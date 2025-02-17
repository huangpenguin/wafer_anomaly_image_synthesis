# 分類を実行する
import sys

sys.path.append(".")
from pathlib import Path

import hydra
import numpy as np
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import disable_possible_user_warnings
from omegaconf import DictConfig, OmegaConf, open_dict

from bevel_ml.anomaly_map_visualizer import AnomalymapVisualizer
from bevel_ml.data import UnsupervisedDataModule
from bevel_ml.models import get_model
from bevel_ml.result_visualizer import ResultVisualizer
from bevel_ml.utils.performance import Timer

# lightningのPossibleUserWarningを無効化
disable_possible_user_warnings()


def train(cfg: DictConfig) -> None:
    if cfg.case_ad.get("seed"):
        seed_everything(cfg.case_ad.seed)
    if not Path(cfg.case_ad.save_dir).exists():
        Path(cfg.case_ad.save_dir).mkdir(parents=True)
    total_timer = Timer()
    train_timer = Timer()
    predict_timer = Timer()
    with total_timer:
        datamodule = UnsupervisedDataModule(
            image_dir=cfg.case_ad.data_dir,
            image_size=cfg.case_ad.data_size,
            batch_size=cfg.case_ad.batch_size,
            num_workers=cfg.case_ad.num_workers,
            aug_dict=cfg.case_ad.aug_dict,
            normal_test_split_ratio=cfg.case_ad.normal_test_split_ratio,
            split_seed=cfg.case_ad.seed,
            train_dir=cfg.case_ad.train_dir,
            test_dir=cfg.case_ad.test_dir
        )
        datamodule.prepare_data_info()
        if cfg.case_ad.save_split_result:
            datamodule.save_split_result(
                test_fold=0,
                save_dir=Path(cfg.case_ad.data_dir).parent,
                data_name=cfg.case_ad.data_name
            )
        model = get_model(
            model_name=cfg.case_ad.model,
            transform=datamodule.test_transform,
            **cfg.case_ad,
        )
        save_dir = cfg.case_ad.save_dir
        exp_name = cfg.case_ad.id
        logger = CSVLogger(save_dir=save_dir, name=exp_name)
        trainer = Trainer(
            accelerator="auto",
            max_epochs=cfg.case_ad.epochs,
            log_every_n_steps=100,
            logger=logger,
            deterministic=True,
            num_sanity_val_steps=0,
            callbacks=[],
        )
        if cfg.case_ad.get("checkpoint_path", None) is None:
            with train_timer:
                trainer.fit(model=model, datamodule=datamodule)

        with predict_timer:
            predictions = trainer.predict(datamodule=datamodule, model=model)
            predictions = {
                key: np.concatenate([prediction[key] for prediction in predictions])
                for key in predictions[0].keys()
            }
            # flattened_scores = predictions["anomaly_maps"].reshape(len(datamodule.test_data), -1)
            # pd.DataFrame(flattened_scores, index=datamodule.test_data.image_ids).to_csv(
            #     Path(logger.log_dir) / "anomalymaps.csv", header=False
            # )

    total_time = total_timer.elapsed_time
    time_log = {
        "total_time": total_time,
        "train_time": train_timer.elapsed_time,
        "predict_time": predict_timer.elapsed_time,
        # "read_time_per_image": pred_valid_read_time / len(datamodule.test_data.samples),
        # "preprocess_time_per_image": pred_valid_prep_time / len(datamodule.test_data.samples),
        # "train_time_per_image": train_timer.elapsed_time / len(datamodule.train_data),
        # "predict_time_per_image": predict_timer.elapsed_time / len(datamodule.test_data),
    }
    # configの保存
    with open_dict(cfg):
        if datamodule.train_data is not None:
            cfg.n_train_images = len(datamodule.train_data)
        cfg.n_predict_images = len(datamodule.test_data)
    OmegaConf.save(cfg, Path(logger.log_dir) / "hparams.yaml")
    # 結果の出力
    visualizer = ResultVisualizer(
        case_id=cfg.case_ad.id,
        image_info_df=datamodule.image_info_df,
        pred_scores=predictions["pred_scores"],
        timer=time_log,
        manual_recall=cfg.case_ad.defect_recall,
        save_dir=Path(logger.log_dir),
    )
    visualizer.save_all()
    if (
        ("save_heatmap" in cfg.case_ad)
        and (cfg.case_ad.save_heatmap)
        and ("anomaly_maps" in predictions)
    ):
        aug_dict = cfg.case_ad.aug_dict
        if "change_contrast" in aug_dict and aug_dict["change_contrast"]:
            aug_dict["change_contrast"] = False
        datamodule = UnsupervisedDataModule(
            image_dir=cfg.case_ad.data_dir,
            image_size=cfg.case_ad.data_size,
            batch_size=cfg.case_ad.batch_size,
            num_workers=cfg.case_ad.num_workers,
            aug_dict=aug_dict,
            normal_test_split_ratio=cfg.case_ad.normal_test_split_ratio,
            split_seed=cfg.case_ad.seed,
            train_dir=cfg.case_ad.train_dir,
            test_dir=cfg.case_ad.test_dir
        )
        datamodule.prepare_data_info()
        datamodule.setup(stage="validate")
        a_visualizer = AnomalymapVisualizer(
            datamodule=datamodule,
            anomaly_maps=predictions["anomaly_maps"],
            save_dir=Path(logger.log_dir)
        )
        a_visualizer.save_anomaly_maps(gamma=cfg.case_ad.gamma_heatmap, save_mask=cfg.case_ad.save_mask, mask_thr=cfg.case_ad.mask_thr)


@hydra.main(version_base=None, config_path="..\\config", config_name="config_ad")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
