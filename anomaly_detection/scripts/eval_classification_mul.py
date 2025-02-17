"""分類モデルの性能評価を実行するスクリプト"""

import sys

sys.path.append(".")
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import disable_possible_user_warnings
from omegaconf import DictConfig, OmegaConf, open_dict

from bevel_ml.data import SupervisedDataModule
from bevel_ml.models import get_model
from bevel_ml.result_visualizer import ResultVisualizer
from bevel_ml.utils.performance import Timer

# lightningのPossibleUserWarningを無効化
disable_possible_user_warnings()


def train(cfg: DictConfig) -> tuple[dict, dict]:
    if cfg.case_mul_clf.get("seed"):
        seed_everything(cfg.case_mul_clf.seed)
    if not Path(cfg.case_mul_clf.save_dir).exists():
        Path(cfg.case_mul_clf.save_dir).mkdir(parents=True)
    datamodule = SupervisedDataModule(
        image_dir=cfg.case_mul_clf.data_dir,
        image_size=cfg.case_mul_clf.data_size,
        batch_size=cfg.case_mul_clf.batch_size,
        num_workers=cfg.case_mul_clf.num_workers,
        aug_dict=cfg.case_mul_clf.aug_dict,
        split_num=cfg.case_mul_clf.split_num,
        split_seed=cfg.case_mul_clf.seed,
        is_binary=False,
        train_dir=cfg.case_mul_clf.train_dir,
        test_dir=cfg.case_mul_clf.test_dir
    )
    datamodule.prepare_data_info()
    if cfg.case_mul_clf.save_split_result:
        datamodule.save_split_result(
            test_fold=0,
            save_dir=Path(cfg.case_mul_clf.data_dir).parent,
            data_name=cfg.case_mul_clf.data_name
        )
    num_loop = cfg.case_mul_clf.num_split if cfg.case_mul_clf.cross_val else 1

    for num_fold in range(num_loop):
        total_timer = Timer()
        train_timer = Timer()
        predict_timer = Timer()
        with total_timer:
            if num_loop == 1:
                exp_name = cfg.case_mul_clf.id
            else:
                exp_name = cfg.case_mul_clf.id + f"\\cv{num_fold}"
            datamodule.set_fold(fold=num_fold)
            logger = CSVLogger(save_dir=cfg.case_mul_clf.save_dir, name=exp_name)
            trainer = Trainer(
                accelerator="auto",
                max_epochs=cfg.case_mul_clf.epochs,
                log_every_n_steps=100,
                logger=logger,
                deterministic=True,
                num_sanity_val_steps=0,
                callbacks=[],
            )
            weight = torch.ones(cfg.case_mul_clf.num_classes)
            if cfg.case_mul_clf.use_weighted_loss:
                weight = datamodule.get_weight()
            model = get_model(
                model_name=cfg.case_mul_clf.model,
                transform=datamodule.test_transform,
                weight=weight,
                **cfg.case_mul_clf,
            )
            if cfg.case_mul_clf.get("checkpoint_path", None) is None:
                with train_timer:
                    trainer.fit(datamodule=datamodule, model=model)

            with predict_timer:
                predictions = trainer.predict(datamodule=datamodule, model=model)
                predictions = {
                    key: np.concatenate([prediction[key] for prediction in predictions])
                    for key in predictions[0].keys()
                }

        time_log = {
            "total_time": total_timer.elapsed_time,
            "train_time": train_timer.elapsed_time,
            "predict_time": predict_timer.elapsed_time,
            # "read_time_per_image": pred_valid_read_time / len(datamodule.test_data.samples),
            # "preprocess_time_per_image": pred_valid_prep_time / len(datamodule.test_data.samples),
            # "train_time_per_image": train_timer.elapsed_time
            # / len(datamodule.train_data),
            # "predict_time_per_image": predict_timer.elapsed_time
            # / len(datamodule.test_data),
        }

        # configの保存
        with open_dict(cfg):
            if datamodule.train_data is not None:
                cfg.n_train_images = len(datamodule.train_data)
            cfg.n_predict_images = len(datamodule.test_data)
        OmegaConf.save(cfg, Path(logger.log_dir) / "hparams.yaml")

        # 結果の出力
        visualizer = ResultVisualizer(
            case_id=cfg.case_mul_clf.id,
            image_info_df=datamodule.image_info_df,
            pred_scores=predictions["pred_scores"],
            timer=time_log,
            save_dir=Path(logger.log_dir),
            is_binary=False,
        )
        visualizer.save_all()


@hydra.main(version_base=None, config_path="..\\config", config_name="config_mul_clf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
