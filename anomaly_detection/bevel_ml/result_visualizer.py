import sys

sys.path.append(".")
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import dataframe_image as dfi
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchmetrics.classification import BinaryROC, BinaryAUROC
from torchmetrics.functional.classification import binary_stat_scores

from bevel_ml.metrics.threshold import ManualRecallThreshold


class ResultVisualizer:
    def __init__(
        self,
        case_id: str,
        image_info_df: pd.DataFrame,
        pred_scores: np.ndarray,
        timer: dict,
        manual_recall: float = 0.8,
        thr: float = 0.5,
        save_dir: Path = None,
        is_binary: bool = True,
    ):
        self.case_id = case_id
        self.image_info_df = image_info_df
        self.defect_types = np.unique(image_info_df.defect_class)
        self.val_targets = image_info_df.query("stage=='validate_test'").target.values
        # self.val_targets_mul = np.array(
        #     [int(target.split("_")[0]) - 1 for target in self.val_targets]
        # )
        self.val_original_targets = image_info_df.query(
            "stage=='validate_test'"
        ).defect_class.values
        self.val_targets_mul = np.array(
            [np.where(self.defect_types==target)[0].item() for target in self.val_original_targets]
        )
        self.val_targets_bin = np.where(self.val_targets_mul < 1, 0, 1)
        self.image_id = image_info_df.query("stage=='validate_test'").image_id.values
        self.timer = timer
        self.save_dir = save_dir
        self.is_binary = is_binary

        if self.is_binary:
            # スコアのノーマライズ (二値の場合のみ)
            manual_threshold = ManualRecallThreshold(manual_recall, verbose=False)
            threshold = manual_threshold(
                torch.tensor(pred_scores), torch.tensor(self.val_targets_bin)
            ).numpy()
            normalized_scores = (
                (pred_scores - threshold) / (pred_scores.max() - pred_scores.min())
            ) + 0.5
            normalized_scores = np.minimum(normalized_scores, 1)
            normalized_scores = np.maximum(normalized_scores, 0)
            self.val_scores = normalized_scores
            self.val_scores_raw = pred_scores
            self.val_preds = normalized_scores > thr

            self.class_names = {0: "01_good", 1: "02_defect"}
            predicted_class = [
                self.class_names[val_pred] for val_pred in self.val_preds
            ]
            normal = np.ones_like(self.val_scores) - self.val_scores
            prob_df = pd.DataFrame(
                {"good_prob": normal, "defect_prob": self.val_scores}
            )
        else:
            self.val_preds_mul = np.argmax(pred_scores, axis=1)
            self.val_preds = np.where(self.val_preds_mul < 1, False, True)
            predicted_class = [
                self.defect_types[val_pred] for val_pred in self.val_preds_mul
            ]
            prob_df = pd.DataFrame(pred_scores, columns=self.defect_types)

        # 識別結果テーブルの作成
        pred_df = pd.DataFrame(
            {
                "case_id": case_id,
                "image_id": self.image_id,
                "actual_defect_type": self.val_original_targets,
                "actual_class": self.val_targets,
                "predicted_class": predicted_class,
            }
        )
        self.pred_df = pd.concat([pred_df, prob_df], axis=1)

    def set_threshold(self, thr: float):
        self.val_preds = self.val_scores > thr
        predicted_class = [self.class_names[val_pred] for val_pred in self.val_preds]
        self.pred_df["predicted_class"] = predicted_class

    def save_prediction(self, save_dir: Path | None = None):
        if save_dir is None:
            save_dir = self.save_dir
        self.pred_df.sort_values(by="image_id", ignore_index=True).to_csv(
            save_dir / "prediction.csv", index=False
        )

    def save_metrics(self, save_dir: Path | None = None):
        if save_dir is None:
            save_dir = self.save_dir
        # 基本的な評価指標を入力
        tp, fp, tn, fn, _ = binary_stat_scores(
            torch.tensor(self.val_preds), torch.tensor(self.val_targets_bin)
        )
        fp_rate = (fp / (fp + tn)).numpy()
        fn_rate = (fn / (fn + tp)).numpy()
        accuracy = ((tp + tn) / (tp + fp + tn + fn)).numpy()
        if self.is_binary:
            auroc = float(
                BinaryAUROC()(
                    torch.tensor(self.val_scores_raw), torch.tensor(self.val_targets_bin)
                )
            )
        else:
            auroc = None
        self.metric_df = pd.DataFrame(
            {
                "case_id": [self.case_id],
                "accuracy": accuracy,
                "rocauc": auroc,
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
            }
        )
        # 各クラスの評価指標を入力
        # normal_counts_raw = np.unique(
        #     self.val_original_targets[~self.val_preds], return_counts=True
        # )
        # normal_counts_dict = dict(zip(*normal_counts_raw))
        # normal_counts = [
        #     normal_counts_dict[defect_type] if defect_type in normal_counts_dict else 0
        #     for defect_type in self.defect_types
        # ]
        # all_counts = np.unique(self.val_original_targets, return_counts=True)[1]
        # fn_values = (normal_counts / all_counts)[1:]
        normal_counts = Counter(self.val_original_targets[~self.val_preds])
        all_counts = Counter(self.val_original_targets)
        fn_values = [normal_counts[defect_type]/all_counts[defect_type] if all_counts[defect_type] != 0 else 0  for defect_type in self.defect_types]
        self.metric_df[
            [
                defect_type + "_fn_rate"
                for defect_type in self.defect_types
            ]
        ] = fn_values
        # 処理時間を入力
        self.metric_df[list(self.timer.keys())] = list(self.timer.values())
        # 保存
        self.metric_df.to_csv(save_dir / "result.csv", index=False)

    def save_roc(self, save_dir: Path | None = None):
        if save_dir is None:
            save_dir = self.save_dir
        fig, ax = plt.subplots(1, figsize=(5, 5))
        metric = BinaryROC()
        metric.update(torch.tensor(self.val_scores_raw), torch.tensor(self.val_targets_bin))
        metric.plot(score=True, ax=ax)
        fig.savefig(save_dir / "roc.png")
        plt.clf()

    def save_confmat(self, save_dir: Path | None = None):
        if save_dir is None:
            save_dir = self.save_dir
        if not self.is_binary:
            y_true = self.val_targets_mul
            y_pred = self.val_preds_mul
            figsize = (10, 10)
        else:
            y_true = self.val_targets_bin
            y_pred = self.val_preds
            figsize = (5, 5)
        confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        norm_confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")
        for filename, target in zip(
            ["confmat", "confmat_norm"], [confmat, norm_confmat]
        ):
            fig, ax = plt.subplots(1, figsize=figsize)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=target,
                display_labels=[chr(ord("A") + i) for i in range(target.shape[0])],
            )
            disp.plot(cmap="Blues", ax=ax)
            plt.tight_layout()
            fig.savefig(save_dir / f"{filename}.png")

    def save_table(self, save_dir: Path | None = None):
        if save_dir is None:
            save_dir = self.save_dir
        # normal_counts_raw = np.unique(
        #     self.val_original_targets[~self.val_preds], return_counts=True
        # )
        # normal_counts_dict = dict(zip(*normal_counts_raw))
        # normal_counts = [
        #     normal_counts_dict[defect_type] if defect_type in normal_counts_dict else 0
        #     for defect_type in self.defect_types
        # ]
        # defect_counts_raw = np.unique(
        #     self.val_original_targets[self.val_preds], return_counts=True
        # )
        # defect_counts_dict = dict(zip(*defect_counts_raw))
        # defect_counts = [
        #     defect_counts_dict[defect_type] if defect_type in defect_counts_dict else 0
        #     for defect_type in self.defect_types
        # ]
        # norm_normal_counts = np.array(normal_counts) / (
        #     np.array(normal_counts) + np.array(defect_counts)
        # )
        # norm_defect_counts = np.array(defect_counts) / (
        #     np.array(normal_counts) + np.array(defect_counts)
        # )
        normal_counts = Counter(self.val_original_targets[~self.val_preds])
        defect_counts = Counter(self.val_original_targets[self.val_preds])
        all_counts = Counter(self.val_original_targets)
        norm_normal_counts = [normal_counts[defect_type]/(all_counts[defect_type]) if all_counts[defect_type] != 0 else 0  for defect_type in self.defect_types]
        norm_defect_counts = [defect_counts[defect_type]/(all_counts[defect_type]) if all_counts[defect_type] != 0 else 0  for defect_type in self.defect_types]
        columns = [chr(ord("A") + i) for i in range(len(self.defect_types))]
        class_df = pd.DataFrame(
            [[normal_counts[defect_type] for defect_type in self.defect_types],
            [defect_counts[defect_type] for defect_type in self.defect_types]],
            index=["normal", "defect"], columns=columns
        ).T
        norm_class_df = pd.DataFrame(
            [norm_normal_counts, norm_defect_counts],
            index=["normal", "defect"],
            columns=columns,
        ).T
        dfi.export(class_df.style.background_gradient(), save_dir / "table.png")
        dfi.export(
            norm_class_df.style.format("{:3f}").background_gradient(),
            save_dir / "table_norm.png",
        )

    def save_all(self, save_dir: Path | None = None):
        self.save_prediction(save_dir)
        self.save_metrics(save_dir)
        if self.is_binary:
            self.save_roc(save_dir)
        self.save_confmat(save_dir)
        self.save_table(save_dir)
