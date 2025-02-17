

from copy import deepcopy
from pathlib import Path
from shutil import copy
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anomalib.callbacks.normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.metrics import MinMax
from anomalib.metrics.threshold import BaseThreshold
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchmetrics import Metric

from bevel_ml.metrics.threshold import ManualRecallThreshold
from bevel_ml.models.base import BaseDetector


def _outputs_to_cpu(output: STEP_OUTPUT) -> STEP_OUTPUT | dict[str, Any]:
    """outputをコピーして、CPUに転送して返す。
    """
    if isinstance(output, dict):
        new_output = dict()
        for key, value in output.items():
            new_output[key] = _outputs_to_cpu(value)
    elif isinstance(output, torch.Tensor):
        new_output = output.detach().cpu()
    else:
        new_output = deepcopy(output)
    return new_output


class MetricsContainer(LightningModule):
    """Anomalibの_PostProcessorCallback,_MinMaxNormalizationCallbackから
    呼び出されるための代替LightningModule。

    Note:
        - Thresholding,Normalizationのロジックを以下からコピーして修正した。
            - anomalib.callbacks.thresholding._ThresholdCallback
            - anomalib.callbacks.normalization._MinMaxNormalizationCallback
        - HookはDetectionResultCallback側に実装した。
        - 現状、必ずimage_threshold=pixel_thresholdとして動く実装になっている。
    TODO:
        - anomaly_mapとpred_scoresで閾値を分けるかどうか検討する。
    """
    def __init__(self, threshold: BaseThreshold, normalization_metrics: Metric):
        super().__init__()
        self.image_threshold = threshold
        self.pixel_threshold = threshold.clone()
        self.normalization_metrics = normalization_metrics
    
    def reset_threshold(self) -> None:
        self.image_threshold.reset()
        self.pixel_threshold.reset()

    def update_threshold(self, outputs: STEP_OUTPUT) -> None:
        self.image_threshold.cpu()
        self.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs and "anomaly_maps" in outputs:
            # DEBUG: 現状、ここが呼ばれることはない。
            #        anomaly_mapとpred_scoresで閾値を分けたい場合はここのロジックが使えるのではないか？
            self.pixel_threshold.cpu()
            self.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())
    
    def compute_threshold(self) -> None:
        self.image_threshold.compute()
        if self.pixel_threshold._update_called:  # noqa: SLF001
            # DEBUG: 現状、ここが呼ばれることはない。
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value
    
    def update_normalization_metrics(self, outputs: STEP_OUTPUT) -> None:
        if "anomaly_maps" in outputs:
            # DEBUG: PatchCoreDetector, ReverseDistillationDetectorではここが呼ばれる。
            self.normalization_metrics(outputs["anomaly_maps"])
        elif "box_scores" in outputs:
            # DEBUG: 現状、ここが呼ばれることはない。
            self.normalization_metrics(torch.cat(outputs["box_scores"]))
        elif "pred_scores" in outputs:
            # DEBUG: TimmDetectorではここが呼ばれる。
            self.normalization_metrics(outputs["pred_scores"])
        else:
            msg = "No values found for normalization, provide anomaly maps, bbox scores, or image scores"
            raise ValueError(msg)


class DetectionResultCallbackExample(Callback):
    """異常検知結果を計算、出力するCallbackクラス。
    - validation時に識別の閾値を計算する。
    - test時に出力を正規化して蓄積する。
    - Hookで受け取ったoutputsの参照先は直接変更せず、コピーして使う。
    """     
    def __init__(self,
        threshold: BaseThreshold = ManualRecallThreshold(0.8),
        normalization_metrics: Metric = MinMax().cpu()
    ):
        self._metrics_container = MetricsContainer(threshold, normalization_metrics)
        self._predictions = []
    
    @property
    def image_threshold(self) -> BaseThreshold:
        return self._metrics_container.image_threshold
    
    @property
    def pixel_threshold(self) -> BaseThreshold:
        return self._metrics_container.pixel_threshold
    
    @property
    def normalization_metrics(self) -> Metric:
        return self._metrics_container.normalization_metrics
    

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: BaseDetector) -> None:

        self._metrics_container.reset_threshold()

    def on_validation_batch_end(
        self,
        trainer: Trainer,

        pl_module: BaseDetector,

        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.
        if outputs is not None:
            outputs = _outputs_to_cpu(outputs)
            self._metrics_container.update_threshold(outputs)
            self._metrics_container.update_normalization_metrics(outputs)


    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseDetector) -> None:

        self._metrics_container.compute_threshold()
    

    def on_test_start(self, trainer: Trainer, pl_module: BaseDetector) -> None:

        self._predictions.clear()
    
    def on_test_batch_end(
        self, 
        trainer: Trainer, 

        pl_module: BaseDetector, 

        outputs: STEP_OUTPUT, 
        batch, 
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        outputs = _outputs_to_cpu(outputs)

        # 予測ラベルとバウンディングボックスを追加
        # TODO: ここでpred_boxes,pred_masks等も追加される。これらも必要かどうか要検討。
        _PostProcessorCallback._compute_scores_and_labels(self._metrics_container, outputs)

        # 正規化する
        _MinMaxNormalizationCallback._normalize_batch(outputs, self._metrics_container)

        # TODO: ここで蓄積することによりメモリを圧迫しないか要確認。anomaly_mapsのサイズが気になる。
        self._predictions.append(outputs)
    

    def on_test_end(self, trainer: Trainer, pl_module: BaseDetector) -> None:

        image_ids = trainer.datamodule.test_data.image_ids
        
        labels = np.concatenate([prediction["label"] for prediction in self._predictions])
        pred_scores = np.concatenate([prediction["pred_scores"] for prediction in self._predictions])
        pred_labels = np.concatenate([prediction["pred_labels"] for prediction in self._predictions])
        if "anomaly_maps" in self._predictions[0].keys():
            anomaly_maps = np.concatenate([prediction["anomaly_maps"] for prediction in self._predictions])
        else:
            anomaly_maps = None
        
        # TODO: これ以降に目的のスコア計算、出力等を実装する。
        print()
        print("=========================================================================")
        print_df = pd.DataFrame({
            "image_ids": image_ids,
            "labels": labels,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
        })
        print(print_df)
        if anomaly_maps is not None:
            print(f"anomaly_maps={anomaly_maps.shape}")
        print("=========================================================================")
