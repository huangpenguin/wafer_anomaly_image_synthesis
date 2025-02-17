import logging
from typing import override

import torch
from timm import create_model
from torch.nn.functional import softmax
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy, MulticlassRecall

from bevel_ml.models.base import BaseDetector

logger = logging.getLogger(__name__)


class TimmDetector(BaseDetector):
    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = False,
        num_classes: int = None,
        lr: float = None,
        weight: list = None,
        **kwargs,
    ) -> None:
        """ """

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = create_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            pretrained=self.hparams.pretrained,
        )
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.hparams.weight)

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.recall = MulticlassRecall(
            num_classes=self.hparams.num_classes, average=None
        )

    def forward(self, batch: torch.Tensor):
        x, _ = batch

        return self.net(x)

    def _shared_eval_step(self, x: torch.Tensor, y: torch.Tensor):
        outputs = self.net(x)
        loss = self.criterion(outputs, y)
        probs = softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        return loss, probs, preds

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        loss, _, preds = self._shared_eval_step(x, y)

        self.train_loss(loss)
        self.train_acc(preds, y)

        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        loss, probs, preds = self._shared_eval_step(x, y)
        pred_scores = probs[:, 1]  # 02_anomalyクラスの確率

        self.val_loss(loss)
        self.val_acc(preds, y)

        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # outputsを作成

        outputs = {
            "loss": [None],  # lossを入れておく？
            "pred_scores": pred_scores,
            "label": y,
            # 後でpred_labelを作るため、ここでpredsの予測ラベルを入れない。
        }

        return outputs

    @override
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch

        loss, probs, preds = self._shared_eval_step(x, y)

        if self.hparams.num_classes == 2:
            pred_scores = probs[:, 1]  # 02_anomalyクラスの確率

        else:
            pred_scores = probs

        # outputsを作成
        outputs = {
            "loss": [None],  # lossを入れておく？
            "pred_scores": pred_scores,
            "label": y,
            # 後でpred_labelを作るため、ここでpredsの予測ラベルを入れない。
        }

        return outputs

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.hparams.lr, eps=1e-6
        )

        return optimizer
