from collections.abc import Sequence
from typing import Any, Callable, override

import torch
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.models.image.reverse_distillation.anomaly_map import (
    AnomalyMapGenerationMode,
)
from anomalib.models.image.reverse_distillation.loss import ReverseDistillationLoss
from anomalib.models.image.reverse_distillation.torch_model import (
    ReverseDistillationModel,
)
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from bevel_ml.models.base import BaseDetector


class ReverseDistillationDetector(BaseDetector):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        backbone (str): Backbone of CNN network
            Defaults to ``wide_resnet50_2``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer1", "layer2", "layer3"]``.
        anomaly_map_mode (AnomalyMapGenerationMode, optional): Mode to generate anomaly map.
            Defaults to ``AnomalyMapGenerationMode.ADD``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        anomaly_map_mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.ADD,
        pre_trained: bool = True,
        transform: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__(transform=transform)

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.layers = layers
        self.anomaly_map_mode = anomaly_map_mode
        self.model: ReverseDistillationModel
        self.loss = ReverseDistillationLoss()

        if self.input_size is None:
            msg = "Input size is required for Reverse Distillation model."
            raise ValueError(msg)

        self.model = ReverseDistillationModel(
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            layers=self.layers,
            input_size=self.input_size,
            anomaly_map_mode=self.anomaly_map_mode,
        )

    @override
    def configure_optimizers(self) -> optim.Adam:
        """Configure optimizers for decoder and bottleneck.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=list(self.model.decoder.parameters())
            + list(self.model.bottleneck.parameters()),
            lr=0.005,
            betas=(0.5, 0.99),
        )

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        loss = self.loss(*self.model(x))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        anomaly_maps = self.model(x)

        # outputsを作成
        outputs = {
            "loss": [None],
            "anomaly_maps": anomaly_maps,
            "label": y,
        }

        # 異常度マップを追加
        _PostProcessorCallback._post_process(outputs)

        return outputs

    @override
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx)

    # DEBUG: AnomalyModuleで定義されているメソッド。使っていないが、設定値の参考として残した。
    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Reverse Distillation trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
