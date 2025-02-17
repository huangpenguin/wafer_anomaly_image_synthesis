import logging
from collections.abc import Sequence
from typing import Any, Callable, override

import torch
from anomalib.models.components import MemoryBankMixin
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, Transform

from bevel_ml.models.base import BaseDetector

logger = logging.getLogger(__name__)


class PatchcoreDetector(MemoryBankMixin, BaseDetector):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``wide_resnet50_2``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer2", "layer3"]``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.
            Defaults to ``0.1``.
        num_neighbors (int, optional): Number of nearest neighbors.
            Defaults to ``9``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        transform: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__(transform=transform)

        self.model: PatchcoreModel = PatchcoreModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []

    @override
    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        embedding = self.model(x)
        self.embeddings.append(embedding)

    @override
    def fit(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch

        # Get anomaly maps and predicted scores from the model.
        output_of_model = self.model(x)

        # outputsを作成
        outputs = {
            "loss": [None],
            "anomaly_maps": output_of_model["anomaly_map"],
            "pred_scores": output_of_model["pred_score"],
            "label": y,
        }

        return outputs

    @override
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx)

    # DEBUG: AnomalyModuleで定義されているメソッド。使っていないが、設定値の参考として残した。
    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Patchcore trainer arguments."""

        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    # DEBUG: AnomalyModuleで定義されているメソッド。使っていないが、設定値の参考として残した。
    def configure_transforms(
        self, image_size: tuple[int, int] | None = None
    ) -> Transform:
        """Default transform for Padim."""
        image_size = image_size or (256, 256)
        # scale center crop size proportional to image size
        height, width = image_size
        center_crop_size = (int(height * (224 / 256)), int(width * (224 / 256)))
        return Compose(
            [
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
