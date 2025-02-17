from abc import ABC, abstractmethod
from typing import Any, Callable, override
import torch
from lightning.pytorch.trainer.states import TrainerFn
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class BaseDetector(LightningModule, ABC):
    def __init__(self, transform: Callable | None = None):
        super().__init__()
        self.transform = transform  # effective input sizeの計算用  HACK: できればここから削除したい
        self._is_setup = False  # flag to track if setup has been called from the trainer
    
    def setup(self, stage: str | None = None) -> None:
        """Calls the _setup method to build the model if the model is not already built."""
        if getattr(self, "model", None) is None or not self._is_setup:
            self._setup()
            if isinstance(stage, TrainerFn):
                # only set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
                self._is_setup = True
    
    def _setup(self) -> None:
        """The _setup method is used to build the torch model dynamically or adjust something about them.
        The model implementer may override this method to build the model. This is useful when the model cannot be set
        in the `__init__` method because it requires some information or data that is not available at the time of
        initialization.
        """
        pass
    
    
    def forward(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> Any:  # noqa: ANN401
        """Perform the forward-pass by passing input tensor to the module.
        
        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.
        Returns:
            Tensor: Output tensor from the model.
        """
        del args, kwargs  # These variables are not used.
        x, _ = batch
        
        return self.model(x)
    

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the effective input size of the model.
        The effective input size is the size of the input tensor after the transform has been applied. If the transform
        is not set, or if the transform does not change the shape of the input tensor, this method will return None.
        """
        transform = self.transform or self.configure_transforms()
        if transform is None:
            return None
        dummy_input = torch.zeros(1, 3, 1, 1)
        output_shape = transform(dummy_input).shape[-2:]
        if output_shape == (1, 1):
            return None
        return output_shape[-2:]
    
    @override
    @abstractmethod
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (torch.Tensor): 
            batch_idx (int): 
        Returns:
            STEP_OUTPUT: 
        """
        pass


    @override
    @abstractmethod
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (torch.Tensor): 
            batch_idx (int): 
        Returns:
            STEP_OUTPUT: 以下の要素を含む辞書。
                {
                    "loss": 損失 [必須にすべき？],
                    "anomaly_maps": ピクセル単位の異常度 [任意],
                    "pred_scores": 画像単位の予測スコア [必須],
                    "recons": ピクセル単位の再構成誤差 [任意],
                    "label": 正解ラベル [必須],
                }
            """
        pass
    
    
    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx)
    

