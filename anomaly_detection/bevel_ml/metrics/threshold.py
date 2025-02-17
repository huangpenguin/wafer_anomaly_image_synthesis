"""
"""
import logging
import torch

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from anomalib.metrics.threshold import BaseThreshold

logger = logging.getLogger(__name__)


class ManualRecallThreshold(BinaryPrecisionRecallCurve, BaseThreshold):
    """anomalib.metrics.threshold.F1AdaptiveThresholdを修正し、Recallを指定できるようにしたもの。
    - 指定したmanual_recallの値を最初に越えるthreshold。
    """

    def __init__(
            self, manual_recall: float, default_value: float = 0.5,
            verbose:bool=False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

        self.manual_recall = manual_recall
        self.verbose = verbose

    def compute(self) -> torch.Tensor:
        """Compute the threshold that yields the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        if not any(1 in batch for batch in self.target):
            msg = (
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logging.warning(msg)

        precision, recall, thresholds = super().compute()
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            # manual_recallの値を最初に越えるthresholdを見付ける
            # FIXME: 同じrecallの値が連続する場合は、それを満たす"最大"のthresholdを選択したい。
            #        しかし、探索ロジックがあまいため、それを満たす"最小"のthresholdをみつけてしまう。
            sorted_recall, indices = torch.sort(recall)
            near_match_sorted_index = torch.searchsorted(sorted_recall, self.manual_recall, side="right")
            near_match_index = indices[near_match_sorted_index]
            self.value = thresholds[near_match_index]
        
        if self.verbose:
            print(f"recall={recall}")
            print(f"thresholds={thresholds}")
        return self.value

    def __repr__(self) -> str:
        """Return threshold value within the string representation.

        Returns:
            str: String representation of the class.
        """
        return f"{super().__repr__()} (value={self.value:.2f})"
  