from transformers.trainer_utils import EvalPrediction
from typing import Dict


class MultiChoiceMetrics:
    def __init__(self):
        pass

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        losses = pred.losses
        logis = pred.predictions  # [100, 95, 262208]
        labels = pred.label_ids  # [100, 95]
        return {}
