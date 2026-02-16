from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments
from typing import Optional, Tuple
from typing_extensions import override
import torch
import numpy as np
import os
import json
from typing import Any, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.trainer import PredictionOutput
from ..data.constants import IGNORE_INDEX
from ..extras.logging import get_logger

logger = get_logger(__name__)


class CustomTrainer(Seq2SeqTrainer):
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        sources = inputs.pop("source", None)

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = (
                self.processing_class.pad_token_id
            )
            generated_tokens = generated_tokens.contiguous()

        if sources is not None:
            return loss, generated_tokens, (labels, sources)

        return loss, generated_tokens, labels

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
        file_name: str = "generated_predictions.jsonl",
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, file_name)
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        if isinstance(predict_results.label_ids, tuple):
            label_ids, source_ids = predict_results.label_ids
        else:
            label_ids = predict_results.label_ids
            source_ids = None

        labels = np.where(
            label_ids != IGNORE_INDEX,
            label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )

        decoded_inputs = self.processing_class.batch_decode(
            dataset["input_ids"], skip_special_tokens=False
        )
        decoded_preds = self.processing_class.batch_decode(
            preds, skip_special_tokens=skip_special_tokens
        )
        decoded_labels = self.processing_class.batch_decode(
            labels, skip_special_tokens=skip_special_tokens
        )

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(
                    json.dumps(
                        {"prompt": text, "predict": pred, "label": label},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
