import torch
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from .constants import IGNORE_INDEX


@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.tokenizer.padding_side == "left", (
            "Tokenizer padding side must be left for causal language modeling."
        )

        batch_inputs = [feature["message"] + feature["label"] for feature in features]
        batch_labels = [feature["label"] for feature in features]
        lengths = [
            len(ids)
            for ids in self.tokenizer(
                batch_labels, padding=False, truncation=False, add_special_tokens=False
            )["input_ids"]
        ]

        batch = self.tokenizer(
            batch_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        batch["labels"] = batch["input_ids"].clone()
        for i, length in enumerate(lengths):
            batch["labels"][i, :-length] = IGNORE_INDEX

        # position ids
        attention_mask = batch["attention_mask"]
        position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        batch["position_ids"] = position_ids

        return batch
