import torch
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from .constants import IGNORE_INDEX
from transformers.data.data_collator import DataCollatorForSeq2Seq


class ProverbDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer is not None, "Tokenizer must be provided."
        assert self.tokenizer.padding_side == "left", (
            "Tokenizer must use left padding for causal language modeling."
        )
        self.label_pad_token_id = IGNORE_INDEX

    def __call__(
        self,
        features: list[dict[str, Any]],
        return_tensors: Optional[str] = None,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        batch = super().__call__(features, return_tensors)

        # position ids
        attention_mask = batch["attention_mask"]
        position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        batch["position_ids"] = position_ids

        return batch
