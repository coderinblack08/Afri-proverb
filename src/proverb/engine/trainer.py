from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing import Optional, Tuple


class CustomTrainer(Trainer):
    def __init__(
        self, multi_choices_index: Optional[Tuple[int]] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.multi_choices_index = multi_choices_index
