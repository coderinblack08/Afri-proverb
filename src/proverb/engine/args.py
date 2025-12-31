from dataclasses import dataclass

from transformers.training_args import TrainingArguments
from transformers.hf_argparser import HfArgumentParser, DataClass
from typing import Tuple


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = "google/gemma-3-4b-it"


@dataclass
class DataArguments:
    """
    Arguments pertaining to data input for training and evaluation.
    """

    dataset_dir: str = ""


def parse_args_from_yaml(yaml_file: str) -> Tuple[DataClass, ...]:
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_yaml_file(yaml_file=yaml_file)
    return model_args, training_args, data_args
