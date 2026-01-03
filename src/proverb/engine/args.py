from dataclasses import dataclass

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.hf_argparser import HfArgumentParser, DataClass
from trl import TrlParser
from typing import Tuple, Literal


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
    template_name: str = "gemma"
    location: str = "Kenya"
    language: str = "nubian"
    task_type: Literal[
        "gen_swa_literal", "gen_eng_literal", "gen_swa_fig", "gen_eng_fig"
    ] = "gen_eng_literal"

    override_cache: bool = False
    processing_num_workers: int = 4


def _parse_args() -> Tuple[DataClass, ...]:
    parser = TrlParser(
        dataclass_types=(ModelArguments, Seq2SeqTrainingArguments, DataArguments)
    )
    model_args, training_args, data_args = parser.parse_args_and_config()
    return model_args, training_args, data_args
