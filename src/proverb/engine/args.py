from dataclasses import dataclass, field

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.hf_argparser import HfArgumentParser, DataClass
from trl import TrlParser
from typing import Tuple, Literal, List


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

    dataset_dir: str = "datasets/proverbs"
    template_name: str = "gemma"
    location: str = "Kenya"
    language: str = "nubian"

    override_cache: bool = False
    processing_num_workers: int = 4

    few_shot_num: int = 0

    def __post_init__(self):
        parsed_location = [loc.strip() for loc in self.location.split(",")]
        parsed_language = [lang.strip() for lang in self.language.split(";")]

        assert len(parsed_location) == len(parsed_language), (
            "The number of locations must match the number of language groups."
        )

        self.location_langauge_paris = [
            (
                parsed_location[i],
                tuple(lang.strip() for lang in parsed_language[i].split(",")),
            )
            for i in range(len(parsed_location))
        ]


@dataclass
class TaskArguments:
    """
    Arguments specific to the proverb generation task.
    """

    task_class: Literal["train", "eval"] = "eval"
    task_type: Literal[
        "gen_swa_literal", "gen_eng_literal", "gen_swa_fig", "gen_eng_fig"
    ] = "gen_eng_literal"


def _parse_args() -> Tuple[
    ModelArguments, Seq2SeqTrainingArguments, DataArguments, TaskArguments
]:
    parser = TrlParser(
        dataclass_types=(
            ModelArguments,
            Seq2SeqTrainingArguments,
            DataArguments,
            TaskArguments,
        )
    )
    model_args, training_args, data_args, task_args = parser.parse_args_and_config()
    return model_args, training_args, data_args, task_args
