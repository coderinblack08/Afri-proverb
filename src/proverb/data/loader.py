from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)
from datasets import load_dataset, Dataset
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from typing import List, Dict, Tuple
from pandas import read_csv

import os
from ..engine.args import DataArguments, TaskArguments
from .processor import Processor
from ..extras.misc import print_dataset_sample


def load_proverb_dataset(
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    task_args: TaskArguments,
) -> List[Dict]:
    ret = list()

    for loc, langs in data_args.location_langauge_paris:
        for lang in langs:
            file_path = os.path.join(
                data_args.dataset_dir,
                loc,
                f"{lang}_prov.csv",
            )
            csv_file = read_csv(file_path)
            csv_file.columns = csv_file.columns.str.strip()

            dataset = Dataset.from_pandas(csv_file)

            # dataset = load_dataset("csv", data_files=file_path)["train"]
            processor = Processor(tokenizer, lang, data_args, task_args)

            with training_args.main_process_first(desc="dataset map pre-processing"):
                process_args = {
                    "load_from_cache_file": (not data_args.override_cache)
                    or training_args.local_process_index != 0,
                    "num_proc": data_args.processing_num_workers,
                    "desc": f"Running processor on location: {loc}, language: {lang}",
                }
                dataset = dataset.map(
                    processor, remove_columns=dataset.column_names, **process_args
                )

            if training_args.local_process_index == 0:
                print(f"Sample from location: {loc}, language: {lang}\n")
                print_dataset_sample(tokenizer, dataset, num_samples=2)

            ret.append({"location": loc, "language": lang, "dataset": dataset})

    return ret
