from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)
from datasets import load_dataset
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

import os
from ..engine.args import DataArguments
from .processor import Processor
import re


def load_proverb_dataset(
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
):
    file_path = os.path.join(
        data_args.dataset_dir,
        data_args.location,
        f"{data_args.language}_prov.csv",
    )
    dataset = load_dataset("csv", data_files=file_path)["train"]
    processor = Processor(tokenizer, data_args)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        process_args = {
            "load_from_cache_file": (not data_args.override_cache)
            or training_args.local_process_index != 0,
            "num_proc": data_args.processing_num_workers,
            "desc": "Running processor on dataset",
        }
        dataset = dataset.map(
            processor, remove_columns=dataset.column_names, **process_args
        )
    return dataset


# class ProverbDataset(Dataset):
#     def __init__(self, tokenizer: PreTrainedTokenizerBase, data_args: DataArguments):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.data_args = data_args
#
#     def load_csv(self):
#         file_path = os.path.join(
#             self.data_args.data_dir,
#             self.data_args.location,
#             f"{self.data_args.language}_prov.csv",
#         )
#         return load_dataset("csv", data_files=file_path)["train"]
#
#     def __getitem__(self, idx):


#
# class ProverbDataset(Dataset):
#     def __init__(self, tokenizer: PreTrainedTokenizerBase, data_file: str):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.dataframe = read_csv(data_file)
#
#     def __getitem__(self, idx):
#         data = self.dataframe.rows(named=True)[idx]
#         prompt = GENERATE_PROMPT.format(
#             source_language="Swahili",
#             target_language="English",
#             proverb=data["Swahili"],
#         )
#         label = data["Figurative Meaning"] + self.tokenizer.eos_token
#
#         message = [
#             {"role": "user", "content": prompt},
#         ]
#
#         message = self.tokenizer.apply_chat_template(
#             message, add_generation_prompt=True, tokenize=False
#         )
#
#         return dict(
#             message=message,
#             label=label,
#         )
#
#     def __len__(self):
#         return len(self.dataframe)
