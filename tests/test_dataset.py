import sys

sys.path.append("./src")

from transformers import AutoTokenizer
from proverb.data.collators import ProverbDataCollator
from proverb.engine.args import DataArguments, ModelArguments, TaskArguments
from proverb.data.loader import load_proverb_dataset
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import torch


data_args = DataArguments(
    dataset_dir="dataset/African-Proverbs/Data",
    template_name="mistral",
    location="Kenya, Ethiopia",
    language="nubian, maasai, gikuyu, ekegusii; borana",
    override_cache=False,
    processing_num_workers=1,
    few_shot_num=2,
)

task_args = TaskArguments(
    task_type="gen_eng_literal",
)


training_args = Seq2SeqTrainingArguments()


def test_dataset_loading():
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    loaded_datasets = load_proverb_dataset(
        tokenizer, data_args, training_args, task_args
    )

    dataset = loaded_datasets[0]["dataset"]

    for item in dataset:
        print("INPUT:\n")
        print(tokenizer.decode(item["input_ids"]))
        print("\nLABEL:\n")
        print(tokenizer.decode(item["label"]))
        print("\nINPUT IDS:\n")
        print(item["input_ids"])
        print("LABEL IDS:\n")
        print(item["label"])

    # collator = ProverbDataCollator(tokenizer=tokenizer)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    # for batch in loader:
    #     print(batch)


def test_data_collator():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = load_proverb_dataset(tokenizer, data_args, training_args, task_args)[0][
        "dataset"
    ]

    collator = ProverbDataCollator(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in loader:
        print(batch)


if __name__ == "__main__":
    test_dataset_loading()
    # test_data_collator()
