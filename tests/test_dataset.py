import sys

sys.path.append("./src")

from transformers import AutoTokenizer
from proverb.data.collators import ProverbDataCollator
from proverb.engine.args import DataArguments, ModelArguments
from proverb.data.loader import load_proverb_dataset
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import torch


data_args = DataArguments(
    dataset_dir="dataset/African-Proverbs/Data",
    template_name="gemma",
    location="Kenya",
    language="maasai",
    task_type="gen_eng_literal",
    override_cache=False,
    processing_num_workers=1,
)

training_args = Seq2SeqTrainingArguments()


def test_dataset_loading():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = load_proverb_dataset(tokenizer, data_args, training_args)

    for item in dataset:
        print("INPUT:\n")
        print(tokenizer.decode(item["input_ids"]))
        print("\nLABEL:\n")
        print(tokenizer.decode(item["label"]))
        print("\nINPUT IDS:\n")
        print(item["input_ids"])
        print("LABEL IDS:\n")
        print(item["label"])

    collator = ProverbDataCollator(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in loader:
        print(batch)


def test_data_collator():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = load_proverb_dataset(tokenizer, data_args, training_args)

    collator = ProverbDataCollator(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in loader:
        print(batch)


if __name__ == "__main__":
    # test_dataset_loading()
    test_data_collator()
