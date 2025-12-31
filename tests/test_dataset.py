import sys

sys.path.append("./src")

from proverb.data.dataset import ProverbDataset
from transformers import AutoTokenizer
from proverb.data.collators import DataCollator
import torch


def test_dataset_loading():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = ProverbDataset(tokenizer, "dataset/Hemba_pov - hemba.csv")
    collator = DataCollator(tokenizer)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in loader:
        print(batch)


if __name__ == "__main__":
    test_dataset_loading()
