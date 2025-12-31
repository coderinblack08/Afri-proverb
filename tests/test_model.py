import sys

import torch

sys.path.append("./src")

from proverb.data.dataset import ProverbDataset
from transformers import AutoTokenizer
from proverb.data.collators import DataCollator

from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3ForCausalLM


def test_gemma3_model_forward():
    model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it").to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = ProverbDataset(tokenizer, "dataset/Hemba_pov - hemba.csv")
    collator = DataCollator(tokenizer)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in loader:
        for key in batch:
            batch[key] = batch[key].to("cuda")
        outputs = model(**batch)
        print(outputs)
        break


if __name__ == "__main__":
    test_gemma3_model_forward()
