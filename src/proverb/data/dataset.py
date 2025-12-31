from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)

from torch.utils.data import Dataset
from polars import read_csv
from .constants import GENERATE_PROMPT


class ProverbDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_file: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataframe = read_csv(data_file)

    def __getitem__(self, idx):
        data = self.dataframe.rows(named=True)[idx]
        prompt = GENERATE_PROMPT.format(
            source_language="Swahili",
            target_language="English",
            proverb=data["Swahili"],
        )
        label = data["Figurative Meaning"] + self.tokenizer.eos_token

        message = [
            {"role": "user", "content": prompt},
        ]

        message = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )

        return dict(
            message=message,
            label=label,
        )

    def __len__(self):
        return len(self.dataframe)
