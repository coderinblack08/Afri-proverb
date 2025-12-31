import sys

sys.path.append("./src")


import typer
from proverb.engine.args import parse_args_from_yaml
from proverb.data.dataset import ProverbDataset
from proverb.data.collators import DataCollator

from transformers import AutoTokenizer
from transformers.trainer import Trainer
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration


def main(
    config_file: str = "configs/default.yaml",
):
    model_args, training_args, data_args = parse_args_from_yaml(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
    )

    dataset = ProverbDataset(tokenizer, data_args.dataset_dir)
    data_collator = DataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(main)
