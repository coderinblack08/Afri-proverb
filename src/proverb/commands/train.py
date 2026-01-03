import sys

sys.path.append("./src")


from proverb.engine.args import _parse_args
from proverb.data.dataset import ProverbDataset

from transformers import AutoTokenizer
from transformers.trainer import Trainer
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration


def main():
    model_args, training_args, data_args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
    )

    dataset = ProverbDataset(tokenizer, data_args.dataset_dir)
    # data_collator = DataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
