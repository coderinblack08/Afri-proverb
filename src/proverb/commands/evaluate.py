import sys


sys.path.append("./src")


from proverb.engine.args import _parse_args
from proverb.data.loader import load_proverb_dataset
from proverb.data.collators import ProverbDataCollator
from proverb.engine.trainer import CustomTrainer
from proverb.engine.metrics import TranslateMetric

from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from math import exp


def main():
    model_args, training_args, data_args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
    )

    dataset = load_proverb_dataset(tokenizer, data_args, training_args)
    data_collator = ProverbDataCollator(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=TranslateMetric(tokenizer=tokenizer),
    )

    result = trainer.evaluate(dataset)
    if training_args.local_process_index == 0:
        print(result)


if __name__ == "__main__":
    main()
