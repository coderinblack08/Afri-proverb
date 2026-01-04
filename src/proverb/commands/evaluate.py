import sys


sys.path.append("./src")


from proverb.engine.args import _parse_args
from proverb.data.loader import load_proverb_dataset
from proverb.data.collators import ProverbDataCollator
from proverb.engine.trainer import CustomTrainer
from proverb.engine.metrics import TranslateMetric

from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from pprint import pprint


def main():
    model_args, training_args, data_args, task_args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        # attn_implementation="sdpa",
    )

    loaded_datasets = load_proverb_dataset(
        tokenizer, data_args, training_args, task_args
    )
    data_collator = ProverbDataCollator(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=TranslateMetric(tokenizer=tokenizer),
    )

    results = []
    for item in loaded_datasets:
        result = trainer.evaluate(eval_dataset=item["dataset"])
        results.append(
            {
                "task_type": task_args.task_type,
                "location": item["location"],
                "language": item["language"],
                "results": result,
            }
        )

    if training_args.local_process_index == 0:
        pprint(results)


if __name__ == "__main__":
    main()
