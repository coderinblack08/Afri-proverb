from ..engine.args import DataArguments, ModelArguments, TaskArguments
from .prompts import get_prompt_by_task, get_few_shots_prompt_by_task
from .chat_template import get_template
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import Optional


class Processor:
    """
    data -> prompt -> message -> input_ids, attention_mask, labels
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        source_language: str,
        data_args: DataArguments,
        task_args: TaskArguments,
        dataset: Optional[Dataset] = None,
    ):
        self.tokenizer = tokenizer
        self.source_language = source_language

        self.data_args = data_args
        self.task_args = task_args

        self.is_few_shot = False
        if data_args.few_shot_num > 0:
            self.is_few_shot = True
            self._fetch_few_shot_examples(dataset)

    def __call__(self, example):
        source_column = self._get_source_column_name()
        label_column = self._get_label_column_name()

        source = example[source_column]
        label = example[label_column]

        if self.is_few_shot:
            prompt = get_few_shots_prompt_by_task(
                task_type=self.task_args.task_type,
                source_language=self.source_language,
                proverb=source,
                example_inputs=self.few_shot_inputs,
                example_outputs=self.few_shot_outputs,
            )
        else:
            prompt = get_prompt_by_task(
                task_type=self.task_args.task_type,
                source_language=self.source_language,
                proverb=source,
            )

        chat_template = get_template(
            self.data_args.template_name,
            self.task_args.task_type,
            self.tokenizer,
        )

        prompt, label = chat_template(prompt, label).values()

        if "gen" in self.task_args.task_type:
            ret = dict(
                input_ids=self.tokenizer.encode(prompt, add_special_tokens=False),
                label=self.tokenizer.encode(label, add_special_tokens=False),
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_args.task_type}")

        if (
            self.task_args.task_class == "eval"
            and "literal" in self.task_args.task_type
        ):
            ret["source"] = self.tokenizer.encode(source, add_special_tokens=False)

        return ret

    def _get_source_column_name(self):
        return f"{self.source_language}_prov"

    def _get_label_column_name(self):
        if self.task_args.task_type == "gen_swa_literal":
            return "swa_literal"
        elif self.task_args.task_type == "gen_eng_literal":
            return "eng_literal"
        elif self.task_args.task_type == "gen_swa_fig":
            return "swa_figurative"
        elif self.task_args.task_type == "gen_eng_fig":
            return "eng_figurative"
        else:
            raise ValueError(f"Unknown task type: {self.task_args.task_type}")

    def _fetch_few_shot_examples(self, dataset: Dataset):
        source_column = self._get_source_column_name()
        label_column = self._get_label_column_name()

        examples = dataset.select(range(self.data_args.few_shot_num))

        self.few_shot_inputs = [
            examples[i][source_column] for i in range(len(examples))
        ]
        self.few_shot_outputs = [
            examples[i][label_column] for i in range(len(examples))
        ]
