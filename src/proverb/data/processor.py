from ..engine.args import DataArguments, ModelArguments
from .prompts import get_prompt_by_task
from .chat_template import get_template
from transformers import PreTrainedTokenizer


class Processor:
    """
    data -> prompt -> message -> input_ids, attention_mask, labels
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, example):
        source_column = self._get_source_column_name()
        label_column = self._get_label_column_name()

        source = example[source_column]
        label = example[label_column]

        prompt = get_prompt_by_task(
            task_type=self.data_args.task_type,
            source_language=self.data_args.language.lower(),
            proverb=source,
        )

        chat_template = get_template(
            self.data_args.template_name,
            self.data_args.task_type,
            self.tokenizer,
        )

        prompt, label = chat_template(prompt, label).values()

        if "gen" in self.data_args.task_type:
            return dict(
                input_ids=self.tokenizer.encode(prompt, add_special_tokens=False),
                label=self.tokenizer.encode(label, add_special_tokens=False),
            )
        else:
            raise ValueError(f"Unknown task type: {self.data_args.task_type}")

    def _get_source_column_name(self):
        return f"{self.data_args.language}"

    def _get_label_column_name(self):
        if self.data_args.task_type == "gen_swa_literal":
            return "swa_literal"
        elif self.data_args.task_type == "gen_eng_literal":
            return "eng_literal"
        elif self.data_args.task_type == "gen_swa_fig":
            return "swa_figurative"
        elif self.data_args.task_type == "gen_eng_fig":
            return "eng_figurative"
