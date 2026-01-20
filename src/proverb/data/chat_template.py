from transformers import PreTrainedTokenizerBase
from typing import Optional


class Template:
    def __init__(
        self,
        task_type: str,
        tokenizer: PreTrainedTokenizerBase,
        add_generation_prompt: bool = True,
        extra_eos_token: bool = False,
        eos_token: Optional[str] = None,
        extra_bos_token: bool = False,
        bos_token: Optional[str] = None,
    ):
        """
        prompt -> message -> input_ids, attention_mask, labels
        """
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.eos_token = eos_token if eos_token is not None else tokenizer.eos_token
        self.bos_token = bos_token if bos_token is not None else tokenizer.bos_token

        self.extra_eos_token = extra_eos_token
        self.extra_bos_token = extra_bos_token
        self.add_generation_prompt = add_generation_prompt

    def construct_message(self, prompt: str) -> str:
        return [
            {"role": "user", "content": prompt},
        ]

    def format_prompt(self, prompt: str) -> str:
        message = self.construct_message(prompt)
        prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=self.add_generation_prompt,
            tokenize=False,
            enable_thinking=False,
        )
        if self.extra_bos_token:
            prompt = self.bos_token + prompt
        return prompt

    def format_label(self, label: str) -> str:
        if self.extra_eos_token:
            label += self.eos_token
        return label

    def __call__(self, prompt: str, label: str) -> dict[str, str]:
        formatted_prompt = self.format_prompt(prompt)
        formatted_label = self.format_label(label)
        return dict(
            prompt=formatted_prompt,
            label=formatted_label,
        )


def get_template(
    name: str, task_type: str, tokenizer: PreTrainedTokenizerBase
) -> Template:
    if name == "gemma":
        return Template(
            task_type=task_type,
            tokenizer=tokenizer,
            add_generation_prompt=True,
            extra_eos_token=True,
        )
    elif name == "qwen3":
        return Template(
            task_type=task_type,
            tokenizer=tokenizer,
            add_generation_prompt=True,
            extra_eos_token=False,
        )
    elif name == "mistral":
        return Template(
            task_type=task_type,
            tokenizer=tokenizer,
            add_generation_prompt=True,
            extra_eos_token=False,
        )
    else:
        raise ValueError(f"Unknown template name: {name}")
