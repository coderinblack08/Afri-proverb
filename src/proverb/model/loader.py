import torch
from ..engine.args import ModelArguments


from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from transformers import AutoModelForCausalLM


def load_model(model_args: ModelArguments) -> torch.nn.Module:
    """
    Load model from pretrained checkpoint.
    """

    if "gemma" in model_args.model_name_or_path:
        if "1b" in model_args.model_name_or_path:
            model = Gemma3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                # attn_implementation="sdpa",
                # low_cpu_mem_usage=True,
            )
        else:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                # attn_implementation="sdpa",
            )
    elif "Qwen" in model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
    elif "Mistral" in model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_args.model_name_or_path}")

    return model
