import torch
from ..engine.args import ModelArguments


from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration


def load_model(model_args: ModelArguments) -> torch.nn.Module:
    """
    Load model from pretrained checkpoint.
    """

    if "gemma" in model_args.model_name_or_path:
        model = Gemma3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # attn_implementation="sdpa",
        )
    else:
        raise ValueError(f"Unknown model type: {model_args.model_type}")

    return model
