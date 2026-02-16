from transformers.tokenization_mistral_common import MistralCommonTokenizer
from transformers import Mistral3ForConditionalGeneration


tokenizer = MistralCommonTokenizer.from_pretrained(
    "mistralai/Ministral-3-8B-Instruct-2512"
)
# processor = AutoProcessor.from_pretrained("mistralai/Ministral-3-14B-Instruct-2512")
#
model = Mistral3ForConditionalGeneration.from_pretrained(
    "mistralai/Ministral-3-8B-Instruct-2512"
)

message = [
    {"role": "user", "content": "Hello world!"},
    {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    {"role": "user", "content": "Can you tell me a joke?"},
]

print(tokenizer.pad_token_id)

print(
    tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
)
