from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# processor = AutoProcessor.from_pretrained("mistralai/Ministral-3-14B-Instruct-2512")

message = [
    {"role": "user", "content": "Hello world!"},
    {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    {"role": "user", "content": "Can you tell me a joke?"},
]
print(
    tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
)
