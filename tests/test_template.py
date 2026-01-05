from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
print(tokenizer.bos_token_id)

messages = [
    {
        "role": "user",
        "content": "Translate the following English text to French: 'Hello, how are you?'",
    },
]


text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
)

inputs = tokenizer([text], return_tensors="pt")
print(inputs)
