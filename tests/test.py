from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

out = tokenizer.tokenize("F")
print(out)
