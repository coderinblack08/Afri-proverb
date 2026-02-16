from datasets import load_dataset


dataset = load_dataset("sartifyllc/east_africa_language")["test"]

print(len(dataset))
