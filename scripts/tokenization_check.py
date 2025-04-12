from datasets import load_from_disk

dataset = load_from_disk("../data/processed/tokenized_dataset")
print(dataset)
print(dataset[0])
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Decode the first example:
decoded_text = tokenizer.decode(dataset[0]['input_ids'], skip_special_tokens=True)
print(decoded_text)

