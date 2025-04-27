# === Preprocessed logs tokenization script ===

import pandas as pd
import os
from transformers import BertTokenizer
from datasets import Dataset
from tqdm import tqdm

INPUT_CSV = "../data/processed_logs/processed_logs.csv"
OUTPUT_PATH = "../data/processed_logs/tokenized_dataset"

def tokenize_dataset(input_csv, output_path, tokenizer_name="bert-base-uncased", max_length=64):
    print(f"📥 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Drop missing or empty messages
    df = df.dropna(subset=["message"])
    df = df[df["message"].str.strip().astype(bool)]

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    dataset = Dataset.from_pandas(df[['message']])

    print("🔁 Tokenizing log messages...")
    dataset = dataset.map(
        lambda x: tokenizer(x['message'], padding='max_length', truncation=True, max_length=max_length),
        batched=True
    )

    dataset = dataset.remove_columns(['message'])
    dataset = dataset.map(lambda x: {**x, "labels": x["input_ids"]})

    # Set format for PyTorch training
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print(f"💾 Saving tokenized dataset to: {output_path}")
    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    tokenize_dataset(INPUT_CSV, OUTPUT_PATH)
