import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# Path to log file
LOG_FILE = "../data/synthetic_logs.log"
OUTPUT_JSON = "../data/logbert_preprocessed_logs.json"

# Define log levels (Normal vs. Anomalous)
NORMAL_LEVELS = ["INFO", "DEBUG"]
ANOMALOUS_LEVELS = ["WARNING", "ERROR", "CRITICAL"]

# Regular expression to parse logs
LOG_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (\w+): (.+)")

# Function to load and parse logs
def load_logs(filename):
    logs = []
    with open(filename, "r") as file:
        for line in file:
            match = LOG_PATTERN.match(line.strip())
            if match:
                timestamp, level, component, message = match.groups()
                label = 0 if level in NORMAL_LEVELS else 1  # Normal = 0, Anomaly = 1
                logs.append({
                    "timestamp": timestamp,
                    "level": level,
                    "component": component,
                    "message": message,
                    "label": label
                })
    return logs

# Tokenization function (splits log messages into words)
def tokenize(logs):
    tokenized_logs = []
    for log in logs:
        tokens = re.findall(r"\b\w+\b", log["message"].lower())  # Extract words
        log["tokens"] = tokens
        tokenized_logs.append(log)
    return tokenized_logs

# Build vocabulary from tokenized logs
def build_vocab(tokenized_logs, min_freq=2):
    word_counter = Counter()
    for log in tokenized_logs:
        word_counter.update(log["tokens"])

    # Only keep words with minimum frequency
    vocab = {word: i+1 for i, (word, freq) in enumerate(word_counter.items()) if freq >= min_freq}
    vocab["<UNK>"] = len(vocab) + 1  # Unknown token

    return vocab

# Convert logs to numerical sequences
def encode_logs(tokenized_logs, vocab, max_seq_len=20):
    for log in tokenized_logs:
        log["sequence"] = [vocab.get(word, vocab["<UNK>"]) for word in log["tokens"][:max_seq_len]]
        log["sequence"] += [0] * (max_seq_len - len(log["sequence"]))  # Padding
    return tokenized_logs

# Split dataset into train/test sets
def split_data(logs, test_size=0.2):
    train_logs, test_logs = train_test_split(logs, test_size=test_size, stratify=[log["label"] for log in logs], random_state=42)
    return train_logs, test_logs

# Save logs to JSON for LogBERT training
def save_to_json(train_logs, test_logs, output_file):
    dataset = {"train": train_logs, "test": test_logs}
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Preprocessed logs saved to {output_file}")

# Main pipeline
def preprocess_logs():
    print("Loading logs...")
    logs = load_logs(LOG_FILE)

    print("Tokenizing logs...")
    tokenized_logs = tokenize(logs)

    print("Building vocabulary...")
    vocab = build_vocab(tokenized_logs)

    print("Encoding logs...")
    encoded_logs = encode_logs(tokenized_logs, vocab)

    print("Splitting data into train/test...")
    train_logs, test_logs = split_data(encoded_logs)

    print("Saving preprocessed data...")
    save_to_json(train_logs, test_logs, OUTPUT_JSON)

# Run preprocessing
if __name__ == "__main__":
    preprocess_logs()
