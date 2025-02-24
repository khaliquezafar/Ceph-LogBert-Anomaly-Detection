import re
import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer

# Paths
LOG_FILE = "../data/synthetic_logs.log"
OUTPUT_JSON = "../data/logbert_preprocessed_logs.json"

# Define log levels (Normal vs. Anomalous)
NORMAL_LEVELS = ["INFO", "DEBUG", "NOTICE"]
ANOMALOUS_LEVELS = ["WARNING", "ERROR", "CRITICAL"]

# Regular expression to parse logs
LOG_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+) (\S+)\[(\d+)\]: (\S+): (.+)")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_logs(filename):
    """
    Load and parse logs from the file.
    """
    logs = {"normal": [], "anomaly": []}
    
    with open(filename, "r") as file:
        for line in file:
            match = LOG_PATTERN.match(line.strip())
            if match:
                timestamp, node, component, process_id, level, message = match.groups()
                label = 0 if level.upper() in NORMAL_LEVELS else 1  # Normal = 0, Anomaly = 1
                
                log_entry = {
                    "timestamp": timestamp,
                    "node": node,
                    "component": component,
                    "process_id": process_id,
                    "level": level,
                    "message": message,
                    "label": label
                }
                
                # Categorize logs based on normal or anomaly
                if label == 0:
                    logs["normal"].append(log_entry)
                else:
                    logs["anomaly"].append(log_entry)
    
    return logs

def balance_data(logs):
    """
    Ensure that the number of normal logs matches the number of anomaly logs.
    """
    num_anomalies = len(logs["anomaly"])
    num_normals = len(logs["normal"])

    # Balance the dataset by undersampling or oversampling normal logs
    if num_normals > num_anomalies:
        logs["normal"] = random.sample(logs["normal"], num_anomalies)
    else:
        logs["anomaly"] = random.sample(logs["anomaly"], num_normals)

    balanced_logs = logs["normal"] + logs["anomaly"]
    random.shuffle(balanced_logs)

    return balanced_logs

def tokenize(logs):
    """
    Tokenize log messages using BertTokenizer.
    """
    tokenized_logs = []
    for log in logs:
        tokens = tokenizer.tokenize(log["message"].lower())
        log["tokens"] = tokens
        tokenized_logs.append(log)
    return tokenized_logs

def build_vocab(tokenized_logs, min_freq=2):
    """
    Build vocabulary from tokenized logs.
    """
    word_counter = Counter()
    for log in tokenized_logs:
        word_counter.update(log["tokens"])

    # Only keep words with minimum frequency
    vocab = {word: i + 1 for i, (word, freq) in enumerate(word_counter.items()) if freq >= min_freq}
    vocab["<UNK>"] = len(vocab) + 1  # Unknown token

    return vocab

def encode_logs(tokenized_logs, vocab, max_seq_len=20):
    """
    Convert logs to numerical sequences.
    """
    for log in tokenized_logs:
        log["sequence"] = [vocab.get(word, vocab["<UNK>"]) for word in log["tokens"][:max_seq_len]]
        log["sequence"] += [0] * (max_seq_len - len(log["sequence"]))  # Padding
    return tokenized_logs

def split_data(logs, test_size=0.2):
    """
    Ensure that normal logs are well-represented in both train and test datasets.
    """
    train_logs, test_logs = train_test_split(logs, test_size=test_size, stratify=[log["label"] for log in logs], random_state=42)
    
    return train_logs, test_logs

def save_to_json(train_logs, test_logs, output_file):
    """
    Save the train/test datasets to a JSON file.
    """
    dataset = {"train": train_logs, "test": test_logs}
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Preprocessed logs saved to {output_file}")

def preprocess_logs():
    """
    Main pipeline for log preprocessing.
    """
    print("Loading logs...")
    logs = load_logs(LOG_FILE)

    print("Balancing dataset...")
    balanced_logs = balance_data(logs)

    print("Tokenizing logs...")
    tokenized_logs = tokenize(balanced_logs)

    print("Building vocabulary...")
    vocab = build_vocab(tokenized_logs)

    print("Encoding logs...")
    encoded_logs = encode_logs(tokenized_logs, vocab)

    print("Splitting data into train/test...")
    train_logs, test_logs = split_data(encoded_logs)

    print("Saving preprocessed data...")
    save_to_json(train_logs, test_logs, OUTPUT_JSON)


if __name__ == "__main__":
    preprocess_logs()