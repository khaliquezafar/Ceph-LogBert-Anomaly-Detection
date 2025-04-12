# anomaly_detection.py
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from datasets import Dataset
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
LOG_FILE = "../data/logs/ceph_real.log"
MODEL_PATH = "../models/logbert.pth"
OUTPUT_FILE = "../results/anomalies.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model and Tokenizer ===
print("📦 Loading model and tokenizer...")
model = BertForMaskedLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()

# === Load Logs ===
print(f"📑 Loading log file: {LOG_FILE}")
with open(LOG_FILE, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# === Normalize Logs ===
def normalize_log(log):
    log = log.lower()
    log = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "", log)
    log = re.sub(r"\d+", "0", log)
    return log.strip()

lines = [normalize_log(line) for line in lines]
df = pd.DataFrame({"log": lines})
df = df.drop_duplicates(subset="log").reset_index(drop=True)
dataset = Dataset.from_pandas(df)

# === Tokenize ===
print("✍️ Tokenizing logs...")
def tokenize_fn(ex):
    return tokenizer(ex["log"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === Score Each Log Entry ===
print("🔍 Scoring logs for anomalies...")
anomalies = []
all_log_probs = []
softmax = torch.nn.Softmax(dim=-1)

for i, batch in enumerate(dataset):
    input_ids = batch["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = softmax(logits)
    token_log_probs = torch.log(probs + 1e-10)
    input_token_log_probs = token_log_probs[0, torch.arange(len(input_ids[0])), input_ids[0]]
    avg_log_prob = input_token_log_probs.mean().item()
    all_log_probs.append(avg_log_prob)

# === Determine Threshold Based on Distribution ===
threshold = np.percentile(all_log_probs, 10)
print(f"📉 Dynamic threshold set to: {threshold:.4f}")

# === Filter Anomalies ===
COMMON_PATTERNS = ["scrub starts"]
for i, score in enumerate(all_log_probs):
    if score < threshold and not any(pat in lines[i] for pat in COMMON_PATTERNS):
        anomalies.append({"line": lines[i], "avg_log_prob": score})

# === Save and Print Results ===
print(f"✅ {len(anomalies)} anomalies detected. Saving to {OUTPUT_FILE}")
for anomaly in anomalies:
    print(f"🚨 {anomaly['avg_log_prob']:.4f}: {anomaly['line']}")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
pd.DataFrame(anomalies).to_csv(OUTPUT_FILE, index=False)

# === Plot Distribution ===
plt.figure(figsize=(10, 4))
plt.hist(all_log_probs, bins=50, color='skyblue')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.title("Log Probability Distribution")
plt.xlabel("Average Log Probability")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/log_prob_dist.png")
print("🎉 Done.")
