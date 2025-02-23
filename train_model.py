import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset, WeightedRandomSampler
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Paths
DATA_PATH = "../data/logbert_preprocessed_logs.json"
MODEL_PATH = "../model/logbert_ceph.pth"

# Ensure model directory exists
os.makedirs("../model", exist_ok=True)

# Load tokenizer (RoBERTa for better log processing)
MODEL_NAME = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Hyperparameters optimized for large datasets
BATCH_SIZE = 64  # Increased batch size for efficiency
EPOCHS = 5  # Large dataset requires fewer epochs
INITIAL_LR = 3e-5  # Optimized learning rate
MAX_SEQ_LEN = 20
THRESHOLD = 0.4
GRAD_ACCUM_STEPS = 2  # Accumulate gradients every 2 batches
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔹 Custom IterableDataset (Efficient Data Loading for Large Datasets)
class LogDataset(IterableDataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def parse_line(self, line):
        log_entry = json.loads(line.strip())
        tokens = log_entry["tokens"]
        encoding = self.tokenizer(
            " ".join(tokens),
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(log_entry["label"], dtype=torch.float)
        return input_ids, attention_mask, label

    def __iter__(self):
        with open(self.file_path, "r") as file:
            for line in file:
                yield self.parse_line(line)


# 🔹 Use Streaming DataLoader for Efficient Large Dataset Loading
train_dataset = LogDataset(DATA_PATH, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load test dataset normally for evaluation
test_dataset = LogDataset(DATA_PATH, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 🔹 LogBERT Model Optimized for Large Datasets
class LogBERT(nn.Module):
    def __init__(self, model_name):
        super(LogBERT, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.fc(cls_output)  # No sigmoid here


# 🔹 Load Model
model = LogBERT(MODEL_NAME).to(DEVICE)

# **Focal Loss for Handling Imbalanced Data**
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
        probas = torch.sigmoid(logits)
        focal_weight = self.alpha * (1 - probas) ** self.gamma
        return (focal_weight * bce_loss).mean()


criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# **Enable Mixed Precision Training for Faster Performance**
scaler = torch.cuda.amp.GradScaler()


# 🔹 Fine-Tune Model with Mixed Precision & Gradient Accumulation
def fine_tune():
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = (
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                labels.to(DEVICE),
            )

            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS  # Normalize loss for accumulation

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:  # Update weights every `GRAD_ACCUM_STEPS`
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS  # Undo normalization

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("Fine-tuning complete.")


# 🔹 Evaluate Model with Adjusted Threshold
def evaluate():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                labels.to(DEVICE),
            )

            outputs = model(input_ids, attention_mask).squeeze()
            preds = (torch.sigmoid(outputs) > THRESHOLD).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


# 🔹 Fine-Tune and Save Model
if __name__ == "__main__":
    fine_tune()
    evaluate()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Fine-Tuned Model saved as '{MODEL_PATH}'.")
