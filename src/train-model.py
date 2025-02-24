import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Paths
DATA_PATH = "../data/logbert_preprocessed_logs.json"
FINE_TUNED_MODEL_PATH = "../model/logbert_finetuned_optimized.pth"

# Ensure model directory exists
os.makedirs("../model", exist_ok=True)

# Load dataset
with open(DATA_PATH, "r") as f:
    data = json.load(f)

train_logs = data["train"]
test_logs = data["test"]

# Load tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
INITIAL_LR = 3e-5
MAX_SEQ_LEN = 50
THRESHOLD = 0.5
DEVICE = torch.device("cpu")

class LogDataset(Dataset):
    def __init__(self, logs, tokenizer):
        self.logs = logs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log_entry = self.logs[idx]
        encoding = self.tokenizer(
            " ".join(log_entry["tokens"]),
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(log_entry["label"], dtype=torch.float)
        return input_ids, attention_mask, label

# Load Data
train_dataset = LogDataset(train_logs, tokenizer)
test_dataset = LogDataset(test_logs, tokenizer)

# Balanced sampling
train_labels = [log["label"] for log in train_logs]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LogBERT(nn.Module):
    def __init__(self, model_name):
        super(LogBERT, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.fc(cls_output)

model = LogBERT(MODEL_NAME).to(DEVICE)

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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def should_stop(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def fine_tune():
    model.train()
    optimizer.zero_grad()
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(EPOCHS):
        total_loss = 0
        for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = (
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                labels.to(DEVICE),
            )
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)
        if early_stopping.should_stop(avg_loss):
            print("Early stopping triggered.")
            break

    print("Fine-tuning complete.")

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

if __name__ == "__main__":
    fine_tune()
    evaluate()
    torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
    print(f"Fine-Tuned Model saved as '{FINE_TUNED_MODEL_PATH}'.")

