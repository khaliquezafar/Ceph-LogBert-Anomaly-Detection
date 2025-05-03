# === LogBERT Model Finetuning script  ===

import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Config ===
log_file = "../data/raw_logs/syn_lab_ceph_logs.log"
base_model_path = "../models/basemodel"
save_path = "../models/finetuned_model"
hl_eval_pth = "../data/eval_data/train_loss.csv"
gr_eval_pth = "../data/eval_data/eval_metrics.csv"
lr_log_pth = "../data/eval_data/learning_rate_log.csv"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""  
class ComputeMetrics:
    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        labels = labels.flatten()
        predictions = predictions.flatten()
        mask = labels != -100

        labels = labels[mask]
        predictions = predictions[mask]

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average='weighted', zero_division=0),
            "recall": recall_score(labels, predictions, average='weighted', zero_division=0),
            "f1": f1_score(labels, predictions, average='weighted', zero_division=0),
        }

def load_logs_with_labels(path):
    logs, labels = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "##ANOMALY" in line:
                logs.append(line.split("##")[0])
                labels.append(1)
            elif "##NORMAL" in line:
                logs.append(line.split("##")[0])
                labels.append(0)
    return pd.DataFrame({"message": logs, "label": labels})

def tokenize_function(batch, tokenizer):
    result = tokenizer(batch["message"], padding="max_length", truncation=True, max_length=64)
    result["labels"] = batch["label"]
    return result

def create_training_args(output_dir, learning_rate):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=learning_rate,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=1,
        logging_dir="../data/eval_data",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
    )

def train_finetune_model(chunked=False, dynamic_lr=3e-5):
    print("Loading labeled data...")
    df = load_logs_with_labels(log_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    dataset = dataset.train_test_split(test_size=0.2)
    test_dataset = dataset["test"]
    full_train = dataset["train"]

    os.makedirs(os.path.dirname(hl_eval_pth), exist_ok=True)
    os.makedirs(os.path.dirname(gr_eval_pth), exist_ok=True)
    os.makedirs(os.path.dirname(lr_log_pth), exist_ok=True)

    last_model_path = ""
    chunk_size = len(full_train) // 5
    for i in range(5):
        print(f"Starting training for chunk {i+1} with learning rate {dynamic_lr}...")
        chunk_train = full_train.select(range(i * chunk_size, (i + 1) * chunk_size))
        model = BertForMaskedLM.from_pretrained(base_model_path if i == 0 else f"../models/checkpoints/finetuned_model_chunk_{i}")
        training_args = create_training_args(f"../models/checkpoints/finetuned_model_chunk_{i+1}", dynamic_lr)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=chunk_train,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=ComputeMetrics()
        )

        trainer.train()
        trainer.save_model(f"../models/checkpoints/finetuned_model_chunk_{i+1}")
        last_model_path = f"../models/checkpoints/finetuned_model_chunk_{i+1}"
        print(f"Finished training for chunk {i+1}")

        # Log average loss
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'epoch' in log]
            if train_logs:
                avg_loss = np.mean([log['loss'] for log in train_logs])
                row = {'chunk': i+1, 'avg_train_loss': avg_loss}
                pd.DataFrame([row]).to_csv(hl_eval_pth, mode="a", index=False, header=not os.path.exists(hl_eval_pth))

            # Log average learning rate
            lr_logs = [log['learning_rate'] for log in trainer.state.log_history if 'learning_rate' in log]
            if lr_logs:
                avg_lr = np.mean(lr_logs)
                lr_row = {'chunk': i+1, 'avg_learning_rate': avg_lr}
                pd.DataFrame([lr_row]).to_csv(lr_log_pth, mode="a", index=False, header=not os.path.exists(lr_log_pth))

        eval_metrics = trainer.evaluate()
        rounded_metrics = {k: round(v, 4) for k, v in eval_metrics.items()}
        rounded_metrics["chunk"] = i+1
        pd.DataFrame([rounded_metrics]).to_csv(gr_eval_pth, mode='a', index=False, header=not os.path.exists(gr_eval_pth))

    return last_model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked", action="store_true", help="Enable chunked training mode")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Set dynamic learning rate")
    args = parser.parse_args()

    last_model_path = train_finetune_model(chunked=args.chunked, dynamic_lr=args.learning_rate)

    if args.chunked and last_model_path:
        final_model = BertForMaskedLM.from_pretrained(last_model_path)
        final_tokenizer = BertTokenizer.from_pretrained(last_model_path)
        final_model.save_pretrained(save_path)
        final_tokenizer.save_pretrained(save_path)
        print("Final chunked model saved to", save_path)
