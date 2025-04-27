# === LogBERT Model Pretrained script ===

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

def create_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=1,
        logging_dir="../data/eval_data/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
    )

def train_finetune_model(chunked=False):
    last_model_path = ""
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

    if chunked:
        chunk_size = len(full_train) // 5
        last_model_path = ""
        for i in range(5):
            print(f"Starting training for chunk {i+1}...")
            chunk_train = full_train.select(range(i * chunk_size, (i + 1) * chunk_size))
            model = BertForMaskedLM.from_pretrained(base_model_path if i == 0 else f"../models/checkpoints/finetuned_model_chunk_{i}")
            training_args = create_training_args(f"../models/checkpoints/finetuned_model_chunk_{i+1}")
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

            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'epoch' in log]
                if train_logs:
                    avg_loss = np.mean([log['loss'] for log in train_logs])
                    row = {'chunk': i+1, 'avg_train_loss': avg_loss}
                    pd.DataFrame([row]).to_csv(hl_eval_pth, mode="a", index=False, header=not os.path.exists(hl_eval_pth))

            eval_metrics = trainer.evaluate()
            print("Evaluation after chunk", i+1, ":", eval_metrics)
            pd.DataFrame([{**eval_metrics, "chunk": i+1}]).to_csv(gr_eval_pth, mode='a', index=False, header=not os.path.exists(gr_eval_pth))
    else:
        train_dataset = full_train
        model = BertForMaskedLM.from_pretrained(base_model_path)
        training_args = create_training_args(save_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=ComputeMetrics()
        )

        trainer.train()
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'epoch' in log]
            if train_logs:
                avg_loss = np.mean([log['loss'] for log in train_logs])
                row = {'chunk': 1, 'avg_train_loss': avg_loss}
                pd.DataFrame([row]).to_csv(hl_eval_pth, mode="a", index=False, header=not os.path.exists(hl_eval_pth))

        predictions = trainer.predict(test_dataset)
        metrics = predictions.metrics

        pd.DataFrame([{**metrics, "chunk": 1}]).to_csv(gr_eval_pth, mode='a', index=False, header=not os.path.exists(gr_eval_pth))

        print("Final Evaluation Metrics:")
        print("Accuracy: {:.4f}".format(metrics['eval_accuracy']))
        print("Precision: {:.4f}".format(metrics['eval_precision']))
        print("Recall: {:.4f}".format(metrics['eval_recall']))
        print("F1 Score: {:.4f}".format(metrics['eval_f1']))

    return last_model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked", action="store_true", help="Enable chunked training mode")
    args = parser.parse_args()

    last_model_path = train_finetune_model(chunked=args.chunked)

    if args.chunked and last_model_path:
        final_model = BertForMaskedLM.from_pretrained(last_model_path)
        final_tokenizer = BertTokenizer.from_pretrained(last_model_path)
        final_model.save_pretrained(save_path)
        final_tokenizer.save_pretrained(save_path)
        print("Final chunked model saved to", save_path)
