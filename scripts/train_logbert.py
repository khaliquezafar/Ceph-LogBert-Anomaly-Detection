# train_logbert.py
import torch
import numpy as np
import pandas as pd
import time
import os
import psutil
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ComputeMetrics:
    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average='weighted', zero_division=0),
            "recall": recall_score(labels, predictions, average='weighted', zero_division=0),
            "f1": f1_score(labels, predictions, average='weighted', zero_division=0),
        }

def create_model_and_tokenizer():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    # model.gradient_checkpointing_enable()  # Disabled for faster training
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

def create_training_args(output_dir="../models"):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="steps",
        logging_dir="../data/logs",
        report_to="none",
        no_cuda=False,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4
    )

def train(chunked=False):
    import shutil
    import glob
    import argparse
    
    print(f"Memory usage before training: {psutil.virtual_memory().percent}%")

    from datasets import load_from_disk
    print("📦 Loading tokenized dataset from disk...")
    dataset = load_from_disk("../data/processed/tokenized_dataset")
    dataset.set_format("torch", device="cuda" if torch.cuda.is_available() else "cpu")

    if chunked:
        chunk_size = 10000
        import math
        num_chunks = math.ceil(len(dataset) / chunk_size)

        model, tokenizer = create_model_and_tokenizer()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
        training_args = create_training_args()

        completed_chunks = [int(f.name.split('_')[-1].split('.')[0]) for f in os.scandir("../models") if f.name.startswith("logbert_chunk_") and f.name.endswith(".pth")]
        print(f"📊 Total chunks: {num_chunks}")
        print(f"✅ Completed chunks: {sorted(completed_chunks)}")
        print(f"🚀 Resuming from chunk(s): {[i+1 for i in range(num_chunks) if (i+1) not in completed_chunks]}")
        for i in range(num_chunks):
            if (i + 1) in completed_chunks:
                print(f"⏭️ Skipping already completed chunk {i+1}")
                continue
            print(f"🔁 Training chunk {i+1}/{num_chunks}")
            chunk = dataset.select(range(i * chunk_size, min((i + 1) * chunk_size, len(dataset))))
            chunk = chunk.train_test_split(test_size=0.2)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=chunk['train'],
                eval_dataset=chunk['test'],
                compute_metrics=ComputeMetrics(),
                data_collator=data_collator
            )

            trainer.train()

            # Log final training loss and metrics at epoch-level
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'epoch' in log]
                for log in train_logs:
                    row = {
                        'epoch': log.get('epoch'),
                        'train_loss': log.get('loss'),
                        'accuracy': log.get('eval_accuracy'),
                        'precision': log.get('eval_precision'),
                        'recall': log.get('eval_recall'),
                        'f1': log.get('eval_f1')
                    }
                    pd.DataFrame([row]).to_csv("../data/logs/training_metrics_log.csv", mode="a", index=False, header=not os.path.exists("../data/logs/training_metrics_log.csv"))
            eval_metrics = trainer.evaluate()
            print("📈 Evaluation after chunk:", eval_metrics)
            pd.DataFrame([{**eval_metrics, "chunk": i+1}]).to_csv("../data/logs/chunk_eval_metrics.csv", mode="a", index=False, header=not os.path.exists("../data/logs/chunk_eval_metrics.csv"))
            logs = pd.DataFrame(trainer.state.log_history)
            logs.to_csv("../data/logs/training_metrics_log.csv", mode="a", index=False, header=not os.path.exists("../data/logs/training_metrics_log.csv"))
            trainer.save_model(f"../models/logbert_chunk_{i+1}.pth")

        # Save final model after last chunk
        if i + 1 == num_chunks:
            model.save_pretrained("../models/logbert.pth")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model.save_pretrained(f"../models/logbert_final_{timestamp}.pth")
            print("✅ Final model saved as ../models/logbert.pth")

        return

    dataset = dataset.train_test_split(test_size=0.2)

    model, tokenizer = create_model_and_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    training_args = create_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=ComputeMetrics(),
        data_collator=data_collator
    )

    print("\nResuming training if checkpoint exists...\n")
    start_time = time.time()

    latest_checkpoint = None
    checkpoints = [f.path for f in os.scandir("../models") if f.is_dir() and f.name.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
        print(f"Found checkpoint: {latest_checkpoint}. Resuming training.\n")

    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    end_time = time.time()
    print(f"\n✅ Training complete in {(end_time - start_time)/60:.2f} minutes.\n")

    trainer.save_model("../models/logbert.pth")
    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv("../data/logs/training_metrics_log.csv", index=False)

    predictions = trainer.predict(dataset['test'])
    metrics = predictions.metrics

    print("Final Evaluation Metrics:")
    print("Accuracy: {:.4f}".format(metrics['test_accuracy']))
    print("Precision: {:.4f}".format(metrics['test_precision']))
    print("Recall: {:.4f}".format(metrics['test_recall']))
    print("F1 Score: {:.4f}".format(metrics['test_f1']))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--report", action="store_true", help="Generate training report after completion")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (skip report generation)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked training mode for large datasets")
    args = parser.parse_args()

    train(args.chunked)

    if args.report and not args.headless:
        os.system("python ../scripts/generate_report.py")
