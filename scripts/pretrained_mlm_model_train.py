# === LogBERT Model Pretrained script ===

import torch
import time
import os
import psutil
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

def create_model_and_tokenizer():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

def create_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        no_cuda=False,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False
    )

def train(chunked=False):
    final_model_output_dir = "../models/basemodel"
    print(f"Memory usage before training: {psutil.virtual_memory().percent}%")

    dataset = load_from_disk("../data/processed_logs/tokenized_dataset")
    dataset.set_format("torch", device="cuda" if torch.cuda.is_available() else "cpu")

    if chunked:
        num_chunks = 10
        chunk_size = len(dataset) // num_chunks
        print(f"🔢 Total logs: {len(dataset)}, Logs per chunk: {chunk_size}")

        for i in range(num_chunks):
            print(f"\n🔁 Processing Chunk {i+1}/{num_chunks}")
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(dataset)
            chunk = dataset.select(range(start_idx, end_idx))
            chunk = chunk.train_test_split(test_size=0.2)

            model, tokenizer = create_model_and_tokenizer()
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
            chunk_model_dir = f"../models/base_model_chunk_{i+1}"
            training_args = create_training_args(chunk_model_dir)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=chunk['train'],
                eval_dataset=chunk['test'],
                data_collator=data_collator
            )

            trainer.train()

            print(f"💾 Saving chunk {i+1} model to {chunk_model_dir}")
            model.save_pretrained(chunk_model_dir)
            tokenizer.save_pretrained(chunk_model_dir)

            # Save final model after last chunk
            if i == num_chunks - 1:
                print(f"💾 Saving final model to {final_model_output_dir}")
                model.save_pretrained(final_model_output_dir)
                tokenizer.save_pretrained(final_model_output_dir)
                print(f"✅ Final model saved to {final_model_output_dir}")

    else:
        dataset = dataset.train_test_split(test_size=0.2)
        model, tokenizer = create_model_and_tokenizer()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
        training_args = create_training_args(final_model_output_dir)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator
        )

        trainer.train()

        print(f"💾 Saving model to {final_model_output_dir}")
        model.save_pretrained(final_model_output_dir)
        tokenizer.save_pretrained(final_model_output_dir)
        print(f"✅ Model saved to {final_model_output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked", action="store_true", help="Enable chunked training")
    args = parser.parse_args()
    train(chunked=args.chunked)
