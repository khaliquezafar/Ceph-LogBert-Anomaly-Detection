# preprocess.py
import os
import re
import pandas as pd

LOG_PATH = "../data/logs/ceph.log"
OUTPUT_PATH = "../data/processed/processed_logs.csv"

def extract_logs(log_path):
    def log_lines_generator():
        with open(log_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Extract timestamp and message
                timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    message = line[len(timestamp):].strip()
                else:
                    timestamp = None
                    message = line

                yield {
                    "timestamp": timestamp,
                    "message": message
                }

    return pd.DataFrame(log_lines_generator())

def main():
    print(f"📄 Reading logs from: {LOG_PATH}")
    df = extract_logs(LOG_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Processed logs saved to: {OUTPUT_PATH}")
    print(f"📈 Total log entries processed: {len(df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-tokenization even if output exists")
    args = parser.parse_args()

    main()

    tokenized_path = "../data/processed/tokenized_dataset"
    if args.force or not os.path.exists(tokenized_path):
        print("🚀 Running tokenizer script...")
        os.system("python ../scripts/tokenize_logs.py")
    else:
        print("✅ Tokenized dataset already exists. Skipping tokenization.")
