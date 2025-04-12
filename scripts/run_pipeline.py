import os

print("[INFO] Running full LogBERT pipeline...")

# Step 1: Generate synthetic logs
print("[INFO] Generating synthetic logs...")
os.system("python ../scripts/log_gen.py")

# Step 2: Preprocess logs
print("[INFO] Preprocessing logs...")
os.system("python ../scripts/preprocess.py")

# Step 3: Tokenize logs
print("[INFO] Tokenize logs...")
os.system("python ../scripts/preprocesstokenize_logs.py")

# Step 4: Train LogBERT model
print("[INFO] Training LogBERT model...")
os.system("python ../scripts/train_logbert.py")

# Step 4: Generate Report
print("[INFO] Generate Reports...")
os.system("python ../scripts/generate_report.py")

# Step 5: Detect anomalies
print("[INFO] Detecting anomalies...")
os.system("python ../scripts/anomaly_detection.py")

print("[INFO] Full pipeline execution completed!")
