# LogBERT for Ceph Log Anomaly Detection

## 📌 Project Overview
This project trains a self-supervised LogBERT model to detect anomalies in Ceph storage logs. It includes synthetic log generation, model training, anomaly detection, and visualization.

## 📂 Project Structure
logbert_ceph_anomaly_detection/
│── data/
│   ├── logs/                       # Generated synthetic Ceph logs
│   ├── processed/                  # Preprocessed log data
│── models/
│   ├── logbert.pth                 # Trained LogBERT model
│   ├── logbert_finetuned.pth       # Fine-tuned model
│── results/                        # Detected anomalies output
│── plots/                          # Visualization output
│── scripts/
│   ├── gen_training_logs.py        # Generate synthetic logs for training
|   ├── gen_test_logs.py            # Generate synthetic logs for testing for anomaly detection  
│   ├── preprocess.py               # Preprocess logs
│   ├── train_logbert.py            # Train LogBERT model
│   ├── tokenize_logs.py            # tokenize training preprocessed logs for training the model
│   ├── generate_report.py          # Visualization of metrics and reporting
│   ├── anomaly_detection.py        # Detection of anomalies 
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation

## 🚀 Setup Instructions

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Generate Synthetic Logs
```sh
python scripts/log-gen.py
```

### 3️⃣ Preprocess Logs
```sh
python scripts/preprocess.py
```

### 4️⃣ Train LogBERT Model
```sh
python scripts/train_logbert.py
```

### 5️⃣ Fine-Tune LogBERT (Optional)
```sh
python scripts/fine_tune_logbert.py
```

### 6️⃣ Detect Anomalies in New Logs
```sh
python scripts/anomaly_detection.py
```

### 7️⃣ Visualize Anomalies
```sh
python scripts/visualize_anomalies.py
```

## 📊 Results
- Anomaly logs will be saved in `results/anomalies.csv`
- A timeline plot of detected anomalies will be saved in `plots/anomaly_trend.png`

## 🎯 Future Improvements
- Enhance log parsing with Drain for better template extraction
- Deploy as a real-time monitoring system
- Integrate an alerting mechanism for detected anomalies

