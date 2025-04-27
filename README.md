# Ceph Storage Anomaly Detection Using LogBERT through Log Analysis

## 📌 Project Overview




## 📂 Project Structure
logbert_ceph_anomaly_detection/
│── data/
│       ├───eval_data/                          # Models training historical evaluation data 
|       ├───processed_logs/                     # Processed and tokenized dataset 
|       │   └───tokenized_dataset/              # Tokenized dataset 
|       └───raw_logs/                           # Synthetic Ceph logs for models training and testing 
│── models/
│       ├── basemodel/                          # Base Trained LogBERT model (huggingface format)
│       ├── finetuned_model/                    # Finetuned Trained LogBERT model (huggingface format)
|       ├── checkpoints/                        # Models training chunk wise checkpoints directories
│── results/                                    # Anomaly Detection output in csv file from Fine-tuned model
│── plots/                                      # Models Metrics and performance and anomaly detection visualization 
|       ├───model_metrics/                      # Model Metrics html dashboard
|       └───anomaly_detection/                  # Anomaly detection html dashboard             
│── scripts/
│       ├── gen_syntetic_logs.py                # Synthetic logs generation script for model training and testing
│       ├── preprocess_logs.py                  # Preprocess synthetic logs script for model training 
│       ├── pretrained_mlm_model_train.py       # Base Logbert model Training script
|       ├── finetuned_mlm_model_train.py        # Fine-tuned Logbert model Training script
│       ├── tokenize_logs.py                    # Tokenize training preprocessed logs script for training the model
│       ├── visualization_metrics.py            # Model metrics visualization and reporting script
│       ├── anomalies_detection_report.py       # Detection of anomalies script using trained models
│── requirements.txt                            # Python dependencies
│── README.md                                   # Project documentation

## 🚀 Setup Instructions

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```
### 2.  Generate Synthetic training and testing Logs
```sh
python scripts/gen_synthetic_logs.py
```
### 2.Preprocess Training Logs
```sh
python scripts/preprocess_logs.py
python scripts/tokenize_logs.py
```
### 3. Train LogBERT Model
```sh
python scripts/pretrained_mlm_model_train.py --chunked
```
### 4. Train Fine-Tune LogBERT Model
```sh
python scripts/finetuned_mlm_model_train.py --chunked
```
### 5. Visualize Model Metrics and Reporting
```sh
python scripts/visualization_metrics.py

### 6. Anomaly detection and Reporting
```sh
python scripts/anomalies_detection_report.py 
```
## 📊 Results
- Anomaly inference logs will be saved in 'results' directory in the with given output csv file inference_anomalies_summary.csv ../results/inference_anomalies_summary.csv
- Model metrics and anomaly detection html dashboard report with graphs will be saved in 'plots' directory. 
-- Model metrics Report >> ../plots/model_metrics/metrics_visualization.html
-- Anomaly detection Report >> ../polts/anomaly_detection/anomaly_explanation_summary.html



