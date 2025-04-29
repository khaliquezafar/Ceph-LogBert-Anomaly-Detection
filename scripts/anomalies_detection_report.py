# === Anomaly Detection and Reporting script ===

import os
import re
import torch
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
import plotly.io as pio
from transformers import BertTokenizer, BertForMaskedLM
from datasets import Dataset
import argparse

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Anomaly Detection and Reporting Script")
parser.add_argument("--log_file", type=str, default="../data/raw_logs/syn_test_ceph.log", help="Path to the log file")
parser.add_argument("--model_dir", type=str, default="../models/finetuned_model", help="Directory path of the finetuned model")
parser.add_argument("--output_csv", type=str, default="../results/inference_anomalies_summary.csv", help="Path to save output CSV")
parser.add_argument("--percentile", type=int, default=70, help="Anomaly score percentile threshold")
args = parser.parse_args()

# === Config ===
log_file_path = args.log_file
model_dir = args.model_dir
output_csv = args.output_csv
percentile = args.percentile

html_output_dir = "../plots/anomaly_detection"
os.makedirs(html_output_dir, exist_ok=True)
html_file_path = os.path.join(html_output_dir, "anomaly_explanation_summary.html")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained(model_dir)
MAX_LENGTH = 64

# === Load Logs ===
def extract_logs(log_path):
    def log_lines_generator():
        with open(log_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                yield {"message": line}
    return pd.DataFrame(log_lines_generator())

print("\U0001F4C4 Preprocessing logs...")
df_raw = extract_logs(log_file_path)
df_raw = df_raw.dropna(subset=["message"])
df_raw = df_raw[df_raw["message"].str.strip().astype(bool)]

print("\U0001F512 Tokenizing...")
dataset = Dataset.from_pandas(df_raw[["message"]])
dataset = dataset.map(
    lambda x: TOKENIZER(x["message"], padding="max_length", truncation=True, max_length=MAX_LENGTH),
    batched=True
)
dataset = dataset.remove_columns(["message"])
dataset = dataset.map(lambda x: {**x, "labels": x["input_ids"]})
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Load Trained Model ===
print(f"\U0001F4E6 Loading model from: {model_dir}")
model = BertForMaskedLM.from_pretrained(model_dir).to(DEVICE)
model.eval()

# === Score Logs ===
print("\U0001F9B2 Scoring...")
scores = []
losses = []
with torch.no_grad():
    for i in range(len(dataset)):
        inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in dataset[i].items() if k in ["input_ids", "attention_mask"]}
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        scores.append(outputs.loss.item())
        token_loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            reduction="none"
        ).view(labels.size()).cpu().numpy()
        losses.append(token_loss[0])

scores = np.array(scores)
threshold = np.percentile(scores, percentile)
print(f"\U0001F4C8 Anomaly threshold ({percentile}th percentile): {threshold:.4f}")


df_raw["anomaly_score"] = scores
df_raw["is_anomaly"] = scores > threshold

# === Rule-Based Heuristics ===
anomaly_keywords = ["[ERR]", "[WRN]", "[CRIT]"]
df_raw["rule_based_flag"] = df_raw["message"].apply(
    lambda x: any(kw in x for kw in anomaly_keywords)
)

# === Whitelist Logs with [INF] or [DBG] ===
whitelist_keywords = ["[INF]", "[DBG]", "[OK]"]
df_raw["is_whitelisted"] = df_raw["message"].apply(
    lambda x: any(kw in x for kw in whitelist_keywords)
)

df_raw.loc[df_raw["is_whitelisted"], "is_anomaly"] = False
df_raw.loc[df_raw["is_whitelisted"], "rule_based_flag"] = False

# === Combined Anomaly ===
df_raw["combined_anomaly"] = df_raw["is_anomaly"] | df_raw["rule_based_flag"]

# === Confidence Score Calculation ===
min_score = scores.min()
max_score = scores.max()
df_raw["confidence_score"] = df_raw["anomaly_score"].apply(lambda x: (x - min_score) / (max_score - min_score + 1e-5))
df_raw["confidence_score"] = df_raw["confidence_score"].round(4)

# === Contextual Token Explanations ===
token_insight_map = {
    "CRIT": "critical failure event",
    "ERR": "indicates a system failure",
    "WRN": "system degraded",
}

def token_contextual_explanation(tokens):
    explanations = []
    for token in tokens:
        if token in token_insight_map:
            explanations.append(f"'{token}' ({token_insight_map[token]})")
        else:
            explanations.append(f"'{token}' (uncommon usage)")
    return ", ".join(explanations)

# === Generate Explanations ===
enhanced_explanations = []
for i in range(len(df_raw)):
    if df_raw.loc[i, 'is_whitelisted']:
        explanation = (
            "No anomaly detected in this log. The content aligns with expected patterns for normal operations."
        )
    else:
        token_losses = losses[i]
        tokens_decoded = TOKENIZER.convert_ids_to_tokens(dataset[i]["input_ids"].cpu().numpy())
        top_indices = token_losses.argsort()[-3:][::-1]
        top_tokens = [tokens_decoded[idx] for idx in top_indices if tokens_decoded[idx] not in TOKENIZER.all_special_tokens]
        detailed_explanation = token_contextual_explanation(top_tokens)

        if df_raw.loc[i, 'combined_anomaly']:
            explanation = (
                f"This log shows unusual patterns with tokens: {detailed_explanation}. "
                "These contributed significantly to the anomaly score. "
                "It was flagged as anomalous because of high model loss and/or matching critical keywords."
            )
        else:
            if df_raw.loc[i, 'rule_based_flag']:
                reason = "However, despite containing flagged keywords, the overall pattern was not considered anomalous by the model."
            else:
                reason = "The log's content closely aligns with patterns seen in normal logs, resulting in a low anomaly score."
            explanation = (
                f"Although some tokens like: {detailed_explanation} show moderate deviation, "
                f"the model determined this log to be normal. {reason}"
            )
    enhanced_explanations.append(explanation)

df_raw["explanation"] = enhanced_explanations

# === Console Output for Anomalies ===
total_logs = len(df_raw)
mlm_anomalies = df_raw['is_anomaly'].sum()
rule_anomalies = df_raw['rule_based_flag'].sum()
combined_anomalies = df_raw['combined_anomaly'].sum()

mlm_percent = (mlm_anomalies / total_logs) * 100
rule_percent = (rule_anomalies / total_logs) * 100
combined_percent = (combined_anomalies / total_logs) * 100

print(f"\n\U0001F4CA Total MLM-Detected Anomalies: {mlm_anomalies} ({mlm_percent:.2f}%)")
print(f"\n\U0001F4CA Total Rule-Based Anomalies: {rule_anomalies} ({rule_percent:.2f}%)")
print(f"\n\U0001F4CA Total Combined Anomalies: {combined_anomalies} ({combined_percent:.2f}%)")

# === Save CSV ===
df_raw.to_csv(output_csv, index=False)
print(f"\n✅ Anomalies saved to: {output_csv}")

# === HTML report generation ===
# === Plotly Charts ===
# Bar Chart
bar_fig = go.Figure(data=[
    go.Bar(name='Anomaly Types', x=['MLM', 'Rule-Based', 'Combined'], y=[mlm_anomalies, rule_anomalies, combined_anomalies])
])
bar_fig.update_layout(title='Anomaly Counts', xaxis_title='Type', yaxis_title='Count')
bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs=False)

# === Grid-Style Heatmap (Visual Style) ===
norm_anomaly_score = (df_raw['anomaly_score'] - df_raw['anomaly_score'].min()) / (df_raw['anomaly_score'].max() - df_raw['anomaly_score'].min() + 1e-5)
norm_confidence_score = df_raw['confidence_score']
norm_combined = df_raw['combined_anomaly'].astype(int)

heatmap_matrix = [
    norm_anomaly_score.values,
    norm_confidence_score.values,
    norm_combined.values
]

heatmap_fig = go.Figure(data=go.Heatmap(
    z=heatmap_matrix,
    x=[f"Log {i+1}" for i in df_raw.index],
    y=['Anomaly Score', 'Confidence Score', 'Combined Anomaly'],
    colorscale=[
        [0.0, 'blue'],
        [0.25, 'cyan'],
        [0.5, 'yellow'],
        [0.75, 'orange'],
        [1.0, 'red']
    ],
    zmin=0, zmax=1,
    showscale=True
))

heatmap_fig.update_layout(
    title='Anomaly Heatmap',
    xaxis_title='Logs',
    yaxis_title='Metrics',
    xaxis=dict(showgrid=True, tickangle=45, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, tickfont=dict(size=12)),
    height=500,
    margin=dict(l=80, r=50, t=80, b=120)
)

heatmap_html = pio.to_html(heatmap_fig, full_html=False, include_plotlyjs=False)

# === Full HTML Report ===
html_content = f"""
<html><head><title>Anomaly Detection Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
.anomaly-log {{
    border: 2px solid red;
    background-color: #ffe6e6;
    padding: 10px;
    margin: 10px 0;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}}
.normal-log {{
    border: 1px solid green;
    background-color: #e6ffe6;
    padding: 10px;
    margin: 10px 0;
    border-radius: 10px;
    font-size: 14px;
}}
body {{
    font-family: Arial, sans-serif;
}}
</style></head><body>
<h1>Anomaly Detection Report</h1>
<h2>Total Anomalies Summary</h2>
<p><strong>Total Logs Processed:</strong> {total_logs}<br/>
<strong>MLM-Detected Anomalies:</strong> {mlm_anomalies} ({mlm_percent:.2f}%)<br/>
<strong>Rule-Based Anomalies:</strong> {rule_anomalies} ({rule_percent:.2f}%)<br/>
<strong>Combined Anomalies:</strong> {combined_anomalies} ({combined_percent:.2f}%)</p><hr>

<h3>Interactive Anomaly Counts Visualization</h3>{bar_html}<hr>
<h3>Anomaly Heatmap</h3>{heatmap_html}<hr>
<h2>Detailed Log Analysis</h2>
"""

for i, row in df_raw.iterrows():
    div_class = "anomaly-log" if row['combined_anomaly'] else "normal-log"
    html_content += f"<div class='{div_class}'><p><strong>Log {i+1}:</strong> {row['message']}<br/>"
    html_content += f"<strong>Anomaly Score:</strong> {row['anomaly_score']:.4f} | "
    html_content += f"<strong>Confidence:</strong> {row['confidence_score']:.2f} | "
    html_content += f"<strong>MLM Prediction:</strong> {int(row['is_anomaly'])} | "
    html_content += f"<strong>Rule Match:</strong> {int(row['rule_based_flag'])} | "
    html_content += f"<strong>Combined:</strong> {int(row['combined_anomaly'])}<br/>"
    html_content += f"<strong>Explanation:</strong> {row['explanation']}</p></div><hr>"

html_content += "</body></html>"

with open(html_file_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"✅ Interactive HTML report generated: {html_file_path}")
webbrowser.open(f"file://{os.path.abspath(html_file_path)}")
