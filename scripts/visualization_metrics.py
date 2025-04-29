# === LogBERT Model Metrics Visualization and Reporting script ===

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# === Load CSV Files Directly ===
eval_csv = "../data/eval_data/eval_metrics.csv"
train_csv = "../data/eval_data/train_loss.csv"

# === Load Data ===
eval_df = pd.read_csv(eval_csv)
train_df = pd.read_csv(train_csv)

# === Merge on 'chunk' Only ===
df = eval_df.merge(train_df[["chunk", "avg_train_loss"]], on="chunk", how="left")

# === 1. Tabulated Overall Metrics ===
overall_metrics = eval_df[["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]].mean().to_frame().reset_index()
overall_metrics.columns = ["Metric", "Value"]
overall_metrics["Metric"] = overall_metrics["Metric"].map({
    "eval_accuracy": "Accuracy",
    "eval_precision": "Precision",
    "eval_recall": "Recall",
    "eval_f1": "F1-Score"
})
overall_metrics["Value"] = overall_metrics["Value"].round(2)

# === 2. Combined Overall Bar Graph ===
bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(x=overall_metrics["Metric"], y=overall_metrics["Value"], text=overall_metrics["Value"],
                         textposition='outside'))
bar_fig.update_layout(title="Overall Metrics", xaxis_title="Metric", yaxis_title="Score")

# === 3. Line Graphs for Accuracy, Precision, Recall, F1 by Chunk ===
metrics_fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy", "Precision", "Recall", "F1-Score"))
metrics_fig.add_trace(go.Scatter(x=df["chunk"], y=df["eval_accuracy"], mode='lines+markers', name="Accuracy"), row=1, col=1)
metrics_fig.add_trace(go.Scatter(x=df["chunk"], y=df["eval_precision"], mode='lines+markers', name="Precision"), row=1, col=2)
metrics_fig.add_trace(go.Scatter(x=df["chunk"], y=df["eval_recall"], mode='lines+markers', name="Recall"), row=2, col=1)
metrics_fig.add_trace(go.Scatter(x=df["chunk"], y=df["eval_f1"], mode='lines+markers', name="F1-Score"), row=2, col=2)
metrics_fig.update_layout(height=1000, width=1600, title_text="Metrics by Chunk Datasets", showlegend=False)

# === 4. Training and Eval Loss Line Graph ===
loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(x=df["chunk"], y=df["avg_train_loss"], mode='lines+markers', name="Training Loss"))
loss_fig.add_trace(go.Scatter(x=df["chunk"], y=df["eval_loss"], mode='lines+markers', name="Validation Loss"))
loss_fig.update_layout(title="Training and Validation Loss by Chunk", xaxis_title="Chunk Datasets", yaxis_title="Training and Validation Loss")

# === HTML Dashboard Generation ===
html_output = "../plots/model_metrics/metrics_visualization.html"
os.makedirs(os.path.dirname(html_output), exist_ok=True)
with open(html_output, "w") as f:
    f.write("<html><head><title>Ceph AD LogBERT MLM Model Performance Dashboard</title></head><body>")
    f.write("<h1>Ceph AD LogBERT MLM Model Performance Dashboard</h1>")
    f.write("<h2>Overall Metrics Table</h2>")
    f.write(overall_metrics.to_html(index=False, border=1, classes='full-width-table'))
    f.write("<style>.full-width-table { width: 100%; border-collapse: collapse; } .full-width-table th, .full-width-table td { padding: 8px; text-align: center; }</style>")
    f.write("<h2>Overall Metrics Bar Graph</h2>")
    f.write(bar_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<h2>Metrics by Chunk Datasets</h2>")
    f.write(metrics_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<h2>Training and Validation Loss by Chunk</h2>")
    f.write(loss_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("</body></html>")

webbrowser.open('file://' + os.path.realpath(html_output))
print(f"✅ Interactive dashboard saved to {html_output}")
