# generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--detailed", action="store_true", help="Include chunk-wise evaluation plots")
args = parser.parse_args()

# Load metrics CSV
logs_path = "../data/logs/training_metrics_log.csv"
plots_dir = "../plots"
report_path = os.path.join(plots_dir, "logbert_training_report.html")

logs = pd.read_csv(logs_path)

# Load final evaluation metrics if available
metrics = {
    "Accuracy": logs["accuracy"].dropna().values[-1] if "accuracy" in logs.columns else None,
    "Precision": logs["precision"].dropna().values[-1] if "precision" in logs.columns else None,
    "Recall": logs["recall"].dropna().values[-1] if "recall" in logs.columns else None,
    "F1 Score": logs["f1"].dropna().values[-1] if "f1" in logs.columns else None,
    "Train Loss": logs["train_loss"].dropna().values[-1] if "train_loss" in logs.columns else None,
    "Validation Loss": logs["eval_loss"].dropna().values[-1] if "eval_loss" in logs.columns else None,
}

# Prepare chunk section string
chunk_section = ""

# Plot training and validation loss
if "train_loss" in logs.columns and "eval_loss" in logs.columns:
    # Training Loss Only
    plt.figure()
    train_epoch_group = logs.loc[logs["train_loss"].notna()].groupby("epoch")["train_loss"].mean()
    plt.plot(train_epoch_group.index, train_epoch_group.values, label="Train Loss", color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "train_loss_curve.png"))
    plt.close()

    # Validation Loss Only
    plt.figure()
    val_epoch_group = logs.loc[logs["eval_loss"].notna()].groupby("epoch")["eval_loss"].mean()
    plt.plot(val_epoch_group.index, val_epoch_group.values, label="Validation Loss", color='orange')
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "val_loss_curve.png"))
    plt.close()

    # Combined View
    plt.figure()
    plt.plot(train_epoch_group.index, train_epoch_group.values, label="Train Loss", color='blue')
    plt.plot(val_epoch_group.index, val_epoch_group.values, label="Validation Loss", color='orange')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "train_val_loss_curve.png"))
    plt.close()

# Plot learning rate
if "learning_rate" in logs.columns:
    plt.figure()
    logs["learning_rate"].plot()
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "learning_rate_schedule.png"))
    plt.close()

# Plot gradient norm
if "grad_norm" in logs.columns:
    plt.figure()
    logs["grad_norm"].plot()
    plt.title("Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "gradient_norm_curve.png"))
    plt.close()

# Real metrics dashboard
if all(val is not None for val in metrics.values()):
    plt.figure(figsize=(6, 3))
    bars = plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Final Evaluation Metrics")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='bottom')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_dashboard.png"))
    plt.close()

# Plot chunk-wise evaluation metrics with best chunk and timing
chunk_eval_path = "../data/logs/chunk_eval_metrics.csv"
if args.detailed and os.path.exists(chunk_eval_path):
    chunk_section += "<h2>📊 Chunk-wise Evaluation Metrics</h2>"
    chunk_logs = pd.read_csv(chunk_eval_path)
    if "chunk" in chunk_logs.columns and "eval_loss" in chunk_logs.columns:
        best_idx = chunk_logs["eval_loss"].idxmin()
        best_chunk = chunk_logs.loc[best_idx, "chunk"]
        chunk_section += f"<p><strong>🌟 Best Chunk:</strong> Chunk {int(best_chunk)} with eval_loss {chunk_logs.loc[best_idx, 'eval_loss']:.4f}</p>"
    if "duration" in chunk_logs.columns:
        plt.figure()
        plt.plot(chunk_logs["chunk"], chunk_logs["duration"], marker='o')
        plt.title("Training Time per Chunk (seconds)")
        plt.xlabel("Chunk")
        plt.ylabel("Duration (s)")
        plt.grid(True)
        time_path = os.path.join(plots_dir, "chunk_durations.png")
        plt.savefig(time_path)
        plt.close()
        chunk_section += f"<img src='{os.path.basename(time_path)}' width='100%'>"
    for metric in ["eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]:
        if metric in chunk_logs.columns:
            plt.figure()
            plt.plot(chunk_logs["chunk"], chunk_logs[metric], marker='o')
            plt.title(f"Chunk-wise {metric.replace('_', ' ').title()}")
            plt.xlabel("Chunk")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plot_path = os.path.join(plots_dir, f"chunk_{metric}.png")
            plt.savefig(plot_path)
            plt.close()
            chunk_section += f"<img src='{os.path.basename(plot_path)}' width='100%'>"
    if "train_loss" in chunk_logs.columns and "eval_loss" in chunk_logs.columns:
        plt.figure()
        plt.plot(chunk_logs["chunk"], chunk_logs["train_loss"], label="Train Loss", marker='o')
        plt.plot(chunk_logs["chunk"], chunk_logs["eval_loss"], label="Validation Loss", marker='o')
        plt.title("Chunk-wise Training vs Validation Loss")
        plt.xlabel("Chunk")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, "chunk_train_val_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()
        chunk_section += f"<img src='{os.path.basename(loss_plot_path)}' width='100%'>"

# Build HTML content
html = f"""
<html>
<head><title>📊 LogBERT Evaluation Report</title></head>
<body>
    <h1>LogBERT Ceph Anomaly Detection Report</h1>
    <h2>✅ Final Evaluation Metrics</h2>
    <ul>
"""
for metric, value in metrics.items():
    if value is not None:
        html += f"<li><strong>{metric}:</strong> {value:.4f}</li>"
html += f"""
    </ul>
    <h2>📈 Metrics Dashboard</h2>
    <img src='metrics_dashboard.png' width='100%'>
    <h2>📉 Training Loss</h2>
    <img src='train_loss_curve.png' width='100%'>
    <h2>📉 Validation Loss</h2>
    <img src='val_loss_curve.png' width='100%'>
    <h2>📉 Train and Validation Loss Curve</h2>
    <img src='train_val_loss_curve.png' width='100%'>
    <h2>📐 Learning Rate Schedule</h2>
    <img src='learning_rate_schedule.png' width='100%'>
    <h2>🌀 Gradient Norm</h2>
    <img src='gradient_norm_curve.png' width='100%'>
    {chunk_section}
</body>
</html>
"""

# Write to file
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Report generated: {report_path}")

import webbrowser
webbrowser.open(report_path)
