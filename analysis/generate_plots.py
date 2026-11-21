"""
generate_plots.py
-----------------
Generates publication-quality scientific figures in both PNG and PDF formats:
1. FLOPs vs. median latency
2. Parameters vs. median latency
3. Model size vs. median latency
4. FLOPs vs. peak sampled PSS
5. Parameters vs. peak sampled PSS
6. Model size vs. peak sampled PSS
7. Accuracy vs. median latency
8. Accuracy vs. model artifact size
9. FP32 vs. FP16 vs. INT8 precision comparison
10. Cross-device comparison: Realme C71 vs. Poco X3 Pro
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analyze_results import load_and_validate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

COLOR_MAP = {
    "mobilenet_v2": "#1f77b4",
    "mobilenet_v3_small": "#2ca02c",
    "mobilenet_v3_large": "#ff7f0e",
    "efficientnet_lite0": "#9467bd",
    "efficientnet_lite1": "#d62728"
}

MARKER_MAP = {
    "FP32": "o",
    "FP16": "s",
    "INT8": "^"
}

def save_fig(fig, base_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{base_name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved plot: {base_name}.png and .pdf")

def plot_scatter_metric(df, x_col, y_col, x_label, y_label, title, file_name):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    devices = df["device_model"].unique()
    
    for dev in devices:
        df_dev = df[df["device_model"] == dev]
        for prec in ["FP32", "FP16", "INT8"]:
            sub = df_dev[df_dev["precision"] == prec]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                label=f"{dev} ({prec})",
                marker=MARKER_MAP.get(prec, "o"),
                alpha=0.85,
                s=70
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["model_name"].replace("mobilenet", "MN").replace("efficientnet", "Eff"),
                    (row[x_col], row[y_col]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points"
                )
                
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, file_name)

def plot_precision_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped = df.groupby(["model_name", "precision"])["median_latency_ms"].mean().unstack()
    if not grouped.empty:
        grouped.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
        ax.set_xlabel("Model Architecture")
        ax.set_ylabel("Median Latency (ms)")
        ax.set_title("Inference Latency by Precision Format (FP32 vs FP16 vs INT8)", fontweight="bold")
        ax.legend(title="Precision")
        plt.xticks(rotation=15)
        save_fig(fig, "fig9_precision_latency_comparison")

def plot_cross_device_comparison(df):
    devices = df["device_model"].unique()
    if len(devices) < 2:
        logging.info("Cross-device plot requires data from at least 2 devices.")
        return
        
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot = df.pivot_table(index="model_name", columns="device_model", values="median_latency_ms", aggfunc="mean")
    if not pivot.empty:
        pivot.plot(kind="bar", ax=ax, width=0.6, color=["#3b82f6", "#ef4444"])
        ax.set_xlabel("Model Architecture")
        ax.set_ylabel("Mean Median Latency (ms)")
        ax.set_title(f"Cross-Device Latency Comparison: {devices[0]} vs {devices[1]}", fontweight="bold")
        plt.xticks(rotation=15)
        save_fig(fig, "fig10_cross_device_comparison")

def generate_all_plots(df: pd.DataFrame):
    if df.empty:
        logging.warning("Dataframe is empty. Cannot generate plots.")
        return
        
    logging.info("Generating publication figures...")
    # 1. FLOPs vs Latency
    plot_scatter_metric(df, "flops", "median_latency_ms", "Theoretical FLOPs", "Median Latency (ms)", "FLOPs vs. Median Inference Latency", "fig1_flops_vs_latency")
    # 2. Parameters vs Latency
    plot_scatter_metric(df, "parameter_count", "median_latency_ms", "Parameter Count", "Median Latency (ms)", "Parameters vs. Median Inference Latency", "fig2_parameters_vs_latency")
    # 3. Model Size vs Latency
    plot_scatter_metric(df, "model_size_mb", "median_latency_ms", "Model Size (MB)", "Median Latency (ms)", "Model Size vs. Median Inference Latency", "fig3_model_size_vs_latency")
    # 4. FLOPs vs Peak PSS
    plot_scatter_metric(df, "flops", "peak_sampled_pss_mb", "Theoretical FLOPs", "Peak Sampled PSS (MB)", "FLOPs vs. Peak Sampled Process PSS", "fig4_flops_vs_pss")
    # 5. Parameters vs Peak PSS
    plot_scatter_metric(df, "parameter_count", "peak_sampled_pss_mb", "Parameter Count", "Peak Sampled PSS (MB)", "Parameters vs. Peak Sampled Process PSS", "fig5_parameters_vs_pss")
    # 6. Model Size vs Peak PSS
    plot_scatter_metric(df, "model_size_mb", "peak_sampled_pss_mb", "Model Size (MB)", "Peak Sampled PSS (MB)", "Model Size vs. Peak Sampled Process PSS", "fig6_model_size_vs_pss")
    # 7. Accuracy vs Latency
    plot_scatter_metric(df, "median_latency_ms", "top1_accuracy", "Median Latency (ms)", "Top-1 Accuracy (%)", "Classification Accuracy vs. Inference Latency", "fig7_accuracy_vs_latency")
    # 8. Accuracy vs Model Size
    plot_scatter_metric(df, "model_size_mb", "top1_accuracy", "Model Size (MB)", "Top-1 Accuracy (%)", "Classification Accuracy vs. Model Size", "fig8_accuracy_vs_model_size")
    # 9. Precision comparison
    plot_precision_comparison(df)
    # 10. Device comparison
    plot_cross_device_comparison(df)

def main():
    df = load_and_validate_results()
    if df.empty:
        logging.info("No experimental data found in results/. Awaiting smartphone benchmark exports.")
        return
    generate_all_plots(df)

if __name__ == "__main__":
    main()
