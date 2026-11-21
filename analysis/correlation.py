"""
correlation.py
--------------
Computes Spearman's rank correlation and Kendall's tau rank correlation between
theoretical efficiency metrics (FLOPs, parameters, model artifact size) and
empirical deployment metrics (median latency, peak sampled PSS).

Separates architecture-level analyses from pooled precision analyses to maintain
statistical validity regarding degrees of freedom.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from scipy import stats

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analyze_results import load_and_validate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

METRIC_PAIRS = [
    ("flops", "median_latency_ms", "FLOPs vs Median Latency"),
    ("parameter_count", "median_latency_ms", "Parameters vs Median Latency"),
    ("model_size_mb", "median_latency_ms", "Model Size vs Median Latency"),
    ("flops", "peak_sampled_pss_mb", "FLOPs vs Peak Sampled PSS"),
    ("parameter_count", "peak_sampled_pss_mb", "Parameters vs Peak Sampled PSS"),
    ("model_size_mb", "peak_sampled_pss_mb", "Model Size vs Peak Sampled PSS"),
]

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    
    if df.empty:
        logging.warning("No data provided to correlation analysis.")
        return pd.DataFrame()
        
    devices = df["device_model"].unique()
    precisions = ["FP32", "FP16", "INT8", "ALL_POOLED"]
    
    for dev in devices:
        df_dev = df[df["device_model"] == dev]
        
        for prec in precisions:
            if prec == "ALL_POOLED":
                sub = df_dev
                scope_note = "Pooled precision variants (Exploratory; note architectural auto-correlation)"
            else:
                sub = df_dev[df_dev["precision"] == prec]
                scope_note = f"Single precision tier ({prec}); N = unique architectures"
                
            n = len(sub)
            if n < 3:
                continue
                
            for x_col, y_col, label in METRIC_PAIRS:
                x = sub[x_col].values
                y = sub[y_col].values
                
                # Check variance
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                    
                spearman_res = stats.spearmanr(x, y)
                kendall_res = stats.kendalltau(x, y)
                
                records.append({
                    "device": dev,
                    "precision_scope": prec,
                    "sample_size_N": n,
                    "independent_metric": x_col,
                    "dependent_metric": y_col,
                    "comparison_label": label,
                    "spearman_rho": round(float(spearman_res.statistic), 4),
                    "spearman_pvalue": round(float(spearman_res.pvalue), 4),
                    "kendall_tau": round(float(kendall_res.statistic), 4),
                    "kendall_pvalue": round(float(kendall_res.pvalue), 4),
                    "statistical_note": scope_note
                })
                
    result_df = pd.DataFrame(records)
    return result_df

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    logging.info("=== Running Rank Correlation Analysis (Spearman & Kendall) ===")
    
    df = load_and_validate_results()
    if df.empty:
        logging.info("Awaiting empirical phone data in results/. To test analysis pipeline, run Android benchmark.")
        return
        
    corr_df = calculate_correlations(df)
    output_csv = os.path.join(TABLES_DIR, "correlation_analysis.csv")
    corr_df.to_csv(output_csv, index=False)
    logging.info(f"Saved correlation analysis table to {output_csv}")
    
    print("\n================ RANK CORRELATION RESULTS ================")
    print(corr_df[["device", "precision_scope", "comparison_label", "sample_size_N", "spearman_rho", "kendall_tau"]].to_string(index=False))

if __name__ == "__main__":
    main()
