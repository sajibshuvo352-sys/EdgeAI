"""
ranking.py
----------
Generates theoretical vs. empirical performance rankings across architectures:
- Lowest FLOPs
- Lowest Parameter Count
- Smallest Model Size
- Lowest Median Latency
- Lowest Peak Sampled PSS
- Highest Top-1 Accuracy

Compares rankings to identify empirical rank inversions on mobile hardware.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analyze_results import load_and_validate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "reports", "tables")

def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
        
    devices = df["device_model"].unique()
    precisions = ["FP32", "FP16", "INT8"]
    all_ranking_rows = []
    
    for dev in devices:
        df_dev = df[df["device_model"] == dev]
        for prec in precisions:
            sub = df_dev[df_dev["precision"] == prec].copy()
            if sub.empty:
                continue
                
            # Rank metrics: lower is better for FLOPs, params, size, latency, PSS.
            # Higher is better for accuracy.
            sub["rank_flops"] = sub["flops"].rank(ascending=True, method="min").astype(int)
            sub["rank_parameters"] = sub["parameter_count"].rank(ascending=True, method="min").astype(int)
            sub["rank_model_size"] = sub["model_size_mb"].rank(ascending=True, method="min").astype(int)
            sub["rank_latency"] = sub["median_latency_ms"].rank(ascending=True, method="min").astype(int)
            sub["rank_pss"] = sub["peak_sampled_pss_mb"].rank(ascending=True, method="min").astype(int)
            sub["rank_accuracy"] = sub["top1_accuracy"].rank(ascending=False, method="min").astype(int)
            
            # Rank Inversion flag: Does latency rank differ from FLOPs rank?
            sub["latency_flops_inversion"] = sub["rank_latency"] != sub["rank_flops"]
            
            cols = [
                "device_model", "precision", "model_name",
                "rank_flops", "rank_parameters", "rank_model_size",
                "rank_latency", "rank_pss", "rank_accuracy",
                "latency_flops_inversion"
            ]
            all_ranking_rows.append(sub[cols])
            
    if not all_ranking_rows:
        return pd.DataFrame()
        
    return pd.concat(all_ranking_rows, ignore_index=True)

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    logging.info("=== Computing Model Performance Rankings ===")
    
    df = load_and_validate_results()
    if df.empty:
        logging.info("No empirical data found in results/. Run benchmarks on phone first.")
        return
        
    rankings_df = compute_rankings(df)
    out_csv = os.path.join(TABLES_DIR, "model_rankings.csv")
    rankings_df.to_csv(out_csv, index=False)
    logging.info(f"Saved rankings to {out_csv}")
    
    print("\n================ MODEL EFFICIENCY RANKINGS ================")
    print(rankings_df.to_string(index=False))

if __name__ == "__main__":
    main()
