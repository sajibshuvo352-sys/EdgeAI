"""
analyze_results.py
------------------
Loads and validates empirical benchmarking CSV results exported from Android smartphones
(Realme C71 and Poco X3 Pro). Checks schema conformity, verifies data types, and computes
aggregate descriptive statistics per device and precision tier.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

REQUIRED_COLUMNS = [
    "device", "device_model", "chipset", "android_version", "runtime_version",
    "model_name", "precision", "input_width", "input_height", "threads",
    "dataset_name", "image_count", "top1_accuracy", "mean_latency_ms",
    "median_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms",
    "p90_latency_ms", "baseline_pss_mb", "peak_sampled_pss_mb", "memory_increase_mb",
    "model_size_bytes", "model_size_mb", "flops", "parameter_count",
    "execution_delegate", "timestamp"
]

def load_and_validate_results(results_folder: str = RESULTS_DIR) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(results_folder, "**", "*.csv"), recursive=True)
    if not all_files:
        logging.warning(f"No experimental CSV result files found in {results_folder}.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            missing = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                logging.warning(f"File {f} is missing required columns: {missing}. Skipping.")
                continue
            dfs.append(df)
            logging.info(f"Loaded {len(df)} records from {os.path.basename(f)}")
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Cast numeric types
    numeric_cols = [
        "input_width", "input_height", "threads", "image_count", "top1_accuracy",
        "mean_latency_ms", "median_latency_ms", "std_latency_ms", "min_latency_ms",
        "max_latency_ms", "p90_latency_ms", "baseline_pss_mb", "peak_sampled_pss_mb",
        "memory_increase_mb", "model_size_bytes", "model_size_mb", "flops", "parameter_count"
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
            
    # Merge theoretical metadata if theoretical fields are 0 or missing
    metadata_path = os.path.join(BASE_DIR, "models", "model_metadata.csv")
    if os.path.exists(metadata_path):
        meta_df = pd.read_csv(metadata_path)
        meta_dict = meta_df.set_index(["model_name", "precision"])[["flops", "parameter_count", "model_size_bytes", "model_size_mb"]].to_dict("index")
        for idx, row in combined.iterrows():
            key = (row["model_name"], row["precision"])
            if key in meta_dict:
                if row.get("flops", 0) == 0 or pd.isna(row.get("flops")):
                    combined.at[idx, "flops"] = meta_dict[key]["flops"]
                if row.get("parameter_count", 0) == 0 or pd.isna(row.get("parameter_count")):
                    combined.at[idx, "parameter_count"] = meta_dict[key]["parameter_count"]
                if row.get("model_size_bytes", 0) == 0 or pd.isna(row.get("model_size_bytes")):
                    combined.at[idx, "model_size_bytes"] = meta_dict[key]["model_size_bytes"]
                if row.get("model_size_mb", 0) == 0 or pd.isna(row.get("model_size_mb")):
                    combined.at[idx, "model_size_mb"] = meta_dict[key]["model_size_mb"]

    # Remove exact duplicates if any
    combined.drop_duplicates(subset=["device_model", "model_name", "precision", "threads", "timestamp"], inplace=True)
    logging.info(f"Total validated experimental records: {len(combined)}")
    return combined

def print_summary_table(df: pd.DataFrame):
    if df.empty:
        logging.info("Result dataset is currently empty. Run Android benchmarks and place CSVs into results/.")
        return
        
    devices = df["device_model"].unique()
    for dev in devices:
        sub = df[df["device_model"] == dev]
        print(f"\n================ Device: {dev} (Total Runs: {len(sub)}) ================")
        cols_to_show = ["model_name", "precision", "median_latency_ms", "peak_sampled_pss_mb", "top1_accuracy", "model_size_mb"]
        print(sub[cols_to_show].to_string(index=False))

def main():
    logging.info("=== Scanning and Validating Empirical Results ===")
    df = load_and_validate_results()
    print_summary_table(df)

if __name__ == "__main__":
    main()
