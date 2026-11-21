"""
generate_model_metadata.py
--------------------------
Extracts and generates comprehensive model metadata for all 15 model configurations.
Aggregates:
- Model name, architecture, input resolution
- Precision (FP32, FP16, INT8)
- FLOPs and parameter counts (trainable and total)
- File size in bytes and megabytes
- Input/output tensor shapes, dtypes, quantization scales, and zero points
- Conversion configuration

Saves output as:
- models/model_metadata.json
- models/model_metadata.csv
"""

import os
import sys
import csv
import json
import logging
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FLOPS_PATH = os.path.join(MODELS_DIR, "flops_summary.json")
PARAMS_PATH = os.path.join(MODELS_DIR, "parameters_summary.json")
OUTPUT_JSON = os.path.join(MODELS_DIR, "model_metadata.json")
OUTPUT_CSV = os.path.join(MODELS_DIR, "model_metadata.csv")

ARCHITECTURES = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_lite0", "efficientnet_lite1"]
PRECISIONS = ["fp32", "fp16", "int8"]

def main():
    logging.info("=== Generating Complete Model Metadata Registry ===")
    
    with open(FLOPS_PATH, "r", encoding="utf-8") as f:
        flops_data = json.load(f)
        
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params_data = json.load(f)
        
    records = []
    
    for arch in ARCHITECTURES:
        arch_flops = flops_data[arch]["calculated_flops"]
        arch_macs = flops_data[arch]["calculated_macs"]
        arch_params = params_data[arch]["total_parameters"]
        arch_trainable = params_data[arch]["trainable_parameters"]
        
        for prec in PRECISIONS:
            filename = f"{arch}_{prec}.tflite"
            file_path = os.path.join(MODELS_DIR, arch, filename)
            
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue
                
            size_bytes = os.path.getsize(file_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            
            interp = tf.lite.Interpreter(
                model_path=file_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            interp.allocate_tensors()
            in_d = interp.get_input_details()[0]
            out_d = interp.get_output_details()[0]
            
            in_q = in_d["quantization_parameters"]
            out_q = out_d["quantization_parameters"]
            
            in_scale = float(in_q["scales"][0]) if len(in_q["scales"]) > 0 else 0.0
            in_zp = int(in_q["zero_points"][0]) if len(in_q["zero_points"]) > 0 else 0
            out_scale = float(out_q["scales"][0]) if len(out_q["scales"]) > 0 else 0.0
            out_zp = int(out_q["zero_points"][0]) if len(out_q["zero_points"]) > 0 else 0
            
            conv_config = {
                "fp32": "Standard TFLite float conversion",
                "fp16": "Post-training float16 weight quantization",
                "int8": "Full integer post-training quantization with representative ImageNet dataset"
            }[prec]
            
            rec = {
                "model_identifier": f"{arch}_{prec}",
                "model_name": arch,
                "precision": prec.upper(),
                "input_width": int(in_d["shape"][2]),
                "input_height": int(in_d["shape"][1]),
                "input_channels": int(in_d["shape"][3]),
                "input_tensor_name": in_d["name"],
                "input_tensor_dtype": in_d["dtype"].__name__,
                "input_scale": in_scale,
                "input_zero_point": in_zp,
                "output_tensor_name": out_d["name"],
                "output_tensor_dtype": out_d["dtype"].__name__,
                "output_scale": out_scale,
                "output_zero_point": out_zp,
                "flops": arch_flops,
                "macs": arch_macs,
                "parameter_count": arch_params,
                "trainable_parameters": arch_trainable,
                "model_size_bytes": size_bytes,
                "model_size_mb": size_mb,
                "conversion_config": conv_config,
                "rel_path": os.path.join("models", arch, filename).replace("\\", "/")
            }
            records.append(rec)
            
    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
        
    # Save CSV
    fieldnames = list(records[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    logging.info(f"Saved {len(records)} metadata records to {OUTPUT_JSON} and {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
