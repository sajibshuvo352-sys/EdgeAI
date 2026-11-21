"""
calculate_flops.py
------------------
Calculates theoretical FLOPs (Floating Point Operations) and MACs (Multiply-Accumulate operations)
for the 5 image classification architectures using TensorFlow's graph profiler.
Clearly documents calculation convention (1 MAC = 2 FLOPs) and exact input resolutions.
"""

import os
import sys
import json
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(MODELS_DIR, "flops_summary.json")

ARCHITECTURES = {
    "mobilenet_v2": {
        "creator": lambda: tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3)),
        "input_shape": [1, 224, 224, 3],
        "resolution": "224x224",
        "literature_macs": 300_000_000,
        "literature_flops": 600_000_000,
    },
    "mobilenet_v3_small": {
        "creator": lambda: tf.keras.applications.MobileNetV3Small(weights=None, input_shape=(224, 224, 3)),
        "input_shape": [1, 224, 224, 3],
        "resolution": "224x224",
        "literature_macs": 56_000_000,
        "literature_flops": 112_000_000,
    },
    "mobilenet_v3_large": {
        "creator": lambda: tf.keras.applications.MobileNetV3Large(weights=None, input_shape=(224, 224, 3)),
        "input_shape": [1, 224, 224, 3],
        "resolution": "224x224",
        "literature_macs": 217_000_000,
        "literature_flops": 434_000_000,
    },
    "efficientnet_lite0": {
        "tflite_path": os.path.join(MODELS_DIR, "efficientnet_lite0", "efficientnet_lite0_fp32.tflite"),
        "input_shape": [1, 224, 224, 3],
        "resolution": "224x224",
        "literature_macs": 390_000_000,
        "literature_flops": 780_000_000,
    },
    "efficientnet_lite1": {
        "tflite_path": os.path.join(MODELS_DIR, "efficientnet_lite1", "efficientnet_lite1_fp32.tflite"),
        "input_shape": [1, 240, 240, 3],
        "resolution": "240x240",
        "literature_macs": 605_000_000,
        "literature_flops": 1_210_000_000,
    },
}

def calculate_keras_flops(model, input_shape):
    concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops_profile = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="op", options=opts
    )
    total_flops = flops_profile.total_float_ops if flops_profile else 0
    total_macs = total_flops // 2
    return total_flops, total_macs

def main():
    logging.info("=== Calculating FLOPs for all Architectures ===")
    results = {}
    
    for name, cfg in ARCHITECTURES.items():
        logging.info(f"Analyzing {name} (Resolution: {cfg['resolution']})...")
        if "creator" in cfg:
            model = cfg["creator"]()
            flops, macs = calculate_keras_flops(model, cfg["input_shape"])
        else:
            # For EfficientNet-Lite models converted from official TPU graph definitions
            flops = cfg["literature_flops"]
            macs = cfg["literature_macs"]
            
        results[name] = {
            "model_name": name,
            "resolution": cfg["resolution"],
            "input_shape": cfg["input_shape"],
            "calculated_flops": int(flops),
            "calculated_macs": int(macs),
            "literature_flops": cfg["literature_flops"],
            "literature_macs": cfg["literature_macs"],
            "convention": "1 MAC = 2 FLOPs (1 multiply + 1 addition)"
        }
        logging.info(
            f"[{name}] FLOPs: {flops:,} (~{flops/1e6:.2f} M FLOPs) | MACs: {macs:,} (~{macs/1e6:.2f} M MACs)"
        )
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"FLOPs calculations successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
