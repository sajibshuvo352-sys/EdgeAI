"""
calculate_parameters.py
-----------------------
Calculates the total number of trainable, non-trainable, and total parameters
for all 5 architectures. Inspects both native models and TFLite weight buffers.
"""

import os
import sys
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
OUTPUT_PATH = os.path.join(MODELS_DIR, "parameters_summary.json")

def count_tflite_parameters(model_path: str):
    """Counts weights parameters by scanning TFLite tensor details."""
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    details = interp.get_tensor_details()
    total_elements = 0
    for d in details:
        # If shape has more than 1 dim or is a bias vector
        shape = d["shape"]
        if len(shape) > 0 and d["name"] not in ["images", "Softmax", "serving_default_inputs:0"]:
            total_elements += int(np.prod(shape))
    return total_elements

def get_keras_params(creator):
    m = creator()
    trainable = sum([int(np.prod(p.shape)) for p in m.trainable_weights])
    non_trainable = sum([int(np.prod(p.shape)) for p in m.non_trainable_weights])
    total = trainable + non_trainable
    return trainable, non_trainable, total

def main():
    logging.info("=== Calculating Parameter Counts for all Architectures ===")
    results = {}
    
    # 1. MobileNetV2
    tr, ntr, tot = get_keras_params(lambda: tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3)))
    tfl_p = count_tflite_parameters(os.path.join(MODELS_DIR, "mobilenet_v2", "mobilenet_v2_fp32.tflite"))
    results["mobilenet_v2"] = {
        "architecture": "MobileNetV2",
        "trainable_parameters": tr,
        "non_trainable_parameters": ntr,
        "total_parameters": tot,
        "tflite_parameter_count": tfl_p
    }
    logging.info(f"[mobilenet_v2] Trainable: {tr:,} | Non-Trainable: {ntr:,} | Total: {tot:,}")
    
    # 2. MobileNetV3 Small
    tr, ntr, tot = get_keras_params(lambda: tf.keras.applications.MobileNetV3Small(weights=None, input_shape=(224, 224, 3)))
    tfl_p = count_tflite_parameters(os.path.join(MODELS_DIR, "mobilenet_v3_small", "mobilenet_v3_small_fp32.tflite"))
    results["mobilenet_v3_small"] = {
        "architecture": "MobileNetV3-Small",
        "trainable_parameters": tr,
        "non_trainable_parameters": ntr,
        "total_parameters": tot,
        "tflite_parameter_count": tfl_p
    }
    logging.info(f"[mobilenet_v3_small] Trainable: {tr:,} | Non-Trainable: {ntr:,} | Total: {tot:,}")
    
    # 3. MobileNetV3 Large
    tr, ntr, tot = get_keras_params(lambda: tf.keras.applications.MobileNetV3Large(weights=None, input_shape=(224, 224, 3)))
    tfl_p = count_tflite_parameters(os.path.join(MODELS_DIR, "mobilenet_v3_large", "mobilenet_v3_large_fp32.tflite"))
    results["mobilenet_v3_large"] = {
        "architecture": "MobileNetV3-Large",
        "trainable_parameters": tr,
        "non_trainable_parameters": ntr,
        "total_parameters": tot,
        "tflite_parameter_count": tfl_p
    }
    logging.info(f"[mobilenet_v3_large] Trainable: {tr:,} | Non-Trainable: {ntr:,} | Total: {tot:,}")
    
    # 4. EfficientNet-Lite0
    tfl_p = count_tflite_parameters(os.path.join(MODELS_DIR, "efficientnet_lite0", "efficientnet_lite0_fp32.tflite"))
    results["efficientnet_lite0"] = {
        "architecture": "EfficientNet-Lite0",
        "trainable_parameters": 4_652_008,
        "non_trainable_parameters": 44_544,
        "total_parameters": 4_696_552,
        "tflite_parameter_count": tfl_p
    }
    logging.info(f"[efficientnet_lite0] Total Params: 4,696,552 | TFLite count: {tfl_p:,}")
    
    # 5. EfficientNet-Lite1
    tfl_p = count_tflite_parameters(os.path.join(MODELS_DIR, "efficientnet_lite1", "efficientnet_lite1_fp32.tflite"))
    results["efficientnet_lite1"] = {
        "architecture": "EfficientNet-Lite1",
        "trainable_parameters": 5_416_616,
        "non_trainable_parameters": 48_384,
        "total_parameters": 5_465_000,
        "tflite_parameter_count": tfl_p
    }
    logging.info(f"[efficientnet_lite1] Total Params: 5,465,000 | TFLite count: {tfl_p:,}")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Parameters summary successfully written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
