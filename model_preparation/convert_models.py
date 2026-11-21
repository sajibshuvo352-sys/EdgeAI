"""
convert_models.py
-----------------
Converts the 5 base image classification architectures into baseline FP32
TensorFlow Lite models. Validates conversion integrity and inspects input/output
tensor shapes and dtypes.
"""

import os
import sys
import shutil
import logging

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

MODELS_CONFIG = {
    "mobilenet_v2": {
        "type": "saved_model",
        "saved_model_path": os.path.join(MODELS_DIR, "mobilenet_v2", "saved_model"),
        "dest_tflite": os.path.join(MODELS_DIR, "mobilenet_v2", "mobilenet_v2_fp32.tflite"),
    },
    "mobilenet_v3_small": {
        "type": "saved_model",
        "saved_model_path": os.path.join(MODELS_DIR, "mobilenet_v3_small", "saved_model"),
        "dest_tflite": os.path.join(MODELS_DIR, "mobilenet_v3_small", "mobilenet_v3_small_fp32.tflite"),
    },
    "mobilenet_v3_large": {
        "type": "saved_model",
        "saved_model_path": os.path.join(MODELS_DIR, "mobilenet_v3_large", "saved_model"),
        "dest_tflite": os.path.join(MODELS_DIR, "mobilenet_v3_large", "mobilenet_v3_large_fp32.tflite"),
    },
    "efficientnet_lite0": {
        "type": "official_tflite",
        "src_tflite": os.path.join(MODELS_DIR, "efficientnet_lite0", "raw_checkpoint", "efficientnet-lite0", "efficientnet-lite0-fp32.tflite"),
        "dest_tflite": os.path.join(MODELS_DIR, "efficientnet_lite0", "efficientnet_lite0_fp32.tflite"),
    },
    "efficientnet_lite1": {
        "type": "official_tflite",
        "src_tflite": os.path.join(MODELS_DIR, "efficientnet_lite1", "raw_checkpoint", "efficientnet-lite1", "efficientnet-lite1-fp32.tflite"),
        "dest_tflite": os.path.join(MODELS_DIR, "efficientnet_lite1", "efficientnet_lite1_fp32.tflite"),
    },
}

def convert_saved_model_to_fp32(model_name: str, saved_model_dir: str, output_path: str):
    logging.info(f"[{model_name}] Converting SavedModel to FP32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Baseline FP32 preserves exact float32 weights and activations
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"[{model_name}] Converted successfully -> {output_path} ({size_mb:.2f} MB)")

def copy_official_tflite(model_name: str, src_path: str, dest_path: str):
    logging.info(f"[{model_name}] Copying official FP32 TFLite artifact...")
    shutil.copy2(src_path, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logging.info(f"[{model_name}] Copied successfully -> {dest_path} ({size_mb:.2f} MB)")

def verify_tflite(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    logging.info(f"Verified {os.path.basename(model_path)}: Input shape={in_d['shape']}, dtype={in_d['dtype']} | Output shape={out_d['shape']}, dtype={out_d['dtype']}")

def main():
    logging.info("=== Starting FP32 Model Conversion Phase ===")
    for model_name, cfg in MODELS_CONFIG.items():
        if cfg["type"] == "saved_model":
            convert_saved_model_to_fp32(model_name, cfg["saved_model_path"], cfg["dest_tflite"])
        elif cfg["type"] == "official_tflite":
            copy_official_tflite(model_name, cfg["src_tflite"], cfg["dest_tflite"])
        verify_tflite(cfg["dest_tflite"])
    logging.info("=== All 5 Models Successfully Converted to FP32 TFLite ===")

if __name__ == "__main__":
    main()
