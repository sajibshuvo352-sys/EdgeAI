"""
download_models.py
------------------
Downloads pretrained vision models (MobileNetV2, MobileNetV3-Small, MobileNetV3-Large,
EfficientNet-Lite0, EfficientNet-Lite1) using official pretrained ImageNet weights.
Saves native SavedModel and weight checkpoints in the EdgeAI-Benchmark models directory.
"""

import os
import sys
import tarfile
import urllib.request
import logging

# Ensure UTF-8 output encoding on Windows consoles
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

EFFICIENTNET_LITE_URLS = {
    "efficientnet_lite0": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz",
    "efficientnet_lite1": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz",
}

def download_and_extract_tar(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, "archive.tar.gz")
    logging.info(f"Downloading from {url} to {tar_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(tar_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    logging.info("Download completed. Extracting archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    logging.info(f"Extracted to {dest_dir}.")
    if os.path.exists(tar_path):
        os.remove(tar_path)

def download_keras_model(model_name: str, app_class, input_shape=(224, 224, 3)):
    target_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(target_dir, exist_ok=True)
    saved_model_dir = os.path.join(target_dir, "saved_model")
    
    if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)) > 0:
        logging.info(f"[{model_name}] SavedModel already exists at {saved_model_dir}. Skipping download.")
        return saved_model_dir
    
    logging.info(f"[{model_name}] Instantiating pretrained Keras model with ImageNet weights...")
    model = app_class(weights="imagenet", input_shape=input_shape)
    logging.info(f"[{model_name}] Exporting SavedModel to {saved_model_dir}...")
    model.export(saved_model_dir)
    logging.info(f"[{model_name}] SavedModel successfully exported.")
    return saved_model_dir

def download_efficientnet_lite(model_name: str):
    target_dir = os.path.join(MODELS_DIR, model_name)
    raw_dir = os.path.join(target_dir, "raw_checkpoint")
    if os.path.exists(raw_dir) and len(os.listdir(raw_dir)) > 0:
        logging.info(f"[{model_name}] Raw checkpoint already exists at {raw_dir}. Skipping download.")
        return raw_dir
    
    url = EFFICIENTNET_LITE_URLS[model_name]
    download_and_extract_tar(url, raw_dir)
    return raw_dir

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    logging.info("=== Starting Pretrained Model Download Phase ===")
    
    # 1. MobileNetV2
    logging.info("--- Downloading MobileNetV2 ---")
    download_keras_model("mobilenet_v2", tf.keras.applications.MobileNetV2, input_shape=(224, 224, 3))
    
    # 2. MobileNetV3 Small
    logging.info("--- Downloading MobileNetV3 Small ---")
    download_keras_model("mobilenet_v3_small", tf.keras.applications.MobileNetV3Small, input_shape=(224, 224, 3))
    
    # 3. MobileNetV3 Large
    logging.info("--- Downloading MobileNetV3 Large ---")
    download_keras_model("mobilenet_v3_large", tf.keras.applications.MobileNetV3Large, input_shape=(224, 224, 3))
    
    # 4. EfficientNet-Lite0
    logging.info("--- Downloading EfficientNet-Lite0 ---")
    download_efficientnet_lite("efficientnet_lite0")
    
    # 5. EfficientNet-Lite1
    logging.info("--- Downloading EfficientNet-Lite1 ---")
    download_efficientnet_lite("efficientnet_lite1")
    
    logging.info("=== All Pretrained Models Successfully Acquired ===")

if __name__ == "__main__":
    main()
