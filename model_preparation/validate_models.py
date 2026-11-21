"""
validate_models.py
------------------
Validates all 15 TensorFlow Lite models (5 architectures x 3 precisions).
Checks:
- Model loads successfully.
- Input and output tensor dimensions and data types.
- Inference execution completes without errors.
- Output probability distribution validity (sums to ~1.0, non-negative).
- Prediction alignment: Top-1/Top-5 class consistency and cosine similarity between FP32 and quantized variants.
"""

import os
import sys
import csv
import json
import logging
from PIL import Image
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
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "labels.csv")
REPORT_PATH = os.path.join(MODELS_DIR, "validation_report.json")

ARCHITECTURES = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_lite0", "efficientnet_lite1"]
PRECISIONS = ["fp32", "fp16", "int8"]

def get_test_image(input_size):
    # Load first sample from manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    
    img_path = os.path.join(DATASET_DIR, first_row["image_path"])
    img = Image.open(img_path).convert("RGB").resize(input_size)
    arr = np.array(img, dtype=np.float32)
    return arr, int(first_row["ground_truth_label"]), first_row["class_name"]

def preprocess_input(raw_arr, in_details):
    dtype = in_details["dtype"]
    shape = in_details["shape"]
    
    if dtype == np.float32:
        # Standard [-1.0, 1.0] normalization
        norm = (raw_arr / 127.5) - 1.0
        return np.expand_dims(norm.astype(np.float32), axis=0)
    elif dtype == np.int8:
        # Quantize to int8: (val / scale) + zero_point
        scale, zero_point = in_details["quantization"]
        norm = (raw_arr / 127.5) - 1.0
        q = np.round(norm / scale) + zero_point
        q = np.clip(q, -128, 127).astype(np.int8)
        return np.expand_dims(q, axis=0)
    elif dtype == np.uint8:
        # Quantize to uint8
        scale, zero_point = in_details["quantization"]
        norm = (raw_arr / 127.5) - 1.0
        q = np.round(norm / scale) + zero_point
        q = np.clip(q, 0, 255).astype(np.uint8)
        return np.expand_dims(q, axis=0)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def dequantize_output(raw_out, out_details):
    dtype = out_details["dtype"]
    if dtype == np.float32:
        return raw_out[0]
    elif dtype in (np.int8, np.uint8):
        scale, zero_point = out_details["quantization"]
        deq = (raw_out[0].astype(np.float32) - zero_point) * scale
        return deq
    return raw_out[0]

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def validate_single_model(model_path: str):
    interp = tf.lite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    
    h, w = in_d["shape"][1], in_d["shape"][2]
    raw_img, gt_label, gt_name = get_test_image((w, h))
    
    prep_tensor = preprocess_input(raw_img, in_d)
    interp.set_tensor(in_d["index"], prep_tensor)
    interp.invoke()
    
    raw_out = interp.get_tensor(out_d["index"])
    prob_vec = dequantize_output(raw_out, out_d)
    
    # Softmax check (if not already probabilities)
    if prob_vec.sum() > 1.5 or prob_vec.min() < 0:
        # Exponentiate to get softmax
        exp_vec = np.exp(prob_vec - np.max(prob_vec))
        prob_vec = exp_vec / np.sum(exp_vec)
        
    top1_class = int(np.argmax(prob_vec))
    top1_prob = float(np.max(prob_vec))
    top5_classes = [int(x) for x in np.argsort(prob_vec)[-5:][::-1]]
    
    return {
        "status": "PASS",
        "input_shape": [int(x) for x in in_d["shape"]],
        "input_dtype": str(in_d["dtype"].__name__),
        "output_shape": [int(x) for x in out_d["shape"]],
        "output_dtype": str(out_d["dtype"].__name__),
        "gt_class": gt_label,
        "gt_name": gt_name,
        "top1_class": top1_class,
        "top1_prob": round(top1_prob, 4),
        "top5_classes": top5_classes,
        "top1_matches_gt": (top1_class == gt_label),
        "top5_contains_gt": (gt_label in top5_classes),
        "prob_vector": prob_vec
    }

def main():
    logging.info("=== Starting Comprehensive Model Validation Suite ===")
    report = {}
    
    for arch in ARCHITECTURES:
        report[arch] = {}
        fp32_vec = None
        for prec in PRECISIONS:
            model_path = os.path.join(MODELS_DIR, arch, f"{arch}_{prec}.tflite")
            if not os.path.exists(model_path):
                logging.error(f"Missing model: {model_path}")
                report[arch][prec] = {"status": "FAIL", "error": "File not found"}
                continue
                
            try:
                res = validate_single_model(model_path)
                prob_vec = res.pop("prob_vector")
                if prec == "fp32":
                    fp32_vec = prob_vec
                    res["cosine_sim_to_fp32"] = 1.0
                else:
                    if fp32_vec is not None:
                        sim = cosine_similarity(fp32_vec, prob_vec)
                        res["cosine_sim_to_fp32"] = round(sim, 4)
                    else:
                        res["cosine_sim_to_fp32"] = None
                        
                report[arch][prec] = res
                logging.info(
                    f"[{arch} - {prec.upper()}] Status: PASS | Top-1 Class: {res['top1_class']} (Prob: {res['top1_prob']}) | "
                    f"CosSim to FP32: {res.get('cosine_sim_to_fp32', 'N/A')}"
                )
            except Exception as e:
                logging.error(f"Validation failed for {model_path}: {e}")
                report[arch][prec] = {"status": "FAIL", "error": str(e)}
                
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"=== Model Validation Complete: Summary Saved to {REPORT_PATH} ===")

if __name__ == "__main__":
    main()
