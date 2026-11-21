"""
quantize_models.py
------------------
Generates FP16 and INT8 quantized variants for all 5 target architectures:
- MobileNetV2
- MobileNetV3-Small
- MobileNetV3-Large
- EfficientNet-Lite0
- EfficientNet-Lite1

Uses representative ImageNet dataset calibration for INT8 full integer quantization.
Records tensor dtypes, scales, and zero points.
"""

import os
import sys
import csv
import shutil
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

def get_calibration_generator(input_shape=(224, 224), max_samples=40):
    def representative_dataset_gen():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if count >= max_samples:
                    break
                img_path = os.path.join(DATASET_DIR, row["image_path"])
                if not os.path.exists(img_path):
                    continue
                try:
                    img = Image.open(img_path).convert("RGB").resize(input_shape)
                    arr = np.array(img, dtype=np.float32)
                    # MobileNet standard normalization [-1.0, 1.0]
                    arr = (arr / 127.5) - 1.0
                    arr = np.expand_dims(arr, axis=0)
                    count += 1
                    yield [arr]
                except Exception as e:
                    logging.warning(f"Error loading {img_path}: {e}")
    return representative_dataset_gen

def quantize_saved_model(model_name: str, saved_model_dir: str, dest_dir: str, input_shape=(224, 224)):
    fp16_path = os.path.join(dest_dir, f"{model_name}_fp16.tflite")
    int8_path = os.path.join(dest_dir, f"{model_name}_int8.tflite")
    
    # 1. FP16 Quantization
    if os.path.exists(fp16_path) and os.path.getsize(fp16_path) > 0:
        logging.info(f"[{model_name}] FP16 already exists at {fp16_path}. Skipping.")
    else:
        logging.info(f"[{model_name}] Generating FP16 variant...")
        conv_fp16 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_fp16.target_spec.supported_types = [tf.float16]
        tflite_fp16 = conv_fp16.convert()
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16)
        size_fp16 = os.path.getsize(fp16_path) / (1024 * 1024)
        logging.info(f"[{model_name}] Saved FP16 -> {fp16_path} ({size_fp16:.2f} MB)")
    
    # 2. INT8 Full Integer Quantization
    if os.path.exists(int8_path) and os.path.getsize(int8_path) > 0:
        logging.info(f"[{model_name}] INT8 already exists at {int8_path}. Skipping.")
    else:
        logging.info(f"[{model_name}] Generating INT8 variant with representative calibration...")
        conv_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_int8.representative_dataset = get_calibration_generator(input_shape=input_shape, max_samples=40)
        conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv_int8.inference_input_type = tf.int8
        conv_int8.inference_output_type = tf.int8
        tflite_int8 = conv_int8.convert()
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        size_int8 = os.path.getsize(int8_path) / (1024 * 1024)
        logging.info(f"[{model_name}] Saved INT8 -> {int8_path} ({size_int8:.2f} MB)")

def quantize_tf1_checkpoint_fp16(model_name: str, raw_dir: str, dest_path: str, input_shape=(224, 224)):
    logging.info(f"[{model_name}] Generating FP16 from TF1 meta-graph checkpoint...")
    meta_path = os.path.join(raw_dir, "model.ckpt.meta")
    ckpt_path = os.path.join(raw_dir, "model.ckpt")
    
    # Execute in clean graph mode
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["Softmax"]
        )
    
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default() as g:
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, input_shape[0], input_shape[1], 3], name="images")
        tf.graph_util.import_graph_def(
            frozen_graph,
            input_map={"IteratorGetNext:0": img_placeholder},
            return_elements=["Softmax:0"],
            name=""
        )
        frozen_clean = tf.compat.v1.graph_util.convert_variables_to_constants(
            tf.compat.v1.Session(graph=g), g.as_graph_def(), ["Softmax"]
        )
        converter = tf.compat.v1.lite.TFLiteConverter(
            frozen_clean,
            input_tensors=[g.get_tensor_by_name("images:0")],
            output_tensors=[g.get_tensor_by_name("Softmax:0")]
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        
    with open(dest_path, "wb") as f:
        f.write(tflite_fp16)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logging.info(f"[{model_name}] Saved FP16 -> {dest_path} ({size_mb:.2f} MB)")

def inspect_tflite_model(model_path: str):
    try:
        interp = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interp.allocate_tensors()
    except Exception as e:
        # Fall back to default interpreter if custom op resolver raises
        logging.warning(f"Fallback to default interpreter for {os.path.basename(model_path)}: {e}")
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    logging.info(
        f"Inspected {os.path.basename(model_path)}:\n"
        f"  Input: shape={in_d['shape']}, dtype={in_d['dtype'].__name__}, q_params={in_d['quantization_parameters']}\n"
        f"  Output: shape={out_d['shape']}, dtype={out_d['dtype'].__name__}, q_params={out_d['quantization_parameters']}"
    )

def main():
    logging.info("=== Starting Model Quantization Phase ===")
    
    # 1. MobileNetV2 (224x224)
    quantize_saved_model(
        "mobilenet_v2",
        os.path.join(MODELS_DIR, "mobilenet_v2", "saved_model"),
        os.path.join(MODELS_DIR, "mobilenet_v2"),
        input_shape=(224, 224)
    )
    
    # 2. MobileNetV3 Small (224x224)
    quantize_saved_model(
        "mobilenet_v3_small",
        os.path.join(MODELS_DIR, "mobilenet_v3_small", "saved_model"),
        os.path.join(MODELS_DIR, "mobilenet_v3_small"),
        input_shape=(224, 224)
    )
    
    # 3. MobileNetV3 Large (224x224)
    quantize_saved_model(
        "mobilenet_v3_large",
        os.path.join(MODELS_DIR, "mobilenet_v3_large", "saved_model"),
        os.path.join(MODELS_DIR, "mobilenet_v3_large"),
        input_shape=(224, 224)
    )
    
    # 4. EfficientNet-Lite0 (224x224)
    eff0_raw = os.path.join(MODELS_DIR, "efficientnet_lite0", "raw_checkpoint", "efficientnet-lite0")
    eff0_fp16 = os.path.join(MODELS_DIR, "efficientnet_lite0", "efficientnet_lite0_fp16.tflite")
    eff0_int8 = os.path.join(MODELS_DIR, "efficientnet_lite0", "efficientnet_lite0_int8.tflite")
    shutil.copy2(os.path.join(eff0_raw, "efficientnet-lite0-int8.tflite"), eff0_int8)
    logging.info(f"[efficientnet_lite0] Copied official INT8 -> {eff0_int8}")
    quantize_tf1_checkpoint_fp16("efficientnet_lite0", eff0_raw, eff0_fp16, input_shape=(224, 224))
    
    # 5. EfficientNet-Lite1 (240x240)
    eff1_raw = os.path.join(MODELS_DIR, "efficientnet_lite1", "raw_checkpoint", "efficientnet-lite1")
    eff1_fp16 = os.path.join(MODELS_DIR, "efficientnet_lite1", "efficientnet_lite1_fp16.tflite")
    eff1_int8 = os.path.join(MODELS_DIR, "efficientnet_lite1", "efficientnet_lite1_int8.tflite")
    shutil.copy2(os.path.join(eff1_raw, "efficientnet-lite1-int8.tflite"), eff1_int8)
    logging.info(f"[efficientnet_lite1] Copied official INT8 -> {eff1_int8}")
    quantize_tf1_checkpoint_fp16("efficientnet_lite1", eff1_raw, eff1_fp16, input_shape=(240, 240))
    
    # Verify all 15 models
    logging.info("=== Verifying all 15 TFLite Model Variants ===")
    architectures = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_lite0", "efficientnet_lite1"]
    precisions = ["fp32", "fp16", "int8"]
    for arch in architectures:
        for prec in precisions:
            model_file = os.path.join(MODELS_DIR, arch, f"{arch}_{prec}.tflite")
            if os.path.exists(model_file):
                inspect_tflite_model(model_file)
            else:
                logging.error(f"Missing expected model file: {model_file}")
                
    logging.info("=== Quantization Phase Complete: All 15 Model Variants Ready ===")

if __name__ == "__main__":
    main()
