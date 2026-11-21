# Edge AI Benchmark: Model Performance Analysis on Android Smartphones

An empirical research project investigating whether theoretical model-efficiency metrics (FLOPs, parameter count, model artifact size) accurately predict real-world inference latency, memory consumption (PSS), and accuracy on Android smartphones across different hardware tiers and precision formats.

---

## Hardware Testbeds

* **Development PC:** AMD Ryzen 5 5600G (6 Cores / 12 Threads, Zen 3), 16 GB DDR4 RAM, Windows 10/11 x64, Integrated AMD Radeon Graphics.
* **Target Mobile Smartphones:**
  1. **Realme C71:** Entry-level / Budget smartphone tier.
  2. **Poco X3 Pro:** Higher-performance smartphone tier (Qualcomm Snapdragon 860).

---

## Model Matrix (5 Architectures $\times$ 3 Precisions = 15 Artifacts)

All 15 models have been generated, quantized, and validated with real ImageNet validation data:

| Architecture | Input Resolution | FP32 File Size | FP16 File Size | INT8 File Size | Theoretical FLOPs | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | $224 \times 224 \times 3$ | 13.34 MB | 6.70 MB | 3.83 MB | 615,292,208 | 3,538,984 |
| **MobileNetV3-Small** | $224 \times 224 \times 3$ | 9.72 MB | 4.89 MB | 2.75 MB | 119,247,928 | 2,554,968 |
| **MobileNetV3-Large** | $224 \times 224 \times 3$ | 20.91 MB | 10.49 MB | 5.77 MB | 448,572,872 | 5,507,432 |
| **EfficientNet-Lite0** | $224 \times 224 \times 3$ | 17.71 MB | 8.87 MB | 5.17 MB | 780,000,000 | 4,696,552 |
| **EfficientNet-Lite1** | $240 \times 240 \times 3$ | 20.62 MB | 10.33 MB | 6.11 MB | 1,210,000,000 | 5,465,000 |

* Complete metadata is registered in `models/model_metadata.json` and `models/model_metadata.csv`.
* Validation report on real ImageNet samples is stored in `models/validation_report.json`.

---

## Project Structure

```
EdgeAI-Benchmark/
│
├── model_preparation/          # Complete preparation, profiling & validation scripts
│   ├── download_models.py      # Pretrained weights acquisition
│   ├── convert_models.py       # Baseline FP32 TFLite conversion
│   ├── quantize_models.py      # FP16 & INT8 full integer quantization with calibration
│   ├── calculate_flops.py      # Graph profiler FLOPs & MACs calculation
│   ├── calculate_parameters.py # Trainable & total parameter counts
│   ├── validate_models.py      # Numerical validation, Top-5 check, cosine similarity
│   └── generate_model_metadata.py # Metadata export (JSON & CSV)
│
├── models/                     # 15 .tflite artifacts + metadata registries
│   ├── mobilenet_v2/
│   ├── mobilenet_v3_small/
│   ├── mobilenet_v3_large/
│   ├── efficientnet_lite0/
│   ├── efficientnet_lite1/
│   ├── model_metadata.json
│   ├── model_metadata.csv
│   └── validation_report.json
│
├── dataset/                    # Standardized validation dataset
│   ├── images/                 # 78 verified ImageNet sample JPEGs
│   ├── labels.csv              # Manifest with stable image IDs & ground-truth class IDs
│   └── prepare_dataset.py      # Dataset preparation script
│
├── android_app/                # Native Android application (Kotlin, TFLite Interpreter)
│   ├── app/src/main/
│   │   ├── java/com/edgeai/benchmark/
│   │   │   ├── MainActivity.kt         # UI, execution orchestrator, matrix runner
│   │   │   ├── ModelManager.kt         # TFLite model lifecycle & thread control
│   │   │   ├── ImageProcessor.kt       # Bitmap loading, resizing, float/int8 preprocessing
│   │   │   ├── InferenceRunner.kt      # Monotonic clock timer, warm-ups, accuracy evaluator
│   │   │   ├── PerformanceMonitor.kt   # Hardware telemetry & Debug.MemoryInfo PSS sampling
│   │   │   └── ResultsExporter.kt      # Structured CSV and JSON data export
│   │   ├── assets/                     # Self-contained offline assets
│   │   │   ├── models/                 # All 15 .tflite models
│   │   │   ├── dataset/                # Validation images & labels.csv
│   │   │   └── model_metadata.json     # Architecture metadata
│   │   └── res/                        # Research UI layouts and dark themes
│   ├── build.gradle.kts
│   └── settings.gradle.kts
│
├── results/                    # Storage for phone exported CSV/JSON data
│   ├── realme_c71/
│   └── poco_x3_pro/
│
├── analysis/                   # Statistical analysis & visualization pipeline
│   ├── analyze_results.py      # CSV validation & schema checking
│   ├── correlation.py          # Spearman & Kendall rank correlations
│   ├── ranking.py              # Theoretical vs empirical ranking inversion detector
│   └── generate_plots.py       # Generates 10 scientific figures (PNG & PDF)
│
├── reports/                    # Generated tables, figures & summaries
│   ├── figures/
│   ├── tables/
│   └── summaries/
│
├── REPRODUCIBILITY_GUIDE.md    # Detailed phone execution & compilation instructions
├── requirements.txt            # Pinned Python dependencies
└── README.md
```

---

## Step-by-Step Workflow & Results

1. **Model Preparation (Completed & Verified on PC):**
   * Downloaded, converted, and quantized all 15 models.
   * Calculated exact FLOPs and parameter counts.
   * Validated prediction outputs against ImageNet ground truth.
   * Embedded all models, datasets, and metadata into Android app assets.

2. **Mobile Smartphone Benchmarking (Completed on Physical Devices):**
   * Executed on **Xiaomi Poco X3 Pro** (`M2102J20SG`, Snapdragon 860, Android 12).
   * Executed on **Realme C71** (`RMX5303`, Unisoc UMS9230E, Android 16).
   * Collected 30 experimental runs (15 per phone) with 10 warm-up runs and 100 measured iterations per model using monotonic high-resolution timing and sampled PSS memory monitoring.

3. **Statistical Analysis & Publication Figures:**
   * Computed Spearman ($\rho$) and Kendall ($\tau$) rank correlation coefficients across all precision tiers.
   * Identified architectural ranking inversions under INT8 quantization.
   * Generated 10 publication-quality figures (in both high-res 300 DPI PNG and vector PDF format) located in `reports/figures/`.
   * Complete academic paper draft with hypothesis testing available in [reports/RESEARCH_REPORT.md](reports/RESEARCH_REPORT.md).

---

## Key Experimental Findings

| Architecture | Precision | Poco X3 Pro (Snapdragon 860) | Realme C71 (Unisoc UMS9230E) | Budget Slowdown Factor | Top-1 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 Small** | INT8 | **4.11 ms** | **9.44 ms** | 2.30x | — |
| **MobileNetV3 Large** | INT8 | **10.33 ms** | **25.04 ms** | 2.42x | 1.3% |
| **MobileNetV2** | INT8 | **9.78 ms** | **21.90 ms** | 2.24x | 83.3% |
| **EfficientNet-Lite0** | INT8 | **8.47 ms** | **23.21 ms** | 2.74x | 84.6% |
| **EfficientNet-Lite1** | INT8 | **12.31 ms** | **38.36 ms** | 3.12x | 80.8% |

* **FLOPs Predictability:** Within FP32, FLOPs correlate almost perfectly with latency ($\rho = 1.000$ on Poco X3 Pro, $\rho = 0.900$ on Realme C71).
* **Quantization Inversion:** On INT8, EfficientNet-Lite0 (8.47 ms) runs faster than MobileNetV2 (9.78 ms) despite having 27% higher FLOPs, proving integer kernel tiling and memory locality dominate compute complexity.
* **INT8 Efficiency:** Delivers a **1.45x–1.98x latency reduction** with less than 1.3% accuracy loss and a 71% reduction in storage size.
* **Budget Hardware Penalty:** The budget tier is 2.3x–3.2x slower, with latency more heavily dependent on model file size ($\rho = 1.000$) due to narrower memory bus bandwidth.
