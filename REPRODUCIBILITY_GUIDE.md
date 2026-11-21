# Empirical Benchmarking & Reproducibility Protocol

This guide details the complete protocol for compiling the Android application, deploying it to physical devices, collecting empirical measurements, and running the statistical analysis pipeline.

---

## 1. Experimental Setup & Protocol Controls

To ensure strict scientific rigor, reproducibility, and prevent external confounders across both smartphones (**Realme C71** and **Poco X3 Pro**):

1. **CPU Inference Only:** GPU (OpenCL/Vulkan) and NNAPI hardware acceleration are disabled. Inference executes deterministically on CPU using standard TensorFlow Lite builtin operators.
2. **Fixed Thread Allocation:** Set CPU threads to **4 threads** on both smartphones.
3. **Battery & Thermal State:**
   * Benchmark with battery charge $\ge 50\%$.
   * **Do not keep the phone connected to a fast charger** during the benchmark run to avoid battery charging thermal dissipation.
   * Allow the device to cool to room ambient temperature ($22^\circ\text{C} - 25^\circ\text{C}$) before starting a run.
4. **Quiescent System State:**
   * Close all background applications via recent apps list.
   * Enable Airplane Mode or Do Not Disturb to eliminate background push notifications and CPU interrupts.
5. **Warm-up & Iteration Counts:**
   * 10 warm-up runs executed and discarded per model to stabilize CPU dynamic voltage and frequency scaling (DVFS) and JIT compiler state.
   * Exactly 100 measured iterations timed using high-resolution monotonic clock (`System.nanoTime()`).
6. **Sampling Window:**
   * Process Proportional Set Size (PSS) sampled before model loading (baseline) and periodically during inference loops via Android `Debug.MemoryInfo`.

---

## 2. Compiling the Android Application

### Option A: Using Android Studio (Recommended)
1. Launch **Android Studio**.
2. Select **Open** and select the folder:
   `f:\Edge AIML Model Performance Analysis on Android Smartphones\EdgeAI-Benchmark\android_app`
3. Allow Gradle to sync dependencies (`org.tensorflow:tensorflow-lite:2.16.1`).
4. Connect your phone via USB with **USB Debugging** enabled in Developer Options.
5. Click the green **Run** button (or press `Shift + F10`) to compile and install directly onto your smartphone.
6. Alternatively, go to **Build -> Build Bundle(s) / APK(s) -> Build APK(s)** to generate `app-debug.apk`.

### Option B: Using Command Line (Gradle)
If Android SDK and JDK 17+ are installed:
```powershell
cd "f:\Edge AIML Model Performance Analysis on Android Smartphones\EdgeAI-Benchmark\android_app"
gradle assembleDebug
```
The output APK is generated at:
`android_app/app/build/outputs/apk/debug/app-debug.apk`

---

## 3. Smartphone Execution Procedure

Run the test suite on each phone:

### Step 1: Realme C71 (Budget Smartphone)
1. Install and open **Edge AI Benchmark**.
2. Confirm the **DEVICE TELEMETRY** card displays the manufacturer, model, and SoC.
3. Tap **Run Full Matrix (15 Models)**.
4. The application will sequentially benchmark all 15 configurations:
   * MobileNetV2 (FP32, FP16, INT8)
   * MobileNetV3 Small (FP32, FP16, INT8)
   * MobileNetV3 Large (FP32, FP16, INT8)
   * EfficientNet-Lite0 (FP32, FP16, INT8)
   * EfficientNet-Lite1 (FP32, FP16, INT8)
5. When complete, tap **Export CSV / JSON Results**.
6. Retrieve the exported file from your phone storage:
   `Android/data/com.edgeai.benchmark/files/benchmark_results_*.csv`
7. Copy this file into your PC project directory:
   `EdgeAI-Benchmark/results/realme_c71/`

### Step 2: Poco X3 Pro (Performance Tier Smartphone)
1. Install and open **Edge AI Benchmark** on the Poco X3 Pro.
2. Confirm device telemetry recognizes the Snapdragon 860 chipset and exact hardware variant.
3. Tap **Run Full Matrix (15 Models)**.
4. Tap **Export CSV / JSON Results**.
5. Copy the exported file from the phone into:
   `EdgeAI-Benchmark/results/poco_x3_pro/`

---

## 4. Running the Statistical Analysis Pipeline

Once the CSV results from either or both phones are placed in `results/`, run the analysis scripts on your PC:

```powershell
cd "f:\Edge AIML Model Performance Analysis on Android Smartphones\EdgeAI-Benchmark\analysis"

# 1. Validate datasets and print raw summary
python analyze_results.py

# 2. Compute Spearman and Kendall rank correlations
python correlation.py

# 3. Compute efficiency rankings and detect rank inversions
python ranking.py

# 4. Generate all 10 publication-quality figures (PNG & PDF)
python generate_plots.py
```

Generated outputs will appear in:
* `reports/tables/correlation_analysis.csv`
* `reports/tables/model_rankings.csv`
* `reports/figures/fig1_flops_vs_latency.png` (& `.pdf`)
* `reports/figures/fig2_parameters_vs_latency.png` (& `.pdf`)
* `reports/figures/fig3_model_size_vs_latency.png` (& `.pdf`)
* `reports/figures/fig4_flops_vs_pss.png` (& `.pdf`)
* `reports/figures/fig5_parameters_vs_pss.png` (& `.pdf`)
* `reports/figures/fig6_model_size_vs_pss.png` (& `.pdf`)
* `reports/figures/fig7_accuracy_vs_latency.png` (& `.pdf`)
* `reports/figures/fig8_accuracy_vs_model_size.png` (& `.pdf`)
* `reports/figures/fig9_precision_latency_comparison.png` (& `.pdf`)
* `reports/figures/fig10_cross_device_comparison.png` (& `.pdf`)
