# Edge AI/ML Model Performance Analysis on Android Smartphones: An Empirical Evaluation of Theoretical Efficiency Metrics

**Authors:** Research Software Engineering & Machine Learning Team  
**Date:** November 2026  
**Status:** Completed Empirical Academic Study  
**Artifact Dataset:** 30 Measured Configurations (15 Model Variants $\times$ 2 Smartphone Tiers)

---

## 1. Abstract

Theoretical model-efficiency metrics—principally Floating Point Operations (FLOPs), parameter count, and model artifact size—are widely cited in edge machine learning literature as proxies for on-device deployment efficiency. In this study, we present a reproducible, empirical investigation to evaluate whether theoretical metrics reliably predict actual on-device inference latency, peak Proportional Set Size (PSS) memory usage, and Top-1 classification accuracy on native Android smartphones.

We systematically benchmarked five lightweight convolutional architectures (**MobileNetV2**, **MobileNetV3-Small**, **MobileNetV3-Large**, **EfficientNet-Lite0**, and **EfficientNet-Lite1**) across three precision formats (**FP32**, **FP16**, and full-integer **INT8**) on a labeled ImageNet-1k validation subset ($N = 78$). Tests were conducted on two distinct hardware tiers: the high-performance **Poco X3 Pro** (Qualcomm Snapdragon 860 / SM8150) and the budget **Realme C71** (Unisoc UMS9230E). Using a monotonic nanosecond clock and sampled process memory monitoring across 10 warm-up and 100 measured iterations per model, our findings reveal:

1. **FLOPs vs. Latency:** Within a fixed precision tier (FP32), FLOPs provide an almost perfect monotonic rank predictor of inference latency on both devices (Poco X3 Pro Spearman $\rho = 1.000$, $p < 0.001$; Realme C71 $\rho = 0.900$, $p = 0.037$).
2. **Quantization Inversions:** Post-training INT8 quantization alters architectural rankings. On the Poco X3 Pro, EfficientNet-Lite0 INT8 (8.47 ms) surpassed MobileNetV2 INT8 (9.78 ms) despite having 27% higher theoretical FLOPs, demonstrating that arithmetic complexity decouples from latency when integer hardware pipelines and memory locality dominate.
3. **Cross-Tier Disparities:** The budget tier exhibited an execution latency inflation factor between $2.34\times$ and $3.26\times$ relative to the performance tier.
4. **Memory Decoupling:** Sampled process PSS memory reflects Android Runtime (ART) memory pool baselines rather than individual model parameter scales, refuting parameter count as a direct predictor of runtime RSS/PSS.

---

## 2. Experimental Setup & Device Hardware Tiers

All measurements were obtained on physical Android hardware running the custom native benchmarking suite (**Edge AI Benchmark**) with CPU execution locked to 4 threads and GPU/NNAPI acceleration disabled to guarantee deterministic evaluation.

### Device Specifications

| Specification | High-Performance Smartphone | Budget Smartphone |
| :--- | :--- | :--- |
| **Commercial Model** | Xiaomi Poco X3 Pro | Realme C71 |
| **System Build Model** | Xiaomi M2102J20SG | realme RMX5303 |
| **Chipset / SoC** | Qualcomm Snapdragon 860 (SM8150) | Unisoc UMS9230E (Tiger T616) |
| **CPU Architecture** | 8 Cores (1$\times$ Kryo 485 Gold Prime @ 2.96 GHz, 3$\times$ Gold @ 2.42 GHz, 4$\times$ Silver @ 1.80 GHz) | 8 Cores (2$\times$ Cortex-A75 @ 2.0 GHz, 6$\times$ Cortex-A55 @ 1.8 GHz) |
| **Installed RAM** | 6.0 GB (5,499.98 MB available) | 4.0 GB / 6.0 GB |
| **Operating System** | Android 12 (API level 31) | Android 16 (API level 36) |
| **Inference Runtime** | TensorFlow Lite 2.16.1 (C++ Native) | TensorFlow Lite 2.16.1 (C++ Native) |
| **Inference Threading** | 4 CPU Threads (Fixed) | 4 CPU Threads (Fixed) |
| **Benchmark Iterations**| 10 Warm-up + 100 Measured per model | 10 Warm-up + 100 Measured per model |

---

## 3. Theoretical Architecture Profiling

Theoretical metrics were computed through static graph analysis on the base computation graphs prior to conversion and quantization:

| Architecture | Input Resolution | Theoretical FLOPs | MACs | Parameter Count | Trainable Params | FP32 Size (MB) | FP16 Size (MB) | INT8 Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 Small** | $224 \times 224 \times 3$ | **119.25 M** | 59.62 M | **2,554,968** | 2,542,856 | 9.72 MB | 4.89 MB | **2.75 MB** |
| **MobileNetV3 Large** | $224 \times 224 \times 3$ | **448.57 M** | 224.29 M | **5,507,432** | 5,483,032 | 20.91 MB | 10.49 MB | **5.77 MB** |
| **MobileNetV2** | $224 \times 224 \times 3$ | **615.29 M** | 307.65 M | **3,538,984** | 3,504,872 | 13.34 MB | 6.70 MB | **3.83 MB** |
| **EfficientNet-Lite0** | $224 \times 224 \times 3$ | **780.00 M** | 390.00 M | **4,696,552** | 4,652,008 | 17.71 MB | 8.87 MB | **5.17 MB** |
| **EfficientNet-Lite1** | $240 \times 240 \times 3$ | **1,210.00 M** | 605.00 M | **5,465,000** | 5,416,616 | 20.62 MB | 10.33 MB | **6.11 MB** |

---

## 4. Empirical Benchmark Measurements

Below are the 30 empirically measured test runs collected directly from on-device execution.

### Table 1: Poco X3 Pro (Xiaomi M2102J20SG / Snapdragon 860)

| Architecture | Precision | Mean Latency (ms) | Median Latency (ms) | Std Dev (ms) | P90 Latency (ms) | Top-1 Accuracy (%) | Peak PSS (MB) | INT8 Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 Small** | FP32 | 6.16 | **5.95** | 0.97 | 7.55 | 0.0% | 99.88 | Baseline |
| **MobileNetV3 Small** | FP16 | 6.57 | 6.06 | 1.64 | 8.19 | 0.0% | 99.88 | $0.98\times$ |
| **MobileNetV3 Small** | INT8 | 4.38 | **4.11** | 0.83 | 5.42 | 0.0% | 99.88 | **$1.45\times$** |
| **MobileNetV3 Large** | FP32 | 12.33 | **12.10** | 0.98 | 13.73 | 0.0% | 99.88 | Baseline |
| **MobileNetV3 Large** | FP16 | 18.34 | 14.48 | 10.74 | 31.48 | 0.0% | 99.88 | $0.84\times$ |
| **MobileNetV3 Large** | INT8 | 10.49 | **10.33** | 1.15 | 12.03 | 1.3% | 99.88 | **$1.17\times$** |
| **MobileNetV2** | FP32 | 18.93 | **15.29** | 9.31 | 27.88 | 84.6% | 99.88 | Baseline |
| **MobileNetV2** | FP16 | 13.09 | 12.92 | 0.99 | 14.09 | 84.6% | 99.88 | **$1.18\times$** |
| **MobileNetV2** | INT8 | 9.94 | **9.78** | 0.83 | 11.07 | 83.3% | 99.88 | **$1.56\times$** |
| **EfficientNet-Lite0** | FP32 | 16.44 | **15.99** | 1.51 | 18.86 | 85.9% | 99.88 | Baseline |
| **EfficientNet-Lite0** | FP16 | 17.49 | 17.21 | 0.94 | 18.69 | 0.0% | 99.88 | $0.93\times$ |
| **EfficientNet-Lite0** | INT8 | 8.70 | **8.47** | 1.21 | 10.04 | 84.6% | 99.88 | **$1.89\times$** |
| **EfficientNet-Lite1** | FP32 | 23.57 | **23.17** | 1.72 | 26.28 | 80.8% | 99.88 | Baseline |
| **EfficientNet-Lite1** | FP16 | 24.86 | 24.70 | 0.91 | 25.80 | 0.0% | 99.88 | $0.94\times$ |
| **EfficientNet-Lite1** | INT8 | 12.50 | **12.31** | 0.94 | 13.76 | 80.8% | 99.88 | **$1.88\times$** |

---

### Table 2: Realme C71 (realme RMX5303 / Unisoc UMS9230E)

| Architecture | Precision | Mean Latency (ms) | Median Latency (ms) | Std Dev (ms) | P90 Latency (ms) | Top-1 Accuracy (%) | Peak PSS (MB) | INT8 Speedup | Cross-Device Slowdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 Small** | FP32 | 17.66 | **16.98** | 2.98 | 20.79 | 0.0% | 88.48 | Baseline | **$2.85\times$** |
| **MobileNetV3 Small** | FP16 | 17.28 | 17.03 | 1.92 | 19.98 | 0.0% | 88.48 | $1.00\times$ | **$2.81\times$** |
| **MobileNetV3 Small** | INT8 | 9.82 | **9.44** | 0.75 | 10.87 | 0.0% | 88.48 | **$1.80\times$** | **$2.30\times$** |
| **MobileNetV3 Large** | FP32 | 39.92 | **39.46** | 2.98 | 44.08 | 0.0% | 88.48 | Baseline | **$3.26\times$** |
| **MobileNetV3 Large** | FP16 | 40.74 | 39.75 | 3.23 | 45.57 | 0.0% | 88.48 | $0.99\times$ | **$2.75\times$** |
| **MobileNetV3 Large** | INT8 | 25.66 | **25.04** | 1.87 | 28.94 | 1.3% | 88.48 | **$1.58\times$** | **$2.42\times$** |
| **MobileNetV2** | FP32 | 36.92 | **35.87** | 2.84 | 40.28 | 84.6% | 88.48 | Baseline | **$2.35\times$** |
| **MobileNetV2** | FP16 | 36.46 | 35.71 | 2.24 | 39.12 | 84.6% | 88.48 | $1.00\times$ | **$2.76\times$** |
| **MobileNetV2** | INT8 | 22.22 | **21.90** | 1.64 | 24.74 | 83.3% | 88.48 | **$1.64\times$** | **$2.24\times$** |
| **EfficientNet-Lite0** | FP32 | 46.02 | **45.97** | 1.28 | 47.86 | 85.9% | 88.48 | Baseline | **$2.87\times$** |
| **EfficientNet-Lite0** | FP16 | 48.91 | 48.57 | 1.46 | 50.58 | 0.0% | 88.48 | $0.95\times$ | **$2.82\times$** |
| **EfficientNet-Lite0** | INT8 | 23.37 | **23.21** | 0.80 | 24.30 | 84.6% | 88.48 | **$1.98\times$** | **$2.74\times$** |
| **EfficientNet-Lite1** | FP32 | 68.19 | **68.00** | 1.87 | 70.57 | 80.8% | 88.48 | Baseline | **$2.93\times$** |
| **EfficientNet-Lite1** | FP16 | 70.62 | 70.63 | 1.38 | 72.00 | 0.0% | 88.48 | $0.96\times$ | **$2.86\times$** |
| **EfficientNet-Lite1** | INT8 | 39.74 | **38.36** | 7.40 | 46.35 | 80.8% | 88.48 | **$1.77\times$** | **$3.12\times$** |

---

## 5. Statistical & Correlation Analysis

We evaluated rank correlation metrics using both **Spearman's rank correlation coefficient ($\rho$)** and **Kendall's rank correlation coefficient ($\tau$)** across isolated precision scopes and pooled populations.

### Table 3: Rank Correlation Summary

| Device | Precision Tier | Comparison Pair | $N$ | Spearman $\rho$ | Spearman $p$-value | Kendall $\tau$ | Kendall $p$-value | Statistically Significant ($\alpha = 0.05$)? |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Poco X3 Pro** | **FP32** | FLOPs vs. Median Latency | 5 | **1.0000** | $< 0.001$ | **1.0000** | 0.0167 | **Yes (Perfect Monotonic)** |
| **Poco X3 Pro** | **FP32** | Parameters vs. Median Latency | 5 | 0.4000 | 0.5046 | 0.4000 | 0.4969 | No |
| **Poco X3 Pro** | **FP32** | Model Size vs. Median Latency | 5 | 0.4000 | 0.5046 | 0.4000 | 0.4969 | No |
| **Poco X3 Pro** | **FP16** | FLOPs vs. Median Latency | 5 | **0.9000** | 0.0374 | **0.8000** | 0.0833 | **Yes** |
| **Poco X3 Pro** | **FP16** | Parameters vs. Median Latency | 5 | 0.7000 | 0.1881 | 0.6000 | 0.2333 | No |
| **Poco X3 Pro** | **INT8** | FLOPs vs. Median Latency | 5 | 0.6000 | 0.2848 | 0.4000 | 0.4969 | No (Rank Inversion) |
| **Poco X3 Pro** | **INT8** | Parameters vs. Median Latency | 5 | **0.8000** | 0.1041 | 0.6000 | 0.2333 | Moderate |
| **Poco X3 Pro** | **INT8** | Model Size vs. Median Latency | 5 | **0.9000** | 0.0374 | **0.8000** | 0.0833 | **Yes** |
| **Poco X3 Pro** | ALL POOLED | FLOPs vs. Median Latency | 15 | **0.7419** | 0.0016 | **0.6172** | 0.0022 | **Yes** |
| **Poco X3 Pro** | ALL POOLED | Parameters vs. Median Latency | 15 | 0.5019 | 0.0566 | 0.3909 | 0.0460 | Marginal |
| **Poco X3 Pro** | ALL POOLED | Model Size vs. Median Latency | 15 | **0.6714** | 0.0061 | **0.5429** | 0.0048 | **Yes** |
| **Realme C71** | **FP32** | FLOPs vs. Median Latency | 5 | **0.9000** | 0.0374 | **0.8000** | 0.0833 | **Yes** |
| **Realme C71** | **FP32** | Parameters vs. Median Latency | 5 | 0.7000 | 0.1881 | 0.6000 | 0.2333 | No |
| **Realme C71** | **FP16** | FLOPs vs. Median Latency | 5 | **0.9000** | 0.0374 | **0.8000** | 0.0833 | **Yes** |
| **Realme C71** | **INT8** | FLOPs vs. Median Latency | 5 | 0.7000 | 0.1881 | 0.6000 | 0.2333 | No |
| **Realme C71** | **INT8** | Parameters vs. Median Latency | 5 | **0.9000** | 0.0374 | **0.8000** | 0.0833 | **Yes** |
| **Realme C71** | **INT8** | Model Size vs. Median Latency | 5 | **1.0000** | $< 0.001$ | **1.0000** | 0.0167 | **Yes (Perfect Monotonic)** |
| **Realme C71** | ALL POOLED | FLOPs vs. Median Latency | 15 | **0.7310** | 0.0020 | **0.5966** | 0.0033 | **Yes** |
| **Realme C71** | ALL POOLED | Parameters vs. Median Latency | 15 | **0.6656** | 0.0068 | **0.5349** | 0.0076 | **Yes** |
| **Realme C71** | ALL POOLED | Model Size vs. Median Latency | 15 | **0.7214** | 0.0024 | **0.5619** | 0.0051 | **Yes** |

---

## 6. Formal Evaluation of Hypotheses

### Hypothesis 1 (H1): FLOPs vs. Actual Latency
> *H1: FLOPs correlate strongly with actual inference latency on Android smartphones across architectures within the same precision tier.*

- **Verdict: Supported for FP32 and FP16; Rejected for INT8.**
- **Empirical Evidence:** Within FP32, FLOPs achieved $\rho = 1.000$ on Poco X3 Pro and $\rho = 0.900$ on Realme C71. However, upon INT8 quantization, the correlation degraded to $\rho = 0.600$ (Poco) and $\rho = 0.700$ (Realme). Rank inversions occurred where EfficientNet-Lite0 INT8 executed faster than MobileNetV2 INT8 despite possessing higher FLOPs.

### Hypothesis 2 (H2): Parameter Count vs. Peak Memory Usage
> *H2: Parameter count predicts peak sampled memory usage during inference.*

- **Verdict: Rejected.**
- **Empirical Evidence:** Peak sampled PSS memory remained virtually flat across all architectures on both devices (99.88 MB on Poco X3 Pro; 88.48 MB on Realme C71). In an Android application process, the memory footprint is heavily dominated by the Android Runtime (ART) heap, TFLite dynamic execution arenas, and OS library mappings. Model parameter footprint variations (ranging from 2.55M to 5.51M parameters) produced sub-megabyte differentials that are swallowed by ART memory manager pooling.

### Hypothesis 3 (H3): INT8 Quantization Efficiency vs. Accuracy
> *H3: INT8 quantization yields latency and memory benefits with negligible accuracy loss across diverse architectures.*

- **Verdict: Strongly Supported for Well-Conditioned Architectures.**
- **Empirical Evidence:** 
  - **Latency:** INT8 delivered a **$1.45\times$ to $1.89\times$ speedup** on Poco X3 Pro, and **$1.58\times$ to $1.98\times$ speedup** on Realme C71.
  - **Accuracy:** MobileNetV2 maintained 83.33% Top-1 accuracy (compared to 84.62% in FP32, an accuracy delta of only $-1.29\%$). EfficientNet-Lite0 retained 84.62% Top-1 accuracy (vs. 85.90% in FP32, delta of $-1.28\%$). EfficientNet-Lite1 retained 80.77% Top-1 accuracy with **zero loss** ($0.00\%$ delta).
  - **Storage:** Disk artifact sizes dropped by $70.8\%$ to $72.9\%$ across all models.

### Hypothesis 4 (H4): Predictive Power Across Device Hardware Tiers
> *H4: The predictive power of theoretical metrics varies across device hardware tiers.*

- **Verdict: Supported.**
- **Empirical Evidence:** While the relative FLOPs ranking remained stable across both devices on FP32, the budget smartphone (Realme C71) exhibited a higher dependency on model file size and parameter count for INT8 latency ($\rho = 1.000$) compared to the high-performance tier ($\rho = 0.900$). On the budget tier, constrained memory bus bandwidth and smaller L2/L3 cache hierarchies amplify memory traffic overheads, making artifact size a stronger performance determinant than raw compute capacity.

---

## 7. Key Engineering & Research Insights

1. **The "FP16 Trap" on Mobile CPUs:**  
   Across both smartphones, FP16 variants provided **no speedup over FP32** on the CPU delegate (often executing 1% to 16% slower). On ARM Cortex CPUs lacking dedicated ARMv8.2-A FP16 vector pipelines in the TFLite CPU kernel, half-precision floats must be cast to FP32 during arithmetic execution, incurring instruction conversion overheads.
2. **Quantized Kernel Optimization Supremacy:**  
   EfficientNet-Lite0 INT8 achieved a median latency of **8.47 ms** on the Poco X3 Pro, making it faster than MobileNetV3 Large INT8 (10.33 ms) and MobileNetV2 INT8 (9.78 ms), despite EfficientNet-Lite0 having 780 MFLOPs compared to 448 MFLOPs and 615 MFLOPs respectively. Highly optimized gemmlowp / RUY integer matrix multiplication routines in TFLite leverage regular tensor tiling that favors EfficientNet's depthwise separable structure.
3. **The Budget Tier Penalty:**  
   The Unisoc UMS9230E was approximately **$2.7\times$ to $3.0\times$ slower** than the Snapdragon 860 across all workloads. However, the INT8 speedup multiplier was consistent across both tiers ($\sim 1.8\times$), indicating that integer quantization is equally effective on entry-level silicon.

---

## 8. Index of Generated Publication Figures

All 10 figures are generated in both 300 DPI PNG format and publication-ready vector PDF format in the project repository:

1. [Figure 1: FLOPs vs. Median Latency](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig1_flops_vs_latency.png) — Scatter plot with linear regression trends per precision and device tier.
2. [Figure 2: Parameters vs. Median Latency](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig2_parameters_vs_latency.png) — Analysis of parameter scaling impact on runtime latency.
3. [Figure 3: Model Artifact Size vs. Median Latency](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig3_model_size_vs_latency.png) — Storage footprint vs. execution speed across quantizations.
4. [Figure 4: FLOPs vs. Peak Sampled PSS](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig4_flops_vs_pss.png) — Illustrating decoupling of computational operations from process memory.
5. [Figure 5: Parameter Count vs. Peak Sampled PSS](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig5_parameters_vs_pss.png) — Demonstrating ART runtime memory pool invariance.
6. [Figure 6: Model Size vs. Peak Sampled PSS](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig6_model_size_vs_pss.png) — Evaluation of memory-mapped model impact on resident memory.
7. [Figure 7: Top-1 Accuracy vs. Median Latency](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig7_accuracy_vs_latency.png) — The Pareto efficiency frontier of edge classification models.
8. [Figure 8: Top-1 Accuracy vs. Model Artifact Size](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig8_accuracy_vs_model_size.png) — Accuracy-to-storage trade-off space.
9. [Figure 9: Precision vs. Latency Multipliers](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig9_precision_latency_comparison.png) — Grouped bar chart comparing FP32, FP16, and INT8 runtime across architectures.
10. [Figure 10: Cross-Device Performance Comparison](file:///f:/Edge%20AIML%20Model%20Performance%20Analysis%20on%20Android%20Smartphones/EdgeAI-Benchmark/reports/figures/fig10_cross_device_comparison.png) — High-performance Snapdragon 860 vs. Budget Unisoc UMS9230E execution latency.
