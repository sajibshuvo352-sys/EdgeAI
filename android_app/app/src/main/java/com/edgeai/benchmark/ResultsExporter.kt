package com.edgeai.benchmark

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ResultsExporter(private val context: Context) {

    data class ExperimentRecord(
        val device: String,
        val deviceModel: String,
        val chipset: String,
        val androidVersion: String,
        val runtimeVersion: String,
        val modelName: String,
        val precision: String,
        val inputWidth: Int,
        val inputHeight: Int,
        val threads: Int,
        val datasetName: String,
        val imageCount: Int,
        val top1Accuracy: Double,
        val meanLatencyMs: Double,
        val medianLatencyMs: Double,
        val stdLatencyMs: Double,
        val minLatencyMs: Double,
        val maxLatencyMs: Double,
        val p90LatencyMs: Double,
        val baselinePssMb: Double,
        val peakSampledPssMb: Double,
        val memoryIncreaseMb: Double,
        val modelSizeBytes: Long,
        val modelSizeMb: Double,
        val flops: Long,
        val parameterCount: Long,
        val executionDelegate: String,
        val timestamp: String
    )

    companion object {
        val CSV_HEADER = listOf(
            "device", "device_model", "chipset", "android_version", "runtime_version",
            "model_name", "precision", "input_width", "input_height", "threads",
            "dataset_name", "image_count", "top1_accuracy", "mean_latency_ms",
            "median_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms",
            "p90_latency_ms", "baseline_pss_mb", "peak_sampled_pss_mb", "memory_increase_mb",
            "model_size_bytes", "model_size_mb", "flops", "parameter_count",
            "execution_delegate", "timestamp"
        ).joinToString(",")
    }

    fun exportToCsvAndJson(records: List<ExperimentRecord>): Pair<File, File> {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val outputDir = context.getExternalFilesDir(null) ?: context.filesDir

        val csvFile = File(outputDir, "benchmark_results_$timeStamp.csv")
        val jsonFile = File(outputDir, "benchmark_results_$timeStamp.json")

        // 1. Write CSV
        FileWriter(csvFile).use { writer ->
            writer.write(CSV_HEADER + "\n")
            for (r in records) {
                val row = listOf(
                    r.device,
                    r.deviceModel,
                    r.chipset,
                    r.androidVersion,
                    r.runtimeVersion,
                    r.modelName,
                    r.precision,
                    r.inputWidth,
                    r.inputHeight,
                    r.threads,
                    r.datasetName,
                    r.imageCount,
                    r.top1Accuracy,
                    r.meanLatencyMs,
                    r.medianLatencyMs,
                    r.stdLatencyMs,
                    r.minLatencyMs,
                    r.maxLatencyMs,
                    r.p90LatencyMs,
                    r.baselinePssMb,
                    r.peakSampledPssMb,
                    r.memoryIncreaseMb,
                    r.modelSizeBytes,
                    r.modelSizeMb,
                    r.flops,
                    r.parameterCount,
                    r.executionDelegate,
                    r.timestamp
                ).joinToString(",")
                writer.write(row + "\n")
            }
        }

        // 2. Write JSON
        val jsonArray = JSONArray()
        for (r in records) {
            val obj = JSONObject().apply {
                put("device", r.device)
                put("device_model", r.deviceModel)
                put("chipset", r.chipset)
                put("android_version", r.androidVersion)
                put("runtime_version", r.runtimeVersion)
                put("model_name", r.modelName)
                put("precision", r.precision)
                put("input_width", r.inputWidth)
                put("input_height", r.inputHeight)
                put("threads", r.threads)
                put("dataset_name", r.datasetName)
                put("image_count", r.imageCount)
                put("top1_accuracy", r.top1Accuracy)
                put("mean_latency_ms", r.meanLatencyMs)
                put("median_latency_ms", r.medianLatencyMs)
                put("std_latency_ms", r.stdLatencyMs)
                put("min_latency_ms", r.minLatencyMs)
                put("max_latency_ms", r.maxLatencyMs)
                put("p90_latency_ms", r.p90LatencyMs)
                put("baseline_pss_mb", r.baselinePssMb)
                put("peak_sampled_pss_mb", r.peakSampledPssMb)
                put("memory_increase_mb", r.memoryIncreaseMb)
                put("model_size_bytes", r.modelSizeBytes)
                put("model_size_mb", r.modelSizeMb)
                put("flops", r.flops)
                put("parameter_count", r.parameterCount)
                put("execution_delegate", r.executionDelegate)
                put("timestamp", r.timestamp)
            }
            jsonArray.put(obj)
        }

        FileWriter(jsonFile).use { writer ->
            writer.write(jsonArray.toString(4))
        }

        return Pair(csvFile, jsonFile)
    }
}
