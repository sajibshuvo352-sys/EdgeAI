package com.edgeai.benchmark

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow
import kotlin.math.sqrt

class InferenceRunner(
    private val context: Context,
    private val performanceMonitor: PerformanceMonitor,
    private val imageProcessor: ImageProcessor
) {

    data class LatencyStats(
        val meanMs: Double,
        val medianMs: Double,
        val stdMs: Double,
        val minMs: Double,
        val maxMs: Double,
        val p90Ms: Double
    )

    data class AccuracyStats(
        val totalImages: Int,
        val correctCount: Int,
        val accuracyPercentage: Double
    )

    data class BenchmarkExecutionResult(
        val latencyStats: LatencyStats,
        val accuracyStats: AccuracyStats,
        val baselinePssMb: Double,
        val peakPssMb: Double,
        val memoryIncreaseMb: Double
    )

    data class DatasetRecord(
        val imageId: String,
        val imagePath: String,
        val groundTruthLabel: Int,
        val className: String
    )

    fun loadDatasetManifest(): List<DatasetRecord> {
        val list = mutableListOf<DatasetRecord>()
        context.assets.open("dataset/labels.csv").use { stream ->
            BufferedReader(InputStreamReader(stream)).use { reader ->
                var line = reader.readLine() // skip header
                while (reader.readLine().also { line = it } != null) {
                    val parts = line!!.split(",")
                    if (parts.size >= 3) {
                        list.add(
                            DatasetRecord(
                                imageId = parts[0].trim(),
                                imagePath = "dataset/" + parts[1].trim(),
                                groundTruthLabel = parts[2].trim().toInt(),
                                className = if (parts.size > 3) parts[3].trim() else ""
                            )
                        )
                    }
                }
            }
        }
        return list
    }

    fun executeBenchmark(
        interpreter: Interpreter,
        dataset: List<DatasetRecord>,
        warmupIterations: Int = 10,
        measuredIterations: Int = 100,
        onProgressUpdate: ((current: Int, total: Int, stage: String) -> Unit)? = null
    ): BenchmarkExecutionResult {

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        val numClasses = outputTensor.shape()[1]
        val outputDataType = outputTensor.dataType()

        // 1. Initial baseline memory sampling
        val baselinePss = performanceMonitor.sampleProcessPssMb()
        var peakPss = baselinePss

        // Prepare test input from first dataset image
        val sampleRecord = dataset.first()
        val sampleBitmap = imageProcessor.loadBitmapFromAsset(sampleRecord.imagePath)
        val sampleBuffer = imageProcessor.preprocess(sampleBitmap, inputTensor)

        val outputBuffer = allocateOutputBuffer(outputDataType, numClasses)

        // 2. Warm-up iterations (excluded from latency metrics)
        for (w in 1..warmupIterations) {
            sampleBuffer.rewind()
            outputBuffer.rewind()
            interpreter.run(sampleBuffer, outputBuffer)
        }

        // 3. Measured inference iterations
        val latencyList = ArrayList<Double>(measuredIterations)
        for (i in 1..measuredIterations) {
            sampleBuffer.rewind()
            outputBuffer.rewind()

            val startNano = System.nanoTime()
            interpreter.run(sampleBuffer, outputBuffer)
            val durationMs = (System.nanoTime() - startNano) / 1_000_000.0
            latencyList.add(durationMs)

            if (i % 20 == 0) {
                val currentPss = performanceMonitor.sampleProcessPssMb()
                if (currentPss > peakPss) peakPss = currentPss
                onProgressUpdate?.invoke(i, measuredIterations, "Latency Measurement")
            }
        }

        val latencyStats = computeLatencyStats(latencyList)

        // 4. Dataset classification accuracy evaluation
        var correctCount = 0
        for ((idx, record) in dataset.withIndex()) {
            val bmp = imageProcessor.loadBitmapFromAsset(record.imagePath)
            val inputBuf = imageProcessor.preprocess(bmp, inputTensor)
            outputBuffer.rewind()

            interpreter.run(inputBuf, outputBuffer)

            val predictedClass = parseTopClass(outputBuffer, outputDataType, numClasses)
            if (predictedClass == record.groundTruthLabel) {
                correctCount++
            }

            if ((idx + 1) % 15 == 0) {
                val currentPss = performanceMonitor.sampleProcessPssMb()
                if (currentPss > peakPss) peakPss = currentPss
                onProgressUpdate?.invoke(idx + 1, dataset.size, "Accuracy Evaluation")
            }
        }

        val accuracyPct = if (dataset.isNotEmpty()) {
            (correctCount.toDouble() / dataset.size.toDouble()) * 100.0
        } else 0.0

        val finalPss = performanceMonitor.sampleProcessPssMb()
        if (finalPss > peakPss) peakPss = finalPss

        return BenchmarkExecutionResult(
            latencyStats = latencyStats,
            accuracyStats = AccuracyStats(dataset.size, correctCount, accuracyPct),
            baselinePssMb = roundDouble(baselinePss),
            peakPssMb = roundDouble(peakPss),
            memoryIncreaseMb = roundDouble((peakPss - baselinePss).coerceAtLeast(0.0))
        )
    }

    private fun allocateOutputBuffer(dataType: DataType, numClasses: Int): ByteBuffer {
        val bytesPerElem = when (dataType) {
            DataType.FLOAT32 -> 4
            DataType.INT8, DataType.UINT8 -> 1
            else -> 4
        }
        val buf = ByteBuffer.allocateDirect(1 * numClasses * bytesPerElem)
        buf.order(ByteOrder.nativeOrder())
        return buf
    }

    private fun parseTopClass(outputBuffer: ByteBuffer, dataType: DataType, numClasses: Int): Int {
        outputBuffer.rewind()
        var maxIdx = -1
        var maxVal = Float.NEGATIVE_INFINITY

        when (dataType) {
            DataType.FLOAT32 -> {
                for (c in 0 until numClasses) {
                    val v = outputBuffer.float
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
            }
            DataType.INT8 -> {
                for (c in 0 until numClasses) {
                    val v = outputBuffer.get().toInt().toFloat()
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
            }
            DataType.UINT8 -> {
                for (c in 0 until numClasses) {
                    val v = (outputBuffer.get().toInt() and 0xFF).toFloat()
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
            }
            else -> {
                for (c in 0 until numClasses) {
                    val v = outputBuffer.float
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
            }
        }
        return maxIdx
    }

    private fun computeLatencyStats(times: List<Double>): LatencyStats {
        val sorted = times.sorted()
        val n = sorted.size
        val mean = sorted.average()

        val variance = sorted.fold(0.0) { acc, d -> acc + (d - mean).pow(2.0) } / n
        val std = sqrt(variance)

        val median = if (n % 2 == 0) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }

        val p90Index = ((0.90 * n).toInt()).coerceIn(0, n - 1)
        val p90 = sorted[p90Index]

        return LatencyStats(
            meanMs = roundDouble(mean),
            medianMs = roundDouble(median),
            stdMs = roundDouble(std),
            minMs = roundDouble(sorted.first()),
            maxMs = roundDouble(sorted.last()),
            p90Ms = roundDouble(p90)
        )
    }

    private fun roundDouble(value: Double): Double {
        return Math.round(value * 100.0) / 100.0
    }
}
