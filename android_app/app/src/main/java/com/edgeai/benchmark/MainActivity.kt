package com.edgeai.benchmark

import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.edgeai.benchmark.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.BufferedReader
import java.io.InputStreamReader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var performanceMonitor: PerformanceMonitor
    private lateinit var modelManager: ModelManager
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var inferenceRunner: InferenceRunner
    private lateinit var resultsExporter: ResultsExporter

    private val collectedResults = mutableListOf<ResultsExporter.ExperimentRecord>()
    private var modelMetadataMap = mutableMapOf<String, MetadataInfo>()

    data class MetadataInfo(
        val flops: Long,
        val parameterCount: Long,
        val sizeBytes: Long,
        val sizeMb: Double,
        val inputWidth: Int,
        val inputHeight: Int
    )

    private val architectures = listOf(
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_lite0",
        "efficientnet_lite1"
    )

    private val precisions = listOf("FP32", "FP16", "INT8")
    private val threadOptions = listOf("4", "1", "2", "8")
    private val runOptions = listOf("100", "50", "200")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        performanceMonitor = PerformanceMonitor(this)
        modelManager = ModelManager(this)
        imageProcessor = ImageProcessor(this)
        inferenceRunner = InferenceRunner(this, performanceMonitor, imageProcessor)
        resultsExporter = ResultsExporter(this)

        setupSpinners()
        displayDeviceTelemetry()
        loadModelMetadata()

        binding.btnRunBenchmark.setOnClickListener {
            runSingleBenchmark()
        }

        binding.btnRunAll.setOnClickListener {
            runFullMatrixBenchmark()
        }

        binding.btnExportResults.setOnClickListener {
            exportResults()
        }
    }

    private fun setupSpinners() {
        binding.spinnerModel.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            architectures
        )

        binding.spinnerPrecision.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            precisions
        )

        binding.spinnerThreads.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            threadOptions
        )

        binding.spinnerRuns.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            runOptions
        )
    }

    private fun displayDeviceTelemetry() {
        val telem = performanceMonitor.getDeviceTelemetry()
        binding.tvDeviceModel.text = "Device: ${telem.manufacturer} ${telem.model} (Build: ${telem.buildId})"
        binding.tvChipset.text = "Chipset / SoC: ${telem.chipset} | CPU Cores: ${telem.cpuCores}"
        binding.tvRamAndroid.text = "RAM: ${telem.totalRamMb} MB total | Android: ${telem.androidVersion} (API ${telem.sdkInt})"
        appendLog("Detected hardware: ${telem.manufacturer} ${telem.model} (${telem.chipset})")
    }

    private fun loadModelMetadata() {
        try {
            assets.open("model_metadata.json").use { stream ->
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    val jsonStr = reader.readText()
                    val array = JSONArray(jsonStr)
                    for (i in 0 until array.length()) {
                        val item = array.getJSONObject(i)
                        val id = item.getString("model_identifier")
                        modelMetadataMap[id] = MetadataInfo(
                            flops = item.getLong("flops"),
                            parameterCount = item.getLong("parameterCount"),
                            sizeBytes = item.getLong("model_size_bytes"),
                            sizeMb = item.getDouble("model_size_mb"),
                            inputWidth = item.getInt("input_width"),
                            inputHeight = item.getInt("input_height")
                        )
                    }
                }
            }
            appendLog("Loaded metadata for ${modelMetadataMap.size} models.")
        } catch (e: Exception) {
            appendLog("Warning: Could not parse model_metadata.json: ${e.message}")
        }
    }

    private fun runSingleBenchmark() {
        val selectedModel = binding.spinnerModel.selectedItem.toString()
        val selectedPrecision = binding.spinnerPrecision.selectedItem.toString()
        val threads = binding.spinnerThreads.selectedItem.toString().toInt()
        val runs = binding.spinnerRuns.selectedItem.toString().toInt()

        setUiBenchmarking(true)

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                benchmarkModel(selectedModel, selectedPrecision, threads, runs)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    appendLog("Error executing $selectedModel $selectedPrecision: ${e.message}")
                    binding.tvStatus.text = "Status: Execution Failed"
                }
            } finally {
                withContext(Dispatchers.Main) {
                    setUiBenchmarking(false)
                }
            }
        }
    }

    private fun runFullMatrixBenchmark() {
        val threads = binding.spinnerThreads.selectedItem.toString().toInt()
        val runs = binding.spinnerRuns.selectedItem.toString().toInt()

        setUiBenchmarking(true)

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val totalConfigs = architectures.size * precisions.size
                var completed = 0

                for (arch in architectures) {
                    for (prec in precisions) {
                        completed++
                        withContext(Dispatchers.Main) {
                            binding.tvStatus.text = "Matrix [$completed/$totalConfigs]: $arch ($prec)"
                            appendLog(">> Starting Matrix Run: $arch [$prec]")
                        }
                        benchmarkModel(arch, prec, threads, runs)
                    }
                }

                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Status: Completed Full Matrix ($totalConfigs runs)"
                    appendLog("Successfully completed full 15-model benchmark matrix.")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    appendLog("Matrix run aborted: ${e.message}")
                }
            } finally {
                withContext(Dispatchers.Main) {
                    setUiBenchmarking(false)
                }
            }
        }
    }

    private suspend fun benchmarkModel(
        modelName: String,
        precision: String,
        threads: Int,
        runs: Int
    ) {
        val telem = performanceMonitor.getDeviceTelemetry()
        val dataset = inferenceRunner.loadDatasetManifest()

        withContext(Dispatchers.Main) {
            appendLog("Loading model: $modelName ($precision) | Threads: $threads | Runs: $runs")
        }

        val interpreter = modelManager.loadModel(modelName, precision, threads)

        val result = inferenceRunner.executeBenchmark(
            interpreter = interpreter,
            dataset = dataset,
            warmupIterations = 10,
            measuredIterations = runs,
            onProgressUpdate = { current, total, stage ->
                lifecycleScope.launch(Dispatchers.Main) {
                    binding.progressBar.visibility = View.VISIBLE
                    binding.progressBar.max = total
                    binding.progressBar.progress = current
                    binding.tvStatus.text = "[$stage] $current / $total"
                }
            }
        )

        val metaId = "${modelName}_${precision.lowercase()}"
        val meta = modelMetadataMap[metaId]

        val record = ResultsExporter.ExperimentRecord(
            device = telem.manufacturer,
            deviceModel = telem.model,
            chipset = telem.chipset,
            androidVersion = telem.androidVersion,
            runtimeVersion = "TFLite 2.16.1",
            modelName = modelName,
            precision = precision,
            inputWidth = meta?.inputWidth ?: 224,
            inputHeight = meta?.inputHeight ?: 224,
            threads = threads,
            datasetName = "ImageNet-1k-Validation-Subset",
            imageCount = result.accuracyStats.totalImages,
            top1Accuracy = result.accuracyStats.accuracyPercentage,
            meanLatencyMs = result.latencyStats.meanMs,
            medianLatencyMs = result.latencyStats.medianMs,
            stdLatencyMs = result.latencyStats.stdMs,
            minLatencyMs = result.latencyStats.minMs,
            maxLatencyMs = result.latencyStats.maxMs,
            p90LatencyMs = result.latencyStats.p90Ms,
            baselinePssMb = result.baselinePssMb,
            peakSampledPssMb = result.peakPssMb,
            memoryIncreaseMb = result.memoryIncreaseMb,
            modelSizeBytes = meta?.sizeBytes ?: 0L,
            modelSizeMb = meta?.sizeMb ?: 0.0,
            flops = meta?.flops ?: 0L,
            parameterCount = meta?.parameterCount ?: 0L,
            executionDelegate = "CPU",
            timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date())
        )

        collectedResults.add(record)

        withContext(Dispatchers.Main) {
            binding.tvLatencySummary.text = "Latency: Median: ${result.latencyStats.medianMs} ms | Mean: ${result.latencyStats.meanMs} ms | P90: ${result.latencyStats.p90Ms} ms"
            binding.tvMemorySummary.text = "Sampled PSS: Baseline: ${result.baselinePssMb} MB | Peak: ${result.peakPssMb} MB | Delta: ${result.memoryIncreaseMb} MB"
            binding.tvAccuracySummary.text = "Accuracy: Top-1: ${String.format("%.1f", result.accuracyStats.accuracyPercentage)}% (${result.accuracyStats.correctCount} / ${result.accuracyStats.totalImages})"
            binding.tvTheoreticalSummary.text = "Theoretical: FLOPs: ${meta?.flops ?: 0} | Params: ${meta?.parameterCount ?: 0} | Size: ${meta?.sizeMb ?: 0.0} MB"

            appendLog("DONE: $modelName ($precision) -> Median Latency: ${result.latencyStats.medianMs} ms, Top-1: ${String.format("%.1f", result.accuracyStats.accuracyPercentage)}%, Peak PSS: ${result.peakPssMb} MB")
            binding.tvStatus.text = "Status: Completed $modelName ($precision)"
        }
    }

    private fun exportResults() {
        if (collectedResults.isEmpty()) {
            Toast.makeText(this, "No benchmark results to export yet!", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val (csvFile, jsonFile) = resultsExporter.exportToCsvAndJson(collectedResults)
            appendLog("Exported CSV: ${csvFile.absolutePath}")
            appendLog("Exported JSON: ${jsonFile.absolutePath}")
            Toast.makeText(this, "Exported ${collectedResults.size} records to ${csvFile.name}", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            appendLog("Export error: ${e.message}")
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setUiBenchmarking(isBusy: Boolean) {
        binding.btnRunBenchmark.isEnabled = !isBusy
        binding.btnRunAll.isEnabled = !isBusy
        binding.btnExportResults.isEnabled = !isBusy
        binding.spinnerModel.isEnabled = !isBusy
        binding.spinnerPrecision.isEnabled = !isBusy
        binding.spinnerThreads.isEnabled = !isBusy
        binding.spinnerRuns.isEnabled = !isBusy
        binding.progressBar.visibility = if (isBusy) View.VISIBLE else View.GONE
    }

    private fun appendLog(msg: String) {
        val time = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        val current = binding.tvExecutionLog.text.toString()
        binding.tvExecutionLog.text = "$current\n[$time] $msg"
    }

    override fun onDestroy() {
        super.onDestroy()
        modelManager.close()
    }
}
