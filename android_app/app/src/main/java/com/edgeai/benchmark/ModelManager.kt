package com.edgeai.benchmark

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Manages loading and life-cycle of TFLite models.
 */
class ModelManager(private val context: Context) {

    var currentInterpreter: Interpreter? = null
        private set

    var currentModelName: String = ""
        private set

    var currentPrecision: String = ""
        private set

    fun loadModel(
        modelName: String,
        precision: String,
        numThreads: Int = 4
    ): Interpreter {
        close()

        val fileName = "models/${modelName}_${precision.lowercase()}.tflite"
        val buffer = loadModelFile(fileName)

        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            useXNNPACK = false
        }

        val interpreter = Interpreter(buffer, options)
        currentInterpreter = interpreter
        currentModelName = modelName
        currentPrecision = precision
        return interpreter
    }

    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    fun close() {
        currentInterpreter?.close()
        currentInterpreter = null
    }
}
