package com.edgeai.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageProcessor(private val context: Context) {

    fun loadBitmapFromAsset(assetRelativePath: String): Bitmap {
        context.assets.open(assetRelativePath).use { stream ->
            return BitmapFactory.decodeStream(stream)
                ?: throw IllegalArgumentException("Cannot decode image from: $assetRelativePath")
        }
    }

    fun preprocess(bitmap: Bitmap, inputTensor: Tensor): ByteBuffer {
        val shape = inputTensor.shape()
        val targetHeight = shape[1]
        val targetWidth = shape[2]
        val dataType = inputTensor.dataType()
        val qParams = inputTensor.quantizationParams()

        val resized = if (bitmap.width != targetWidth || bitmap.height != targetHeight) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }

        val pixels = IntArray(targetWidth * targetHeight)
        resized.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val byteBuffer: ByteBuffer
        when (dataType) {
            DataType.FLOAT32 -> {
                byteBuffer = ByteBuffer.allocateDirect(1 * targetHeight * targetWidth * 3 * 4)
                byteBuffer.order(ByteOrder.nativeOrder())
                for (pixel in pixels) {
                    val r = (pixel shr 16 and 0xFF).toFloat()
                    val g = (pixel shr 8 and 0xFF).toFloat()
                    val b = (pixel and 0xFF).toFloat()

                    byteBuffer.putFloat((r / 127.5f) - 1.0f)
                    byteBuffer.putFloat((g / 127.5f) - 1.0f)
                    byteBuffer.putFloat((b / 127.5f) - 1.0f)
                }
            }
            DataType.INT8 -> {
                byteBuffer = ByteBuffer.allocateDirect(1 * targetHeight * targetWidth * 3 * 1)
                byteBuffer.order(ByteOrder.nativeOrder())
                val scale = if (qParams.scale > 0f) qParams.scale else 0.007843138f
                val zp = qParams.zeroPoint

                for (pixel in pixels) {
                    val rNorm = ((pixel shr 16 and 0xFF).toFloat() / 127.5f) - 1.0f
                    val gNorm = ((pixel shr 8 and 0xFF).toFloat() / 127.5f) - 1.0f
                    val bNorm = ((pixel and 0xFF).toFloat() / 127.5f) - 1.0f

                    byteBuffer.put(quantizeToInt8(rNorm, scale, zp))
                    byteBuffer.put(quantizeToInt8(gNorm, scale, zp))
                    byteBuffer.put(quantizeToInt8(bNorm, scale, zp))
                }
            }
            DataType.UINT8 -> {
                byteBuffer = ByteBuffer.allocateDirect(1 * targetHeight * targetWidth * 3 * 1)
                byteBuffer.order(ByteOrder.nativeOrder())
                val scale = if (qParams.scale > 0f) qParams.scale else 0.012566017f
                val zp = qParams.zeroPoint

                for (pixel in pixels) {
                    val rNorm = ((pixel shr 16 and 0xFF).toFloat() / 127.5f) - 1.0f
                    val gNorm = ((pixel shr 8 and 0xFF).toFloat() / 127.5f) - 1.0f
                    val bNorm = ((pixel and 0xFF).toFloat() / 127.5f) - 1.0f

                    byteBuffer.put(quantizeToUint8(rNorm, scale, zp))
                    byteBuffer.put(quantizeToUint8(gNorm, scale, zp))
                    byteBuffer.put(quantizeToUint8(bNorm, scale, zp))
                }
            }
            else -> throw UnsupportedOperationException("Unsupported tensor data type: $dataType")
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    private fun quantizeToInt8(normValue: Float, scale: Float, zeroPoint: Int): Byte {
        val quantized = Math.round(normValue / scale) + zeroPoint
        return quantized.coerceIn(-128, 127).toByte()
    }

    private fun quantizeToUint8(normValue: Float, scale: Float, zeroPoint: Int): Byte {
        val quantized = Math.round(normValue / scale) + zeroPoint
        return (quantized.coerceIn(0, 255) and 0xFF).toByte()
    }
}
