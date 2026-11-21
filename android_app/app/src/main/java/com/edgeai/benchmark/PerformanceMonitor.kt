package com.edgeai.benchmark

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.Process
import java.io.BufferedReader
import java.io.FileReader

/**
 * PerformanceMonitor
 * ------------------
 * Collects hardware telemetry (SoC, CPU cores, RAM, Android OS, Battery state)
 * and samples Proportional Set Size (PSS) process memory via Debug.MemoryInfo.
 */
class PerformanceMonitor(private val context: Context) {

    data class DeviceTelemetry(
        val manufacturer: String,
        val model: String,
        val chipset: String,
        val androidVersion: String,
        val sdkInt: Int,
        val buildId: String,
        val cpuCores: Int,
        val totalRamMb: Double,
        val availableRamMb: Double,
        val batteryPct: Float,
        val batteryTempCelsius: Float
    )

    data class MemorySnapshot(
        val baselinePssMb: Double,
        var peakPssMb: Double
    ) {
        val memoryIncreaseMb: Double
            get() = String.format("%.2f", (peakPssMb - baselinePssMb).coerceAtLeast(0.0)).toDouble()
    }

    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    fun getDeviceTelemetry(): DeviceTelemetry {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalRamMb = memInfo.totalMem.toDouble() / (1024 * 1024)
        val availRamMb = memInfo.availMem.toDouble() / (1024 * 1024)

        val chipset = readSocOrHardware()

        // Battery state
        val ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val batteryStatus: Intent? = context.registerReceiver(null, ifilter)
        val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val batteryPct = if (level >= 0 && scale > 0) (level / scale.toFloat()) * 100f else -1f
        val tempRaw = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val batteryTemp = tempRaw / 10f

        return DeviceTelemetry(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            chipset = chipset,
            androidVersion = Build.VERSION.RELEASE,
            sdkInt = Build.VERSION.SDK_INT,
            buildId = Build.DISPLAY,
            cpuCores = Runtime.getRuntime().availableProcessors(),
            totalRamMb = String.format("%.2f", totalRamMb).toDouble(),
            availableRamMb = String.format("%.2f", availRamMb).toDouble(),
            batteryPct = batteryPct,
            batteryTempCelsius = batteryTemp
        )
    }

    fun sampleProcessPssMb(): Double {
        return try {
            val pids = intArrayOf(Process.myPid())
            val memInfos = activityManager.getProcessMemoryInfo(pids)
            if (memInfos != null && memInfos.isNotEmpty()) {
                val pssKb = memInfos[0].totalPss
                pssKb.toDouble() / 1024.0
            } else {
                val pssKb = Debug.getPss()
                pssKb.toDouble() / 1024.0
            }
        } catch (e: Exception) {
            val pssKb = Debug.getPss()
            pssKb.toDouble() / 1024.0
        }
    }

    private fun readSocOrHardware(): String {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                val soc = Build.SOC_MODEL
                if (!soc.isNullOrBlank() && soc != Build.UNKNOWN) {
                    return soc
                }
            } catch (_: Exception) {}
        }

        // Try reading /proc/cpuinfo Hardware line
        try {
            val reader = BufferedReader(FileReader("/proc/cpuinfo"))
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                if (line!!.startsWith("Hardware")) {
                    val parts = line!!.split(":")
                    if (parts.size > 1) {
                        reader.close()
                        return parts[1].trim()
                    }
                }
            }
            reader.close()
        } catch (_: Exception) {}

        return Build.HARDWARE
    }
}
