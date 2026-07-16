package androidx.runtime.rknn.internal

import android.os.Build
import java.io.File

/** 读取芯片型号、NPU 驱动版本等运行环境信息。 */
internal object RknnEnvironmentProbe {

    private val driverVersionFiles = listOf(
        File("/d/rknpu/version"),
        File("/sys/kernel/debug/rknpu/version"),
    )

    fun chip(): String = runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
    }.getOrNull().orEmpty().ifBlank { "unknown" }

    fun driverVersion(): String = driverVersionFiles.firstNotNullOfOrNull { file ->
        runCatching {
            if (!file.isFile) return@runCatching null
            file.readText()
                .lineSequence()
                .firstOrNull { it.contains("RKNPU driver", ignoreCase = true) }
                ?.substringAfter(':')
                ?.trim()
                ?.removePrefix("v")
                ?.ifBlank { null }
        }.getOrNull()
    } ?: "unknown"
}
