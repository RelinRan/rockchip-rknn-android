package androidx.runtime.rknn.internal

import android.os.Build
import java.io.File

/**
 * Provides the `RknnEnvironmentProbe` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnEnvironmentProbe` where its surrounding API requires this contract.
 */
internal object RknnEnvironmentProbe {

    private val driverVersionFiles = listOf(
        File("/d/rknpu/version"),
        File("/sys/kernel/debug/rknpu/version"),
    )

    /**
     * Executes `chip` for the RKNN runtime contract.
     */
    fun chip(): String = runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
    }.getOrNull().orEmpty().ifBlank { "unknown" }

    /**
     * Executes `driverVersion` for the RKNN runtime contract.
     */
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
