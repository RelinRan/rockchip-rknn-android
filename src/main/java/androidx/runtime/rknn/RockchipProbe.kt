package androidx.runtime.rknn

import java.io.File

/**
 * Provides the `RockchipProbe` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RockchipProbe` where its surrounding API requires this contract.
 */
internal object RockchipProbe {

    private val supportMarkers = listOf(
        "/d/rknpu/load",
        "/d/rknpu/freq",
        "/dev/rknpu",
        "/sys/kernel/debug/rknpu",
        "/sys/devices/platform/fb000000.gpu",
    )

    /**
     * Executes `isRockchipNpuAvailable` for the RKNN runtime contract.
     * @param reasons Value supplied for `reasons`.
     */
    fun isRockchipNpuAvailable(reasons: MutableList<String> = mutableListOf()): Boolean {
        val hit = supportMarkers.firstOrNull { File(it).exists() }
        if (hit != null) {
            reasons += "marker:$hit"
            return true
        }
        reasons += "no rockchip npu marker found"
        return false
    }
}
