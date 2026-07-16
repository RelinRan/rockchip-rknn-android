package androidx.runtime.rknn

import java.io.File

/** 根据系统节点判断当前设备是否提供 Rockchip NPU。 */
internal object RockchipProbe {

    private val supportMarkers = listOf(
        "/d/rknpu/load",
        "/d/rknpu/freq",
        "/dev/rknpu",
        "/sys/kernel/debug/rknpu",
        "/sys/devices/platform/fb000000.gpu",
    )

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
