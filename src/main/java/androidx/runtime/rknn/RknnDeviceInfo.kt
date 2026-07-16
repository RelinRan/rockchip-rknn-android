package androidx.runtime.rknn

/** 设备的 RKNN 支持状态以及探测过程中得到的原因。 */
data class RknnDeviceInfo(
    val backend: RknnBackend,
    val supported: Boolean,
    val reasons: List<String> = emptyList(),
)
