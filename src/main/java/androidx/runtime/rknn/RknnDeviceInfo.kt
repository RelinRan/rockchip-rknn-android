package androidx.runtime.rknn

/**
 * Snapshot of RKNN capability discovery for the current device.
 *
 * Obtain it with `runtime.deviceInfo()` after initialization.
 *
 * @property backend Backend that was probed.
 * @property supported Whether the required runtime and device nodes are available.
 * @property reasons Human-readable discovery failures; empty when fully supported.
 */
data class RknnDeviceInfo(
    val backend: RknnBackend,
    val supported: Boolean,
    val reasons: List<String> = emptyList(),
)
