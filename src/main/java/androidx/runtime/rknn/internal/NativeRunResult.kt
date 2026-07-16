package androidx.runtime.rknn.internal

/** 原生桥接层单次执行返回的数据。 */
data class NativeRunResult(
    val success: Boolean,
    val durationMs: Long,
    val message: String?,
    val apiVersion: String?,
    val driverVersion: String?,
    val outputs: List<NativeTensorOutput>,
)
