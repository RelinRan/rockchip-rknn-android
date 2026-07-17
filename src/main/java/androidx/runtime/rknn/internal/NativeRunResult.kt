package androidx.runtime.rknn.internal

/**
 * Provides the `NativeRunResult` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `NativeRunResult` where its surrounding API requires this contract.
 */
data class NativeRunResult(
    val success: Boolean,
    val durationMs: Long,
    val message: String?,
    val apiVersion: String?,
    val driverVersion: String?,
    val outputs: List<NativeTensorOutput>,
)
