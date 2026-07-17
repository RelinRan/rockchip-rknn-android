package androidx.runtime.rknn.internal

/**
 * Provides the `NativeTensorOutput` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `NativeTensorOutput` where its surrounding API requires this contract.
 */
data class NativeTensorOutput(
    val index: Int,
    val name: String,
    val dims: IntArray,
    val type: Int,
    val format: Int,
    val data: FloatArray,
)
