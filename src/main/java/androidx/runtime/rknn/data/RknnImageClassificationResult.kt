package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/**
 * Provides the `RknnImageClassificationResult` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnImageClassificationResult` where its surrounding API requires this contract.
 */
data class RknnImageClassificationResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val categories: List<RknnCategory>,
    val durationMs: Long,
    val message: String? = null,
)
