package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/**
 * Result returned by an object-detection or pose-detection request.
 *
 * Check [success] before consuming [detections]; when false, inspect [message].
 *
 * @property success Whether inference and decoding completed successfully.
 * @property backend Backend that produced the result.
 * @property modelId Identifier of the registered model.
 * @property detections Decoded detections, empty when none matched or the request failed.
 * @property durationMs End-to-end inference duration in milliseconds.
 * @property message Diagnostic text for failures or noteworthy conditions.
 */
data class RknnObjectDetectionResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val detections: List<RknnDetection>,
    val durationMs: Long,
    val message: String? = null,
)
