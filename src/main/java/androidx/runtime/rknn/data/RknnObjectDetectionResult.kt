package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/** 目标检测的业务结果。 */
data class RknnObjectDetectionResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val detections: List<RknnDetection>,
    val durationMs: Long,
    val message: String? = null,
)
