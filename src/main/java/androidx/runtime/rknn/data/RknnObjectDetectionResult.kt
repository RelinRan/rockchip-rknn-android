package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.ModelResult

/**
 * Application-level object detection result.
 *
 * @property success Whether inference and decoding completed successfully.
 * @property backend Backend that handled the request.
 * @property modelId Registered model identifier.
 * @property detections Decoded detections in source image coordinates.
 * @property durationMs End-to-end inference duration in milliseconds.
 * @property message Optional diagnostic or failure reason.
 * @property resultSequence Monotonic sequence assigned when the result is published.
 */
data class RknnObjectDetectionResult(
    override val success: Boolean,
    override val backend: RknnBackend,
    override val modelId: String,
    val detections: List<RknnDetection>,
    override val durationMs: Long,
    override val message: String? = null,
    override val resultSequence: Long = 0L,
) : ModelResult {
    override fun withResultSequence(sequence: Long): ModelResult = copy(resultSequence = sequence)
}
