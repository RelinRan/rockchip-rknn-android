package androidx.runtime.rknn.data

/**
 * A decoded object containing its geometry and ranked category candidates.
 *
 * Obtain instances from `RknnObjectDetectionResult.detections`, then use
 * `categories.firstOrNull()` as the highest-scoring category.
 *
 * @property categories Matching categories ordered by descending score.
 * @property boundingBox Axis-aligned bounds in source-image pixels.
 * @property keyPoints Optional pose keypoints in source-image pixels.
 * @property segmentationMask Optional instance mask in model-input space.
 * @property orientedBox Optional rotated bounds for OBB models.
 */
data class RknnDetection(
    val categories: List<RknnCategory>,
    val boundingBox: RknnBoundingBox,
    val keyPoints: List<RknnKeypoint> = emptyList(),
    val segmentationMask: RknnSegmentationMask? = null,
    val orientedBox: RknnOrientedBox? = null,
)
