package androidx.runtime.rknn.data

/**
 * A YOLO Pose keypoint mapped to source-image coordinates.
 *
 * Read instances from `RknnDetection.keyPoints` and draw [x], [y] when [score]
 * meets the application's visibility threshold.
 *
 * @property index Zero-based keypoint index in the pose model.
 * @property name Semantic keypoint name, such as `nose`.
 * @property x Horizontal position in source-image pixels.
 * @property y Vertical position in source-image pixels.
 * @property score Keypoint confidence, normally in the `0..1` range.
 */
data class RknnKeypoint(
    val index: Int,
    val name: String,
    val x: Float,
    val y: Float,
    val score: Float,
)
