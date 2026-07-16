package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/** Pose Landmark 解码结果。 */
data class RknnPoseLandmarkResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val landmarks: List<RknnLandmark>,
    val worldLandmarks: List<RknnWorldLandmark>,
    val posePresence: Float,
    val segmentationMask: FloatArray? = null,
    val segmentationMaskWidth: Int = 0,
    val segmentationMaskHeight: Int = 0,
    val durationMs: Long = 0,
    val message: String? = null,
)

/** Hand Landmark 解码结果。 */
data class RknnHandLandmarkResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val landmarks: List<RknnLandmark>,
    val worldLandmarks: List<RknnWorldLandmark>,
    val handPresence: Float,
    val handedness: RknnCategory,
    val durationMs: Long = 0,
    val message: String? = null,
)
