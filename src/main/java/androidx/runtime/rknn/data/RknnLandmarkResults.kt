package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/**
 * Provides the `RknnPoseLandmarkResult` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnPoseLandmarkResult` where its surrounding API requires this contract.
 */
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

/**
 * Provides the `RknnHandLandmarkResult` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnHandLandmarkResult` where its surrounding API requires this contract.
 */
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
