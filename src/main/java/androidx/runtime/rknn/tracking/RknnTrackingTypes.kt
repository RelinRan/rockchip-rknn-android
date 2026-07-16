package androidx.runtime.rknn.tracking

import androidx.runtime.rknn.data.RknnDetection

enum class RknnTrackState { TENTATIVE, TRACKED, LOST }

data class RknnTrackerConfig(
    val highScoreThreshold: Float = 0.6f,
    val lowScoreThreshold: Float = 0.1f,
    val matchIouThreshold: Float = 0.3f,
    val maxCenterDistanceRatio: Float = 1.5f,
    val minConfirmedFrames: Int = 2,
    val maxLostFrames: Int = 30,
) {
    init {
        require(lowScoreThreshold in 0f..1f && highScoreThreshold in 0f..1f)
        require(lowScoreThreshold <= highScoreThreshold)
        require(matchIouThreshold in 0f..1f)
        require(maxCenterDistanceRatio > 0f)
        require(minConfirmedFrames > 0)
        require(maxLostFrames >= 0)
    }
}

data class RknnTrackedDetection(
    val trackId: Long,
    val detection: RknnDetection,
    val state: RknnTrackState,
    val ageFrames: Int,
    val missedFrames: Int,
)
