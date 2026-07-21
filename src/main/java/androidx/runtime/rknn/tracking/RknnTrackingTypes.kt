package androidx.runtime.rknn.tracking

import androidx.runtime.rknn.data.RknnDetection

/** Lifecycle state of a tracked detection. */
enum class RknnTrackState {
    /** Track has not yet met the confirmation requirement. */
    TENTATIVE,
    /** Track is confirmed and currently matched. */
    TRACKED,
    /** Track is retained temporarily but was not matched in the current frame. */
    LOST,
}

/**
 * Controls two-stage association in [RknnTracker].
 *
 * @property highScoreThreshold Confidence required for first-stage association.
 * @property lowScoreThreshold Minimum confidence retained for second-stage association.
 * @property matchIouThreshold Minimum intersection-over-union used for a direct match.
 * @property maxCenterDistanceRatio Maximum normalized center distance used for motion recovery.
 * @property minConfirmedFrames Consecutive matches required to confirm a new track.
 * @property maxLostFrames Frames a missing track remains available before removal.
 * @property singleTargetRecovery Reuses the only retained track for the only detection regardless of distance.
 * @property reIdEnabled Whether appearance embeddings participate in association.
 * @property reIdSimilarityThreshold Minimum cosine similarity accepted when both embeddings exist.
 * @property reIdWeight Appearance contribution added to the spatial association score.
 * @property reIdMomentum Previous track feature weight used when updating its appearance embedding.
 * @property duplicateIouThreshold Overlap above which an unmatched detection is treated as a duplicate
 * of a track already matched in the current frame.
 */
data class RknnTrackerConfig(
    val highScoreThreshold: Float = 0.6f,
    val lowScoreThreshold: Float = 0.1f,
    val matchIouThreshold: Float = 0.3f,
    val maxCenterDistanceRatio: Float = 1.5f,
    val minConfirmedFrames: Int = 2,
    val maxLostFrames: Int = 30,
    val singleTargetRecovery: Boolean = false,
    val reIdEnabled: Boolean = false,
    val reIdSimilarityThreshold: Float = 0.75f,
    val reIdWeight: Float = 0.75f,
    val reIdMomentum: Float = 0.9f,
    val duplicateIouThreshold: Float = 0.7f,
) {
    init {
        require(lowScoreThreshold in 0f..1f && highScoreThreshold in 0f..1f)
        require(lowScoreThreshold <= highScoreThreshold)
        require(matchIouThreshold in 0f..1f)
        require(maxCenterDistanceRatio > 0f)
        require(minConfirmedFrames > 0)
        require(maxLostFrames >= 0)
        require(reIdSimilarityThreshold in -1f..1f)
        require(reIdWeight >= 0f)
        require(reIdMomentum in 0f..<1f)
        require(duplicateIouThreshold in 0f..1f)
    }
}

/** Detection and optional normalized appearance feature submitted for one frame. */
data class RknnTrackingInput(
    val detection: RknnDetection,
    val embedding: FloatArray? = null,
)

/**
 * Associates a stable tracking identifier with the latest detection.
 *
 * @property trackId Stable identifier assigned by [RknnTracker].
 * @property detection Latest detection associated with this track.
 * @property state Current tracking lifecycle state.
 * @property ageFrames Number of frames since the track was created.
 * @property missedFrames Consecutive frames without a successful association.
 */
data class RknnTrackedDetection(
    val trackId: Long,
    val detection: RknnDetection,
    val state: RknnTrackState,
    val ageFrames: Int,
    val missedFrames: Int,
)
