package androidx.runtime.rknn.data

/**
 * Provides the `RknnLandmark` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnLandmark` where its surrounding API requires this contract.
 */
data class RknnLandmark(
    val x: Float,
    val y: Float,
    val z: Float,
    val visibility: Float = 1f,
    val presence: Float = 1f,
)

/**
 * Provides the `RknnWorldLandmark` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnWorldLandmark` where its surrounding API requires this contract.
 */
data class RknnWorldLandmark(
    val x: Float,
    val y: Float,
    val z: Float,
)
