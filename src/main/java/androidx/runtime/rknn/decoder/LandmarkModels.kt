package androidx.runtime.rknn.decoder

/**
 * Provides the `PoseLandmarkModel` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `PoseLandmarkModel` where its surrounding API requires this contract.
 */
enum class PoseLandmarkModel(val inputSize: Int) {
    LITE(256),
    FULL(256),
    HEAVY(256),
}

/**
 * Provides the `HandLandmarkModel` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `HandLandmarkModel` where its surrounding API requires this contract.
 */
enum class HandLandmarkModel(val inputSize: Int) {
    FULL(224),
}
