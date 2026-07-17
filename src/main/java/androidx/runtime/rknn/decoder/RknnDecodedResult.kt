package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection

/**
 * Provides the `RknnDecodedResult` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnDecodedResult` where its surrounding API requires this contract.
 */
internal sealed interface RknnDecodedResult {
    data class Detection(val detections: List<RknnDetection>) : RknnDecodedResult
    data class Classification(val categories: List<RknnCategory>) : RknnDecodedResult
    data class PoseLandmark(val pose: DecodedPose) : RknnDecodedResult
    data class HandLandmark(val hand: DecodedHand) : RknnDecodedResult
}
