package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection

/** 不同模型解码结果的统一内部类型。 */
internal sealed interface RknnDecodedResult {
    data class Detection(val detections: List<RknnDetection>) : RknnDecodedResult
    data class Classification(val categories: List<RknnCategory>) : RknnDecodedResult
    data class PoseLandmark(val pose: DecodedPose) : RknnDecodedResult
    data class HandLandmark(val hand: DecodedHand) : RknnDecodedResult
}
