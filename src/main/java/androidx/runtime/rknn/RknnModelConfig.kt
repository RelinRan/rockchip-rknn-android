package androidx.runtime.rknn

import androidx.runtime.rknn.decoder.RknnDecoderType
import androidx.runtime.rknn.decoder.MediaPipeObjectDetectorModel
import androidx.runtime.rknn.decoder.ClassificationScoreType
import androidx.runtime.rknn.decoder.EfficientNetLiteModel
import androidx.runtime.rknn.decoder.HandLandmarkModel
import androidx.runtime.rknn.decoder.PoseLandmarkModel

/**
 * Provides the `RknnModelConfig` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnModelConfig` where its surrounding API requires this contract.
 */
data class RknnModelConfig(
    val id: String,
    val type: RknnModelType,
    val fileName: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val labels: List<String> = emptyList(),
    val scoreThreshold: Float = 0.25f,
    val maxResults: Int = 100,
    val inputType: RknnInputType = RknnInputType.AUTO,
    val inputLayout: RknnInputLayout = RknnInputLayout.AUTO,
    val normalization: RknnNormalization = RknnNormalization(),
    val decoderType: RknnDecoderType = RknnDecoderType.AUTO,
    val mediaPipeModel: MediaPipeObjectDetectorModel = MediaPipeObjectDetectorModel.AUTO,
    val classifierModel: EfficientNetLiteModel? = null,
    val classifierScoreType: ClassificationScoreType = ClassificationScoreType.AUTO,
    val poseLandmarkModel: PoseLandmarkModel? = null,
    val handLandmarkModel: HandLandmarkModel? = null,
    val nmsThreshold: Float = 0.5f,
    val poseKeyPointCount: Int = 17,
    val multiLabel: Boolean = false,
) {
    init {
        require(inputWidth > 0 && inputHeight > 0) { "Model input size must be positive" }
        require(scoreThreshold in 0f..1f) { "Score threshold must be between 0 and 1" }
        require(nmsThreshold in 0f..1f) { "NMS threshold must be between 0 and 1" }
        require(maxResults > 0) { "Maximum results must be positive" }
        require(poseKeyPointCount > 0) { "Pose key point count must be positive" }
        classifierModel?.let { model ->
            require(inputWidth == model.inputSize && inputHeight == model.inputSize) {
                "${model.name} requires ${model.inputSize}x${model.inputSize} input"
            }
        }
        poseLandmarkModel?.let { model ->
            require(inputWidth == model.inputSize && inputHeight == model.inputSize) {
                "${model.name} requires ${model.inputSize}x${model.inputSize} input"
            }
        }
        handLandmarkModel?.let { model ->
            require(inputWidth == model.inputSize && inputHeight == model.inputSize) {
                "${model.name} requires ${model.inputSize}x${model.inputSize} input"
            }
        }
    }
}
