package androidx.runtime.rknn

import androidx.runtime.rknn.decoder.RknnDecoderType
import androidx.runtime.rknn.decoder.MediaPipeObjectDetectorModel
import androidx.runtime.rknn.decoder.ClassificationScoreType
import androidx.runtime.rknn.decoder.EfficientNetLiteModel
import androidx.runtime.rknn.decoder.HandLandmarkModel
import androidx.runtime.rknn.decoder.PoseLandmarkModel

/**
 * Complete runtime and decoder configuration used to register an RKNN model.
 *
 * Example:
 * ```kotlin
 * val config = RknnModelConfig(
 *     id = "person",
 *     type = RknnModelType.OBJECT_DETECTOR,
 *     fileName = "person.rknn",
 *     inputWidth = 640,
 *     inputHeight = 640,
 *     labels = listOf("person"),
 * )
 * ```
 *
 * @property id Unique non-empty model identifier.
 * @property type Semantic model purpose.
 * @property fileName File name relative to [RknnOptions.modelRoot].
 * @property inputWidth Input tensor width in pixels.
 * @property inputHeight Input tensor height in pixels.
 * @property labels Class labels indexed by model class ID.
 * @property scoreThreshold Minimum accepted confidence.
 * @property maxResults Maximum decoded results returned per request.
 * @property inputType Input tensor data type.
 * @property inputLayout Input tensor layout.
 * @property normalization Pixel normalization applied during preprocessing.
 * @property decoderType Output decoder format.
 * @property mediaPipeModel MediaPipe object detector specification when applicable.
 * @property classifierModel EfficientNet-Lite classification specification.
 * @property classifierScoreType Interpretation of classifier output values.
 * @property poseLandmarkModel MediaPipe pose landmark specification.
 * @property handLandmarkModel MediaPipe hand landmark specification.
 * @property nmsThreshold IoU threshold used by non-maximum suppression.
 * @property poseKeyPointCount Number of keypoints emitted by pose detection.
 * @property multiLabel Whether one box may return multiple categories.
 * @property embeddingSize Expected feature vector length for ReID embedding models.
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
    val embeddingSize: Int = 512,
) {
    init {
        require(inputWidth > 0 && inputHeight > 0) { "Model input size must be positive" }
        require(scoreThreshold in 0f..1f) { "Score threshold must be between 0 and 1" }
        require(nmsThreshold in 0f..1f) { "NMS threshold must be between 0 and 1" }
        require(maxResults > 0) { "Maximum results must be positive" }
        require(poseKeyPointCount > 0) { "Pose key point count must be positive" }
        require(embeddingSize > 0) { "Embedding size must be positive" }
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
