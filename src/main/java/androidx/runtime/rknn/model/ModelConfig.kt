package androidx.runtime.rknn.model

import androidx.runtime.rknn.RknnInputLayout
import androidx.runtime.rknn.RknnInputType
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.RknnNormalization
import androidx.runtime.rknn.decoder.MediaPipeObjectDetectorModel
import androidx.runtime.rknn.decoder.RknnDecoderType

/**
 * Configures one RKNN detection model.
 *
 * Example:
 * ```kotlin
 * val action = DetectorModel(
 *     fileName = "action.rknn",
 *     labelFileName = "action.labels",
 *     decoderType = RknnDecoderType.YOLO_END_TO_END,
 * )
 * ```
 *
 * @property enabled Whether the model is loaded and accepts detection requests.
 * @property fileName Model file name relative to the configured model root.
 * @property labels Inline class labels. These take precedence over [labelFileName].
 * @property labelFileName Optional label file name relative to the model root.
 * @property scoreThreshold Minimum confidence accepted by the decoder.
 * @property maxResults Maximum number of detections returned per image.
 * @property inputWidth Model input width in pixels.
 * @property inputHeight Model input height in pixels.
 * @property inputType Input tensor data type, or [RknnInputType.AUTO].
 * @property inputLayout Input tensor layout, or [RknnInputLayout.AUTO].
 * @property normalization Pixel normalization applied before inference.
 * @property decoderType Decoder used to interpret output tensors.
 * @property mediaPipeModel MediaPipe detector specification when applicable.
 * @property modelType Semantic purpose of the model.
 * @property nmsThreshold Intersection-over-union threshold used by NMS.
 * @property poseKeyPointCount Number of keypoints produced by pose models.
 * @property multiLabel Whether a box may contain more than one class result.
 * @property embeddingSize Expected feature vector length for ReID embedding models.
 */
data class DetectorModel(
    val enabled: Boolean = true,
    val fileName: String,
    val labels: List<String> = emptyList(),
    val labelFileName: String? = null,
    val scoreThreshold: Float = 0.25f,
    val maxResults: Int = 100,
    val inputWidth: Int = 640,
    val inputHeight: Int = 640,
    val inputType: RknnInputType = RknnInputType.AUTO,
    val inputLayout: RknnInputLayout = RknnInputLayout.AUTO,
    val normalization: RknnNormalization = RknnNormalization(),
    val decoderType: RknnDecoderType = RknnDecoderType.AUTO,
    val mediaPipeModel: MediaPipeObjectDetectorModel = MediaPipeObjectDetectorModel.AUTO,
    val modelType: RknnModelType = RknnModelType.OBJECT_DETECTOR,
    val nmsThreshold: Float = 0.5f,
    val poseKeyPointCount: Int = 17,
    val multiLabel: Boolean = false,
    val embeddingSize: Int = 512,
) {
    init {
        require(fileName.isNotBlank()) { "Model file name must not be blank" }
        require(scoreThreshold in 0f..1f) { "Score threshold must be between 0 and 1" }
        require(nmsThreshold in 0f..1f) { "NMS threshold must be between 0 and 1" }
        require(maxResults > 0) { "Maximum results must be positive" }
        require(poseKeyPointCount > 0) { "Pose key point count must be positive" }
        require(embeddingSize > 0) { "Embedding size must be positive" }
        require(inputWidth > 0 && inputHeight > 0) { "Model input size must be positive" }
    }
}

/** Controls how inference requests for different models share the RKNN runtime. */
enum class RunMode {
    /** Run all model inference requests through one shared lock. */
    SERIAL,
    /** Allow different models to run concurrently while serializing each model individually. */
    PARALLEL,
}

/**
 * Configures the models managed by [ModelApi].
 *
 * @property models Models indexed by their stable application keys.
 * @property debug Whether verbose runtime and tensor diagnostics are enabled.
 * @property runMode Scheduling policy used for multi-model inference.
 */
data class ModelConfig(
    val models: Map<ModelKey, DetectorModel>,
    val debug: Boolean = false,
    val runMode: RunMode = RunMode.SERIAL,
) {
    init {
        require(models.isNotEmpty()) { "At least one model must be configured" }
    }
}

/**
 * Stable identifier used to register, execute, and observe a model.
 *
 * Use [TARGET], [ACTION], and [MASK] for built-in roles, or create a custom key:
 * ```kotlin
 * val pose = ModelKey("pose")
 * ```
 *
 * @property value Non-blank identifier passed to the low-level runtime.
 */
@JvmInline
value class ModelKey(val value: String) {
    init {
        require(value.isNotBlank()) { "Model key must not be blank" }
    }

    companion object {
        /** Default key for person or target detection. */
        val TARGET = ModelKey("target")
        /** Default key for action detection. */
        val ACTION = ModelKey("action")
        /** Default key for mask or protective-equipment detection. */
        val MASK = ModelKey("mask")
        /** Default key for person ReID embedding extraction. */
        val REID = ModelKey("reid")
    }
}
