package androidx.runtime.rknn.model

import androidx.runtime.rknn.RknnInputLayout
import androidx.runtime.rknn.RknnInputType
import androidx.runtime.rknn.RknnNormalization
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.decoder.MediaPipeObjectDetectorModel
import androidx.runtime.rknn.decoder.RknnDecoderType

/**
 * Provides the `DetectorModel` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `DetectorModel` where its surrounding API requires this contract.
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
) {
    init {
        require(fileName.isNotBlank()) { "Model file name must not be blank" }
        require(scoreThreshold in 0f..1f) { "Score threshold must be between 0 and 1" }
        require(nmsThreshold in 0f..1f) { "NMS threshold must be between 0 and 1" }
        require(maxResults > 0) { "Maximum results must be positive" }
        require(poseKeyPointCount > 0) { "Pose key point count must be positive" }
        require(inputWidth > 0 && inputHeight > 0) { "Model input size must be positive" }
    }
}

/**
 * Provides the `ModelExecutionMode` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ModelExecutionMode` where its surrounding API requires this contract.
 */
enum class ModelExecutionMode {
    SERIAL,
    PARALLEL,
}

data class ModelConfig(
    val models: Map<ModelKey, DetectorModel>,
    val debug: Boolean = false,
    val executionMode: ModelExecutionMode = ModelExecutionMode.SERIAL,
) {
    init {
        require(models.isNotEmpty()) { "At least one model must be configured" }
    }
}

/**
 * Provides the `ModelKey` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ModelKey` where its surrounding API requires this contract.
 */
@JvmInline
value class ModelKey(val value: String) {
    init {
        require(value.isNotBlank()) { "Model key must not be blank" }
    }

    companion object {
        val TARGET = ModelKey("target")
        val ACTION = ModelKey("action")
        val MASK = ModelKey("mask")
    }
}
