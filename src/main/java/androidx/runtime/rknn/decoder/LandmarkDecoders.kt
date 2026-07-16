package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnLandmark
import androidx.runtime.rknn.data.RknnWorldLandmark
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp

/** 解码 MediaPipe Pose Landmark 的关键点、世界坐标和分割掩码。 */
internal object PoseLandmarkDecoder {
    private const val MODEL_LANDMARK_COUNT = 39
    private const val OUTPUT_LANDMARK_COUNT = 33

    fun decode(outputs: List<NativeTensorOutput>, config: RknnModelConfig): DecodedPose {
        val landmarkTensor = outputs.singleOrNull { it.data.size == MODEL_LANDMARK_COUNT * 5 }
            ?: error("Pose landmark tensor [1,195] not found")
        val worldTensor = outputs.singleOrNull { it.data.size == MODEL_LANDMARK_COUNT * 3 }
            ?: error("Pose world landmark tensor [1,117] not found")
        val presenceTensor = outputs.singleOrNull { it.data.size == 1 && it.dims.size <= 2 }
            ?: error("Pose presence tensor [1,1] not found")
        val maskTensor = outputs.singleOrNull { output ->
            output.dims.size >= 3 && output.data.size != 1 && output.dims.any { it == config.inputWidth } &&
                output.dims.any { it == config.inputHeight }
        }
        val landmarks = List(OUTPUT_LANDMARK_COUNT) { index ->
            val offset = index * 5
            RknnLandmark(
                x = landmarkTensor.data[offset] / config.inputWidth,
                y = landmarkTensor.data[offset + 1] / config.inputHeight,
                z = landmarkTensor.data[offset + 2] / config.inputWidth,
                visibility = probability(landmarkTensor.data[offset + 3]),
                presence = probability(landmarkTensor.data[offset + 4]),
            )
        }
        val worldLandmarks = List(OUTPUT_LANDMARK_COUNT) { index ->
            val offset = index * 3
            RknnWorldLandmark(
                worldTensor.data[offset], worldTensor.data[offset + 1], worldTensor.data[offset + 2],
            )
        }
        return DecodedPose(
            landmarks = landmarks,
            worldLandmarks = worldLandmarks,
            posePresence = probability(presenceTensor.data.single()),
            segmentationMask = maskTensor?.data?.let(::probabilities),
            segmentationMaskWidth = maskTensor?.let { tensorWidth(it, config.inputWidth) } ?: 0,
            segmentationMaskHeight = maskTensor?.let { tensorHeight(it, config.inputHeight) } ?: 0,
        )
    }

    private fun tensorWidth(tensor: NativeTensorOutput, fallback: Int): Int = when {
        tensor.dims.size < 3 -> fallback
        tensor.format == 0 -> tensor.dims.last()
        else -> tensor.dims[tensor.dims.size - 2]
    }

    private fun tensorHeight(tensor: NativeTensorOutput, fallback: Int): Int = when {
        tensor.dims.size < 3 -> fallback
        tensor.format == 0 -> tensor.dims[tensor.dims.size - 2]
        else -> tensor.dims[tensor.dims.size - 3]
    }
}

/** 解码 MediaPipe Hand Landmark 的关键点、世界坐标、存在分数和左右手。 */
internal object HandLandmarkDecoder {
    private const val LANDMARK_COUNT = 21

    fun decode(outputs: List<NativeTensorOutput>, config: RknnModelConfig): DecodedHand {
        val ordered = outputs.sortedBy(NativeTensorOutput::index)
        val landmarkTensors = ordered.filter { it.data.size == LANDMARK_COUNT * 3 }
        require(landmarkTensors.size == 2) { "Hand landmark tensors [1,63] not found" }
        val scalarTensors = ordered.filter { it.data.size == 1 }
        require(scalarTensors.size == 2) { "Hand presence and handedness tensors [1,1] not found" }
        val normalized = landmarkTensors[0].data
        val world = landmarkTensors[1].data
        val landmarks = List(LANDMARK_COUNT) { index ->
            val offset = index * 3
            RknnLandmark(
                normalized[offset] / config.inputWidth,
                normalized[offset + 1] / config.inputHeight,
                normalized[offset + 2] / config.inputWidth,
            )
        }
        val worldLandmarks = List(LANDMARK_COUNT) { index ->
            val offset = index * 3
            RknnWorldLandmark(world[offset], world[offset + 1], world[offset + 2])
        }
        val rightScore = probability(scalarTensors[1].data.single())
        val handedness = if (rightScore >= 0.5f) {
            RknnCategory(1, "Right", rightScore)
        } else {
            RknnCategory(0, "Left", 1f - rightScore)
        }
        return DecodedHand(
            landmarks, worldLandmarks, probability(scalarTensors[0].data.single()), handedness,
        )
    }
}

internal data class DecodedPose(
    val landmarks: List<RknnLandmark>,
    val worldLandmarks: List<RknnWorldLandmark>,
    val posePresence: Float,
    val segmentationMask: FloatArray?,
    val segmentationMaskWidth: Int,
    val segmentationMaskHeight: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DecodedPose

        if (posePresence != other.posePresence) return false
        if (segmentationMaskWidth != other.segmentationMaskWidth) return false
        if (segmentationMaskHeight != other.segmentationMaskHeight) return false
        if (landmarks != other.landmarks) return false
        if (worldLandmarks != other.worldLandmarks) return false
        if (segmentationMask != null) {
            if (other.segmentationMask == null) return false
            if (!segmentationMask.contentEquals(other.segmentationMask)) return false
        } else if (other.segmentationMask != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = posePresence.hashCode()
        result = 31 * result + segmentationMaskWidth
        result = 31 * result + segmentationMaskHeight
        result = 31 * result + landmarks.hashCode()
        result = 31 * result + worldLandmarks.hashCode()
        result = 31 * result + (segmentationMask?.contentHashCode() ?: 0)
        return result
    }
}

internal data class DecodedHand(
    val landmarks: List<RknnLandmark>,
    val worldLandmarks: List<RknnWorldLandmark>,
    val handPresence: Float,
    val handedness: RknnCategory,
)

private fun probability(value: Float): Float = if (value in 0f..1f) value else sigmoid(value)

private fun probabilities(values: FloatArray): FloatArray = if (values.all { it in 0f..1f }) {
    values.copyOf()
} else {
    FloatArray(values.size) { index -> sigmoid(values[index]) }
}

private fun sigmoid(value: Float): Float = (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
