package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnKeypoint
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Provides the `YoloPoseDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloPoseDecoder` where its surrounding API requires this contract.
 */
internal object YoloPoseDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param tensor Native tensor containing model output values and dimensions.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(tensor: NativeTensorOutput, image: RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val valuesPerRow = 6 + config.poseKeyPointCount * 3
        val view = YoloTensorView.create(tensor, setOf(valuesPerRow))
        val candidates = ArrayList<YoloCandidate>()
        repeat(view.rows) { row ->
            val score = view[row, 4]
            val rawClass = view[row, 5]
            val classId = rawClass.roundToInt()
            if (!score.isFinite() || score < config.scoreThreshold || !rawClass.isFinite() ||
                abs(rawClass - classId) > 0.001f || classId !in config.labels.indices
            ) return@repeat
            val keyPoints = List(config.poseKeyPointCount) { index ->
                val offset = 6 + index * 3
                RknnKeypoint(
                    index = index,
                    name = YoloPoseNames.name(index),
                    x = view[row, offset],
                    y = view[row, offset + 1],
                    score = view[row, offset + 2].takeIf(Float::isFinite) ?: 0f,
                )
            }
            candidates += YoloCandidate(
                view[row, 0], view[row, 1], view[row, 2], view[row, 3],
                classId, score, keyPoints,
            )
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = false)
    }
}

internal object YoloPoseNames {
    private val coco = listOf(
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )

    /**
     * Executes `name` for the RKNN runtime contract.
     * @param index Value supplied for `index`.
     */
    fun name(index: Int): String = coco.getOrNull(index) ?: "key_point_$index"
}
