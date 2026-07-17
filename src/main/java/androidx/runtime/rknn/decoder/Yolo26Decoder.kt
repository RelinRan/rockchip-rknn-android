package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Provides the `Yolo26Decoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `Yolo26Decoder` where its surrounding API requires this contract.
 */
internal object Yolo26Decoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param tensor Native tensor containing model output values and dimensions.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(tensor: NativeTensorOutput, image: androidx.runtime.rknn.data.RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val view = YoloTensorView.create(tensor, setOf(6))
        val candidates = ArrayList<YoloCandidate>(view.rows)
        for (row in 0 until view.rows) {
            val score = view[row, 4]
            val rawClassId = view[row, 5]
            val classId = rawClassId.roundToInt()
            if (!score.isFinite() || score < config.scoreThreshold ||
                !rawClassId.isFinite() || abs(rawClassId - classId) > 0.001f || classId !in config.labels.indices
            ) continue
            val left = view[row, 0]
            val top = view[row, 1]
            val right = view[row, 2]
            val bottom = view[row, 3]
            if (!listOf(left, top, right, bottom).all { it.isFinite() } || right <= left || bottom <= top) continue
            candidates += YoloCandidate(left, top, right, bottom, classId, score)
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = false)
    }
}
