package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/**
 * Provides the `RawYoloDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RawYoloDecoder` where its surrounding API requires this contract.
 */
internal object RawYoloDecoder {
    private const val NMS_THRESHOLD = 0.5f

    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param tensor Native tensor containing model output values and dimensions.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(tensor: NativeTensorOutput, image: RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val withoutObjectness = config.labels.size + 4
        val withObjectness = config.labels.size + 5
        val view = YoloTensorView.create(tensor, setOf(withoutObjectness, withObjectness))
        val objectnessOffset = if (view.channels == withObjectness) 1 else 0
        val candidates = ArrayList<YoloCandidate>()
        for (anchor in 0 until view.rows) {
            val objectness = if (objectnessOffset == 1) view[anchor, 4] else 1f
            val passingCategories = config.labels.indices.map { candidateClass ->
                candidateClass to view[anchor, candidateClass + 4 + objectnessOffset] * objectness
            }.filter { (_, candidateScore) -> candidateScore.isFinite() && candidateScore >= config.scoreThreshold }
                .sortedByDescending { it.second }
            var classId = -1
            var score = Float.NEGATIVE_INFINITY
            for (candidateClass in config.labels.indices) {
                val value = view[anchor, candidateClass + 4 + objectnessOffset]
                if (value > score) {
                    score = value
                    classId = candidateClass
                }
            }
            score *= objectness
            if (!score.isFinite() || score < config.scoreThreshold) continue
            val centerX = view[anchor, 0]
            val centerY = view[anchor, 1]
            val width = view[anchor, 2]
            val height = view[anchor, 3]
            if (!centerX.isFinite() || !centerY.isFinite() || width <= 0f || height <= 0f) continue
            candidates += YoloCandidate(
                centerX - width / 2f, centerY - height / 2f,
                centerX + width / 2f, centerY + height / 2f, classId, score,
                categoryScores = if (config.multiLabel) passingCategories else listOf(classId to score),
            )
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }
}
