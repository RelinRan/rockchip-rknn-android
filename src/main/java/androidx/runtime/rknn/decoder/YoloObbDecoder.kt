package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnOrientedBox
import androidx.runtime.rknn.internal.NativeTensorOutput

/**
 * Provides the `YoloObbDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloObbDecoder` where its surrounding API requires this contract.
 */
internal object YoloObbDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param output Native output tensor to decode.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(output: NativeTensorOutput, image: RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val channels = 5 + config.labels.size
        val view = YoloTensorView.create(output, setOf(channels))
        val candidates = ArrayList<YoloCandidate>()
        repeat(view.rows) { row ->
            val categories = config.labels.indices.map { classId -> classId to view[row, 4 + classId] }
                .filter { it.second.isFinite() && it.second >= config.scoreThreshold }
                .sortedByDescending { it.second }
            if (categories.isEmpty()) return@repeat
            val centerX = view[row, 0]
            val centerY = view[row, 1]
            val width = view[row, 2]
            val height = view[row, 3]
            val angle = view[row, 4 + config.labels.size]
            if (width <= 0f || height <= 0f || !angle.isFinite()) return@repeat
            val oriented = RknnOrientedBox(centerX, centerY, width, height, angle)
            val bounds = oriented.boundingBox()
            val primary = categories.first()
            candidates += YoloCandidate(
                bounds.left, bounds.top, bounds.right, bounds.bottom,
                primary.first, primary.second,
                categoryScores = if (config.multiLabel) categories else listOf(primary),
                orientedBox = oriented,
            )
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }
}
