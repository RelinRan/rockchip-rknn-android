package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnOrientedBox
import androidx.runtime.rknn.internal.NativeTensorOutput

/** 解码 YOLO OBB 的中心点、尺寸、类别分数和弧度角。 */
internal object YoloObbDecoder {
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
