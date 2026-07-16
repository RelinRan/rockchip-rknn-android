package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnKeypoint
import androidx.runtime.rknn.internal.NativeTensorOutput

/** 解码未执行 NMS 的 YOLO Pose 单张量输出。 */
internal object YoloRawPoseDecoder {
    fun decode(tensor: NativeTensorOutput, image: RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val keyPointChannels = config.poseKeyPointCount * 3
        val withoutObjectness = 4 + config.labels.size + keyPointChannels
        val withObjectness = 5 + config.labels.size + keyPointChannels
        val view = YoloTensorView.create(tensor, setOf(withoutObjectness, withObjectness))
        val hasObjectness = view.channels == withObjectness
        val classOffset = if (hasObjectness) 5 else 4
        val keyPointOffset = classOffset + config.labels.size
        val candidates = ArrayList<YoloCandidate>()
        repeat(view.rows) { row ->
            var classId = -1
            var score = Float.NEGATIVE_INFINITY
            config.labels.indices.forEach { candidateClass ->
                val value = view[row, classOffset + candidateClass]
                if (value > score) {
                    score = value
                    classId = candidateClass
                }
            }
            if (hasObjectness) score *= view[row, 4]
            if (!score.isFinite() || score < config.scoreThreshold) return@repeat
            val centerX = view[row, 0]
            val centerY = view[row, 1]
            val width = view[row, 2]
            val height = view[row, 3]
            val keyPoints = List(config.poseKeyPointCount) { index ->
                val offset = keyPointOffset + index * 3
                RknnKeypoint(
                    index, YoloPoseNames.name(index), view[row, offset], view[row, offset + 1],
                    view[row, offset + 2].takeIf(Float::isFinite) ?: 0f,
                )
            }
            candidates += YoloCandidate(
                centerX - width / 2f, centerY - height / 2f,
                centerX + width / 2f, centerY + height / 2f,
                classId, score, keyPoints,
            )
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }
}
