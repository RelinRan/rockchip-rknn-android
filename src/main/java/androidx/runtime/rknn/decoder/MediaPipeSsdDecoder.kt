package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnBoundingBox
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

/** 解码未经过 TFLite 后处理的 MediaPipe Model Maker SSD/RetinaNet 输出。 */
internal object MediaPipeSsdDecoder {
    private const val BOX_COORDINATES = 4
    private const val NMS_IOU_THRESHOLD = 0.5f

    fun decode(
        boxes: NativeTensorOutput,
        scores: NativeTensorOutput,
        image: RknnImage,
        config: RknnModelConfig,
    ): List<RknnDetection> {
        require(boxes.data.size % BOX_COORDINATES == 0) { "MediaPipe boxes must contain Nx4 values" }
        val anchorCount = boxes.data.size / BOX_COORDINATES
        val model = when (config.mediaPipeModel) {
            MediaPipeObjectDetectorModel.AUTO -> MediaPipeObjectDetectorModel.resolve(
                config.inputWidth, config.inputHeight, anchorCount,
            )
            else -> config.mediaPipeModel
        }
        require(config.inputWidth == model.inputSize && config.inputHeight == model.inputSize) {
            "${model.name} requires ${model.inputSize}x${model.inputSize} input"
        }
        require(anchorCount == model.anchorCount) {
            "${model.name} boxes must contain ${model.anchorCount}x4 values"
        }
        val classCount = config.labels.size + 1
        require(scores.data.size == anchorCount * classCount) {
            "MediaPipe scores must contain ${anchorCount}x$classCount values"
        }

        val anchors = generateAnchors(model)
        val scoresAreProbabilities = scores.data.all { it.isFinite() && it in 0f..1f }
        val candidates = ArrayList<RknnDetection>()
        for (anchorIndex in 0 until anchorCount) {
            var bestClass = 0
            var bestScore = 0f
            val scoreOffset = anchorIndex * classCount
            for (classIndex in 1 until classCount) {
                val rawScore = scores.data[scoreOffset + classIndex]
                val score = if (scoresAreProbabilities) rawScore else sigmoid(rawScore)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classIndex
                }
            }
            if (bestScore < config.scoreThreshold) continue
            val anchor = anchors[anchorIndex]
            val boxOffset = anchorIndex * BOX_COORDINATES
            val centerY = boxes.data[boxOffset] / 10f * anchor.height + anchor.centerY
            val centerX = boxes.data[boxOffset + 1] / 10f * anchor.width + anchor.centerX
            val height = exp((boxes.data[boxOffset + 2] / 5f).toDouble()).toFloat() * anchor.height
            val width = exp((boxes.data[boxOffset + 3] / 5f).toDouble()).toFloat() * anchor.width
            val decoded = RknnBoundingBox(
                left = unmapX(centerX - width / 2f, image),
                top = unmapY(centerY - height / 2f, image),
                right = unmapX(centerX + width / 2f, image),
                bottom = unmapY(centerY + height / 2f, image),
            )
            if (decoded.width <= 0f || decoded.height <= 0f) continue
            candidates += RknnDetection(
                categories = listOf(RknnCategory(bestClass - 1, config.labels[bestClass - 1], bestScore)),
                boundingBox = decoded,
            )
        }
        return nonMaximumSuppression(candidates, config.maxResults)
    }

    /** 按模型的特征层、尺度和宽高比生成与训练端顺序一致的锚框。 */
    internal fun generateAnchors(model: MediaPipeObjectDetectorModel): List<Anchor> {
        require(model != MediaPipeObjectDetectorModel.AUTO)
        val anchors = ArrayList<Anchor>(model.anchorCount)
        val scales = floatArrayOf(1f, 2f.pow(1f / 3f), 2f.pow(2f / 3f))
        val ratios = floatArrayOf(0.5f, 1f, 2f)
        for (level in model.minLevel..model.maxLevel) {
            val stride = 1 shl level
            val grid = model.inputSize / stride
            for (y in 0 until grid) for (x in 0 until grid) {
                for (scale in scales) for (ratio in ratios) {
                    val size = 3f * stride * scale
                    val ratioSqrt = sqrt(ratio)
                    anchors += Anchor(
                        centerY = (y + 0.5f) * stride,
                        centerX = (x + 0.5f) * stride,
                        height = size / ratioSqrt,
                        width = size * ratioSqrt,
                    )
                }
            }
        }
        return anchors
    }

    private fun nonMaximumSuppression(candidates: List<RknnDetection>, limit: Int): List<RknnDetection> {
        val selected = ArrayList<RknnDetection>(limit)
        for (candidate in candidates.sortedByDescending { it.categories.first().score }) {
            val classId = candidate.categories.first().index
            if (selected.any { it.categories.first().index == classId && iou(it.boundingBox, candidate.boundingBox) > NMS_IOU_THRESHOLD }) continue
            selected += candidate
            if (selected.size == limit) break
        }
        return selected
    }

    private fun iou(a: RknnBoundingBox, b: RknnBoundingBox): Float {
        val width = (minOf(a.right, b.right) - maxOf(a.left, b.left)).coerceAtLeast(0f)
        val height = (minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)).coerceAtLeast(0f)
        val intersection = width * height
        return intersection / (a.width * a.height + b.width * b.height - intersection).coerceAtLeast(1e-6f)
    }

    private fun sigmoid(value: Float): Float = (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()

    private fun unmapX(value: Float, image: RknnImage): Float =
        ((value - image.paddingLeft) / image.scale).coerceIn(0f, image.originalWidth.toFloat())

    private fun unmapY(value: Float, image: RknnImage): Float =
        ((value - image.paddingTop) / image.scale).coerceIn(0f, image.originalHeight.toFloat())

    internal data class Anchor(val centerY: Float, val centerX: Float, val height: Float, val width: Float)
}
