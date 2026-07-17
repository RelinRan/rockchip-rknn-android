package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnSegmentationMask
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp

/**
 * Provides the `YoloSegmentationDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloSegmentationDecoder` where its surrounding API requires this contract.
 */
internal object YoloSegmentationDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param outputs Native output tensors to decode.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(
        outputs: List<NativeTensorOutput>,
        image: RknnImage,
        config: RknnModelConfig,
    ): List<RknnDetection> {
        val prototypeOutput = outputs.singleOrNull { it.dims.size == 4 }
            ?: error("YOLO segmentation requires exactly one rank-4 prototype tensor")
        val prototype = Prototype.create(prototypeOutput)
        val detectionOutput = outputs.singleOrNull { it !== prototypeOutput }
            ?: error("YOLO segmentation requires exactly one detection tensor")
        val channels = 4 + config.labels.size + prototype.channels
        val view = YoloTensorView.create(detectionOutput, setOf(channels))
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
            val left = centerX - width / 2f
            val top = centerY - height / 2f
            val right = centerX + width / 2f
            val bottom = centerY + height / 2f
            val coefficients = FloatArray(prototype.channels) { view[row, 4 + config.labels.size + it] }
            val mask = prototype.combine(
                coefficients, left, top, right, bottom, config.inputWidth, config.inputHeight,
            )
            val primary = categories.first()
            candidates += YoloCandidate(
                left, top, right, bottom,
                primary.first, primary.second,
                categoryScores = if (config.multiLabel) categories else listOf(primary),
                segmentationMask = mask,
            )
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }

    private class Prototype private constructor(
        private val tensor: NativeTensorOutput,
        val channels: Int,
        val height: Int,
        val width: Int,
        private val channelFirst: Boolean,
    ) {
        /**
         * Executes `combine` for the RKNN runtime contract.
         * @param coefficients Value supplied for `coefficients`.
         * @param left Value supplied for `left`.
         * @param top Value supplied for `top`.
         * @param right Value supplied for `right`.
         * @param bottom Value supplied for `bottom`.
         * @param inputWidth Value supplied for `inputWidth`.
         * @param inputHeight Value supplied for `inputHeight`.
         */
        fun combine(
            coefficients: FloatArray,
            left: Float,
            top: Float,
            right: Float,
            bottom: Float,
            inputWidth: Int,
            inputHeight: Int,
        ): RknnSegmentationMask {
            val probabilities = FloatArray(width * height)
            repeat(height) { y ->
                repeat(width) { x ->
                    val modelX = (x + 0.5f) * inputWidth / width
                    val modelY = (y + 0.5f) * inputHeight / height
                    if (modelX !in left..right || modelY !in top..bottom) {
                        probabilities[y * width + x] = 0f
                        return@repeat
                    }
                    var logit = 0f
                    repeat(channels) { channel ->
                        val index = if (channelFirst) {
                            channel * height * width + y * width + x
                        } else {
                            (y * width + x) * channels + channel
                        }
                        logit += coefficients[channel] * tensor.data[index]
                    }
                    probabilities[y * width + x] = (1.0 / (1.0 + exp(-logit.toDouble()))).toFloat()
                }
            }
            return RknnSegmentationMask(width, height, probabilities)
        }

        companion object {
            /**
             * Executes `create` for the RKNN runtime contract.
             * @param tensor Native tensor containing model output values and dimensions.
             */
            fun create(tensor: NativeTensorOutput): Prototype {
                val dims = tensor.dims
                require(dims.size == 4 && dims[0] == 1) { "Invalid prototype dimensions: ${dims.contentToString()}" }
                val channelFirst = when (tensor.format) {
                    0 -> true
                    1 -> false
                    else -> error("Unsupported prototype format: ${tensor.format}")
                }
                return if (channelFirst) {
                    Prototype(tensor, dims[1], dims[2], dims[3], true)
                } else {
                    Prototype(tensor, dims[3], dims[1], dims[2], false)
                }
            }
        }
    }
}
