package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnKeypoint
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp

/**
 * Provides the `YoloPoseHeadsDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloPoseHeadsDecoder` where its surrounding API requires this contract.
 */
internal object YoloPoseHeadsDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param outputs Native output tensors to decode.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(outputs: List<NativeTensorOutput>, image: RknnImage, config: RknnModelConfig): List<RknnDetection> {
        val heads = outputs.map(Head::create)
        val poseChannels = config.poseKeyPointCount * 3
        val poseHeads = heads.filter { it.channels == poseChannels }
        val scoreHeads = heads.filter { it.channels == config.labels.size }
        val boxHeads = heads.filter { it.channels % 4 == 0 && it.channels / 4 > 1 && it.channels != poseChannels }
        val candidates = ArrayList<YoloCandidate>()
        boxHeads.forEach { box ->
            val scores = scoreHeads.filter { it.sameGrid(box) }
            val poses = poseHeads.filter { it.sameGrid(box) }
            require(scores.size == 1 && poses.size == 1) {
                "Pose head ${box.tensor.name} requires one class and one key point tensor on ${box.width}x${box.height}"
            }
            decodeGrid(box, scores.single(), poses.single(), config, candidates)
        }
        require(boxHeads.isNotEmpty()) { "DFL Pose regression heads not found" }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }

    private fun decodeGrid(
        box: Head,
        scores: Head,
        pose: Head,
        config: RknnModelConfig,
        candidates: MutableList<YoloCandidate>,
    ) {
        val regMax = box.channels / 4
        val strideX = config.inputWidth.toFloat() / box.width
        val strideY = config.inputHeight.toFloat() / box.height
        repeat(box.height) { y -> repeat(box.width) { x ->
            val categories = config.labels.indices.map { it to scores[x, y, it] }
                .filter { it.second.isFinite() && it.second >= config.scoreThreshold }
                .sortedByDescending { it.second }
            if (categories.isEmpty()) return@repeat
            val centerX = (x + 0.5f) * strideX
            val centerY = (y + 0.5f) * strideY
            val keyPoints = List(config.poseKeyPointCount) { index ->
                val offset = index * 3
                RknnKeypoint(
                    index, YoloPoseNames.name(index),
                    (pose[x, y, offset] * 2f + x - 0.5f) * strideX,
                    (pose[x, y, offset + 1] * 2f + y - 0.5f) * strideY,
                    sigmoid(pose[x, y, offset + 2]),
                )
            }
            val primary = categories.first()
            candidates += YoloCandidate(
                centerX - expectation(box, x, y, 0, regMax) * strideX,
                centerY - expectation(box, x, y, 1, regMax) * strideY,
                centerX + expectation(box, x, y, 2, regMax) * strideX,
                centerY + expectation(box, x, y, 3, regMax) * strideY,
                primary.first, primary.second, keyPoints,
                if (config.multiLabel) categories else listOf(primary),
            )
        } }
    }

    private fun expectation(head: Head, x: Int, y: Int, side: Int, regMax: Int): Float {
        val offset = side * regMax
        var maximum = Float.NEGATIVE_INFINITY
        repeat(regMax) { maximum = maxOf(maximum, head[x, y, offset + it]) }
        var sum = 0.0
        var weighted = 0.0
        repeat(regMax) { bin ->
            val value = exp((head[x, y, offset + bin] - maximum).toDouble())
            sum += value
            weighted += value * bin
        }
        return (weighted / sum).toFloat()
    }

    private fun sigmoid(value: Float) = (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()

    private class Head private constructor(
        val tensor: NativeTensorOutput,
        val channels: Int,
        val height: Int,
        val width: Int,
        private val channelFirst: Boolean,
    ) {
        operator fun get(x: Int, y: Int, channel: Int): Float {
            val index = if (channelFirst) channel * height * width + y * width + x
            else (y * width + x) * channels + channel
            return tensor.data[index]
        }
        /**
         * Executes `sameGrid` for the RKNN runtime contract.
         * @param other Value supplied for `other`.
         */
        fun sameGrid(other: Head) = width == other.width && height == other.height

        companion object {
            /**
             * Executes `create` for the RKNN runtime contract.
             * @param tensor Native tensor containing model output values and dimensions.
             */
            fun create(tensor: NativeTensorOutput): Head {
                val dims = tensor.dims
                require(dims.size == 4 && dims[0] == 1) { "Pose head must be rank-4: ${tensor.name}" }
                return when (tensor.format) {
                    0 -> Head(tensor, dims[1], dims[2], dims[3], true)
                    1 -> Head(tensor, dims[3], dims[1], dims[2], false)
                    else -> error("Unsupported Pose head format ${tensor.format}: ${tensor.name}")
                }
            }
        }
    }
}
