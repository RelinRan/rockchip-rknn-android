package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp

/**
 * Provides the `YoloDflDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloDflDecoder` where its surrounding API requires this contract.
 */
internal object YoloDflDecoder {
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
        val grids = outputs.map { GridTensor.create(it, config.labels.size) }
        val candidates = ArrayList<YoloCandidate>()
        val combined = grids.filter { (it.channels - config.labels.size) % 4 == 0 &&
            (it.channels - config.labels.size) / 4 > 1 }
        combined.forEach { decodeGrid(it, it, config, candidates, combined = true) }

        val used = combined.map { it.source }.toSet()
        val regressions = grids.filter { it.source !in used && it.channels % 4 == 0 && it.channels / 4 > 1 }
        regressions.forEach { regression ->
            val scores = grids.filter {
                it.source !in used && it.source != regression.source &&
                    it.channels == config.labels.size && it.width == regression.width && it.height == regression.height
            }
            require(scores.size == 1) {
                "DFL head ${regression.source.name} requires exactly one class tensor on " +
                    "${regression.width}x${regression.height}, found ${scores.map { it.source.name }}"
            }
            decodeGrid(regression, scores.single(), config, candidates, combined = false)
        }
        require(candidates.isNotEmpty() || combined.isNotEmpty() || regressions.isNotEmpty()) {
            "Supported YOLO DFL heads not found: ${outputs.joinToString { it.name + it.dims.contentToString() }}"
        }
        return YoloPostprocessor.process(candidates, image, config, applyNms = true)
    }

    private fun decodeGrid(
        regression: GridTensor,
        scores: GridTensor,
        config: RknnModelConfig,
        candidates: MutableList<YoloCandidate>,
        combined: Boolean,
    ) {
        val regressionChannels = if (combined) regression.channels - config.labels.size else regression.channels
        val regMax = regressionChannels / 4
        val scoreOffset = if (combined) regressionChannels else 0
        val strideX = config.inputWidth.toFloat() / regression.width
        val strideY = config.inputHeight.toFloat() / regression.height
        repeat(regression.height) { y ->
            repeat(regression.width) { x ->
                val categories = config.labels.indices.map { candidateClass ->
                    candidateClass to scores[x, y, scoreOffset + candidateClass]
                }.filter { it.second.isFinite() && it.second >= config.scoreThreshold }
                    .sortedByDescending { it.second }
                if (categories.isEmpty()) return@repeat
                val primary = categories.first()
                val left = expectation(regression, x, y, 0, regMax) * strideX
                val top = expectation(regression, x, y, 1, regMax) * strideY
                val right = expectation(regression, x, y, 2, regMax) * strideX
                val bottom = expectation(regression, x, y, 3, regMax) * strideY
                val centerX = (x + 0.5f) * strideX
                val centerY = (y + 0.5f) * strideY
                candidates += YoloCandidate(
                    centerX - left, centerY - top, centerX + right, centerY + bottom,
                    primary.first, primary.second,
                    categoryScores = if (config.multiLabel) categories else listOf(primary),
                )
            }
        }
    }

    private fun expectation(grid: GridTensor, x: Int, y: Int, side: Int, regMax: Int): Float {
        val offset = side * regMax
        var maximum = Float.NEGATIVE_INFINITY
        repeat(regMax) { maximum = maxOf(maximum, grid[x, y, offset + it]) }
        var sum = 0.0
        var weighted = 0.0
        repeat(regMax) { bin ->
            val probability = exp((grid[x, y, offset + bin] - maximum).toDouble())
            sum += probability
            weighted += probability * bin
        }
        return (weighted / sum).toFloat()
    }

    private class GridTensor private constructor(
        val source: NativeTensorOutput,
        val width: Int,
        val height: Int,
        val channels: Int,
        private val channelFirst: Boolean,
    ) {
        operator fun get(x: Int, y: Int, channel: Int): Float {
            val index = if (channelFirst) {
                channel * height * width + y * width + x
            } else {
                (y * width + x) * channels + channel
            }
            return source.data[index]
        }

        companion object {
            /**
             * Executes `create` for the RKNN runtime contract.
             * @param tensor Native tensor containing model output values and dimensions.
             * @param classes Value supplied for `classes`.
             */
            fun create(tensor: NativeTensorOutput, classes: Int): GridTensor {
                val dims = tensor.dims.toList()
                require(dims.size == 4 && dims[0] == 1) {
                    "DFL tensor ${tensor.name} must be rank-4 with batch 1, got ${dims}"
                }
                val firstLooksLikeChannels = isChannels(dims[1], classes)
                val lastLooksLikeChannels = isChannels(dims[3], classes)
                val channelFirst = when {
                    firstLooksLikeChannels.xor(lastLooksLikeChannels) -> firstLooksLikeChannels
                    tensor.format == 0 -> true
                    tensor.format == 1 -> false
                    else -> error("DFL tensor ${tensor.name} layout is ambiguous: $dims, format=${tensor.format}")
                }
                val grid = if (channelFirst) {
                    GridTensor(tensor, dims[3], dims[2], dims[1], true)
                } else {
                    GridTensor(tensor, dims[2], dims[1], dims[3], false)
                }
                require(grid.width * grid.height * grid.channels == tensor.data.size) {
                    "DFL tensor ${tensor.name} data length mismatch: $dims"
                }
                return grid
            }

            private fun isChannels(value: Int, classes: Int): Boolean =
                value == classes || (value % 4 == 0 && value / 4 > 1) ||
                    ((value - classes) % 4 == 0 && (value - classes) / 4 > 1)
        }
    }
}
