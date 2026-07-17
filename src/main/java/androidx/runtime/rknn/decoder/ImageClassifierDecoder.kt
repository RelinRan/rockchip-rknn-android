package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.internal.NativeTensorOutput
import kotlin.math.exp

/**
 * Provides the `ImageClassifierDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ImageClassifierDecoder` where its surrounding API requires this contract.
 */
internal object ImageClassifierDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param output Native output tensor to decode.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(output: NativeTensorOutput, config: RknnModelConfig): List<RknnCategory> {
        require(config.labels.isNotEmpty()) { "Image classifier labels must not be empty" }
        require(output.data.size == config.labels.size) {
            "Classifier output contains ${output.data.size} values, but ${config.labels.size} labels were configured"
        }
        require(output.data.all(Float::isFinite)) { "Classifier output contains non-finite values" }

        val probabilities = when (config.classifierScoreType) {
            ClassificationScoreType.PROBABILITIES -> output.data
            ClassificationScoreType.LOGITS -> softmax(output.data)
            ClassificationScoreType.AUTO -> if (isProbabilityDistribution(output.data)) {
                output.data
            } else {
                softmax(output.data)
            }
        }
        return probabilities.indices
            .asSequence()
            .map { index -> RknnCategory(index, config.labels[index], probabilities[index]) }
            .filter { it.score >= config.scoreThreshold }
            .sortedByDescending(RknnCategory::score)
            .take(config.maxResults)
            .toList()
    }

    private fun isProbabilityDistribution(values: FloatArray): Boolean {
        if (values.any { it !in 0f..1f }) return false
        return values.sum() in 0.99f..1.01f
    }

    private fun softmax(values: FloatArray): FloatArray {
        val maximum = values.max()
        val exponentials = DoubleArray(values.size) { index ->
            exp((values[index] - maximum).toDouble())
        }
        val sum = exponentials.sum().coerceAtLeast(1e-300)
        return FloatArray(values.size) { index -> (exponentials[index] / sum).toFloat() }
    }
}
