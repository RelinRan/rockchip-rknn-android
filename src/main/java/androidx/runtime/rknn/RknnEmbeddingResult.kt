package androidx.runtime.rknn

import kotlin.math.sqrt

/** L2-normalized feature vector returned by a ReID or other embedding model. */
data class RknnEmbeddingResult(
    override val success: Boolean,
    override val backend: RknnBackend,
    override val modelId: String,
    val embedding: FloatArray,
    override val durationMs: Long,
    override val message: String? = null,
    override val resultSequence: Long = 0,
) : ModelResult {
    override fun withResultSequence(sequence: Long): ModelResult = copy(resultSequence = sequence)

    companion object {
        fun normalized(
            backend: RknnBackend,
            modelId: String,
            values: FloatArray,
            expectedSize: Int,
            durationMs: Long,
        ): RknnEmbeddingResult {
            if (values.size != expectedSize) {
                return failure(backend, modelId, durationMs, "Expected $expectedSize embedding values, got ${values.size}")
            }
            if (values.any { !it.isFinite() }) {
                return failure(backend, modelId, durationMs, "Embedding contains non-finite values")
            }
            var squaredNorm = 0.0
            values.forEach { squaredNorm += it.toDouble() * it.toDouble() }
            val norm = sqrt(squaredNorm).toFloat()
            if (!norm.isFinite() || norm <= 1e-12f) {
                return failure(backend, modelId, durationMs, "Embedding norm is zero or invalid")
            }
            return RknnEmbeddingResult(
                success = true,
                backend = backend,
                modelId = modelId,
                embedding = FloatArray(values.size) { values[it] / norm },
                durationMs = durationMs,
            )
        }

        fun failure(
            backend: RknnBackend,
            modelId: String,
            durationMs: Long,
            message: String,
        ) = RknnEmbeddingResult(false, backend, modelId, FloatArray(0), durationMs, message)
    }
}
