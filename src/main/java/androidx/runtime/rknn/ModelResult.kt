package androidx.runtime.rknn

/** Common result contract published by the multi-model API. */
interface ModelResult {
    val success: Boolean
    val backend: RknnBackend
    val modelId: String
    val durationMs: Long
    val message: String?
    val resultSequence: Long
    fun withResultSequence(sequence: Long): ModelResult
}

/** Failure result used when inference cannot produce a model-specific result. */
data class ModelFailureResult(
    override val success: Boolean = false,
    override val backend: RknnBackend = RknnBackend.ROCKCHIP_RKNN,
    override val modelId: String,
    override val durationMs: Long = 0,
    override val message: String? = null,
    override val resultSequence: Long = 0,
) : ModelResult {
    override fun withResultSequence(sequence: Long): ModelResult = copy(resultSequence = sequence)
}
