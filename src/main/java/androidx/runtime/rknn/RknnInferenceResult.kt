package androidx.runtime.rknn

/** 单次底层推理结果，包含原始张量和运行耗时。 */
data class RknnInferenceResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val durationMs: Long,
    val message: String? = null,
    val raw: RknnRawInference? = null,
    @Deprecated("Use raw")
    val outputs: Map<String, Any?> = emptyMap(),
)
