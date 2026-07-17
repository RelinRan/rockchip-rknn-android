package androidx.runtime.rknn

/**
 * Low-level result of one model execution.
 *
 * Call `runtime.run(modelId, bitmap)` and inspect [raw] only when [success] is true.
 *
 * @property success Whether native execution completed successfully.
 * @property backend Backend used for execution.
 * @property modelId Identifier of the executed model.
 * @property durationMs Native execution and transfer duration in milliseconds.
 * @property message Failure or diagnostic message.
 * @property raw Raw tensor outputs and preprocessing metadata.
 * @property outputs Legacy untyped output map; use [raw] instead.
 */
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
