package androidx.runtime.rknn.model

/** 多模型 SDK 的整体生命周期。 */
enum class MultimodalLifecycle {
    UNINITIALIZED,
    INITIALIZING,
    READY,
    PARTIALLY_READY,
    ERROR,
    RELEASED,
}

/** 单个业务模型当前是否启用、是否可执行推理。 */
data class ModelReadiness(
    val enabled: Boolean = false,
    val ready: Boolean = false,
    val message: String? = null,
)

/** 三类业务模型的统一状态快照。 */
data class MultimodalState(
    val lifecycle: MultimodalLifecycle = MultimodalLifecycle.UNINITIALIZED,
    val models: Map<ModelKey, ModelReadiness> = emptyMap(),
    val messages: List<String> = emptyList(),
) {
    fun readiness(key: ModelKey): ModelReadiness = models[key] ?: ModelReadiness()
}
