package androidx.runtime.rknn.model

/**
 * Provides the `MultimodalLifecycle` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `MultimodalLifecycle` where its surrounding API requires this contract.
 */
enum class MultimodalLifecycle {
    UNINITIALIZED,
    INITIALIZING,
    READY,
    PARTIALLY_READY,
    ERROR,
    RELEASED,
}

/**
 * Provides the `ModelReadiness` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ModelReadiness` where its surrounding API requires this contract.
 */
data class ModelReadiness(
    val enabled: Boolean = false,
    val ready: Boolean = false,
    val message: String? = null,
)

/**
 * Provides the `MultimodalState` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `MultimodalState` where its surrounding API requires this contract.
 */
data class MultimodalState(
    val lifecycle: MultimodalLifecycle = MultimodalLifecycle.UNINITIALIZED,
    val models: Map<ModelKey, ModelReadiness> = emptyMap(),
    val messages: List<String> = emptyList(),
) {
    /**
     * Executes `readiness` for the RKNN runtime contract.
     * @param key Stable key identifying a configured model.
     */
    fun readiness(key: ModelKey): ModelReadiness = models[key] ?: ModelReadiness()
}
