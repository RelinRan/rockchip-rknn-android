package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.ModelFailureResult
import androidx.runtime.rknn.ModelResult
import androidx.runtime.rknn.RknnEmbeddingResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import java.io.Closeable
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex

/**
 * Coroutine-based facade for configuring and running multiple RKNN models.
 *
 * Each model accepts at most one asynchronous request at a time, preventing camera frames from
 * accumulating while inference is busy. Observe [state] and [result] from Kotlin or Compose.
 *
 * Example:
 * ```kotlin
 * val api = ModelApi()
 * api.initialize(context, modelRoot, ModelConfig(mapOf(ModelKey.ACTION to actionModel)))
 * val accepted = api.detectAsync(ModelKey.ACTION, bitmap)
 * val results = api.result(ModelKey.ACTION)
 * api.close()
 * ```
 */
class ModelApi internal constructor(
    private val runtime: ModelRuntime,
    private val inferenceDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : Closeable {
    constructor() : this(DefaultModelRuntime())

    private val resultFlows = ConcurrentHashMap<ModelKey, MutableStateFlow<ModelResult?>>()
    private val resultSequence = AtomicLong(0L)
    private val _state = MutableStateFlow(MultimodalState())

    /** Current initialization and per-model readiness state. */
    val state: StateFlow<MultimodalState> = _state.asStateFlow()

    private val runtimeLock = Any()
    private val activeModels = ConcurrentHashMap<ModelKey, RknnModelConfig>()
    private val busy = ConcurrentHashMap<ModelKey, Mutex>()
    private val modelExecutionLocks = ConcurrentHashMap<ModelKey, Any>()
    @Volatile
    private var runMode = RunMode.SERIAL
    private var scope: CoroutineScope? = null

    /**
     * Initializes the runtime and registers enabled models from an explicit directory.
     *
     * Calling this method again releases all existing model sessions first.
     *
     * @param context Android context used to load native libraries and inspect the device.
     * @param modelRoot Directory containing RKNN model and label files.
     * @param config Models, diagnostics, and execution policy to apply.
     * @return Initialization snapshot, including readiness for every configured model.
     */
    fun initialize(context: Context, modelRoot: File, config: ModelConfig): MultimodalState {
        releaseResources()
        val runtimeState = runtime.initialize(
            context.applicationContext,
            RknnOptions(modelRoot = modelRoot, debug = config.debug),
        )
        return initializeModelRoot(modelRoot, config, runtimeState)
    }

    /**
     * Initializes models below the shared external-storage directory.
     *
     * @param context Android context used by the RKNN runtime.
     * @param project Project directory below external storage.
     * @param modelDir Model directory below [project].
     * @param config Models and execution policy to apply.
     * @return Initialization snapshot for all configured models.
     */
    fun initialize(
        context: Context,
        project: String,
        modelDir: String,
        config: ModelConfig,
    ): MultimodalState {
        val root = File(Environment.getExternalStorageDirectory(), "$project/$modelDir")
        return initialize(context, root, config)
    }

    internal fun initializeModelRoot(
        modelRoot: File,
        config: ModelConfig,
        runtimeState: RknnState,
    ): MultimodalState {
        _state.value = MultimodalState(lifecycle = MultimodalLifecycle.INITIALIZING)
        runMode = config.runMode
        scope = CoroutineScope(SupervisorJob() + inferenceDispatcher)

        val modelConfigs = config.models
        modelConfigs.keys.forEach(::resultFlow)
        if (runtimeState != RknnState.READY) {
            val message = "RKNN runtime: $runtimeState"
            return publishInitializationState(
                modelConfigs,
                emptyMap(),
                mapOfAllEnabled(modelConfigs, message),
                MultimodalLifecycle.ERROR,
                listOf(message),
            )
        }

        val readiness = mutableMapOf<ModelKey, ModelReadiness>()
        val messages = mutableListOf<String>()
        modelConfigs.forEach { (key, detector) ->
            if (!detector.enabled) {
                readiness[key] = ModelReadiness(enabled = false, message = "disabled")
                return@forEach
            }
            val modelFile = File(modelRoot, detector.fileName)
            val labels = resolveLabels(modelRoot, detector)
            val error = when {
                !modelFile.isFile -> "${key.value} model missing: ${detector.fileName}"
                requiresLabels(detector.modelType) && labels.isEmpty() ->
                    "${key.value} labels missing or empty"
                else -> null
            }
            if (error != null) {
                readiness[key] = ModelReadiness(enabled = true, message = error)
                messages += error
                return@forEach
            }
            val model = RknnModelConfig(
                id = key.value,
                type = detector.modelType,
                fileName = detector.fileName,
                inputWidth = detector.inputWidth,
                inputHeight = detector.inputHeight,
                labels = labels,
                scoreThreshold = detector.scoreThreshold,
                nmsThreshold = detector.nmsThreshold,
                maxResults = detector.maxResults,
                poseKeyPointCount = detector.poseKeyPointCount,
                multiLabel = detector.multiLabel,
                embeddingSize = detector.embeddingSize,
                inputType = detector.inputType,
                inputLayout = detector.inputLayout,
                normalization = detector.normalization,
                decoderType = detector.decoderType,
                mediaPipeModel = detector.mediaPipeModel,
            )
            if (runtime.registerModel(model)) {
                activeModels[key] = model
                readiness[key] = ModelReadiness(enabled = true, ready = true)
            } else {
                val registrationError = "${key.value} model registration failed"
                readiness[key] = ModelReadiness(enabled = true, message = registrationError)
                messages += registrationError
            }
        }

        val enabledCount = modelConfigs.values.count { it.enabled }
        val readyCount = readiness.values.count { it.ready }
        val lifecycle = when {
            enabledCount > 0 && readyCount == enabledCount -> MultimodalLifecycle.READY
            readyCount > 0 -> MultimodalLifecycle.PARTIALLY_READY
            else -> MultimodalLifecycle.ERROR
        }
        if (enabledCount == 0) messages += "No models enabled"
        return publishInitializationState(modelConfigs, readiness, emptyMap(), lifecycle, messages)
    }

    /**
     * Returns the latest result stream for [key]. Custom keys may be observed before initialization.
     *
     * @param key Model whose results should be observed.
     * @return Read-only state flow whose initial value is `null`.
     */
    fun result(key: ModelKey): StateFlow<ModelResult?> = resultFlow(key).asStateFlow()

    /**
     * Runs synchronous inference and returns the result type declared by the model configuration.
     *
     * @param key Registered model identifier.
     * @param bitmap Source image. Ownership remains with the caller.
     * @return Detection, embedding, or failure result with `success == false` on failure.
     */
    fun detect(key: ModelKey, bitmap: Bitmap): ModelResult =
        execute(key) { runConfiguredModel(key, bitmap) }

    /**
     * Schedules model inference without blocking the caller.
     *
     * @param key Registered model identifier.
     * @param bitmap Source image. Keep it valid until the result is published.
     * @return `true` when accepted, or `false` when unavailable or already busy.
     */
    fun detectAsync(key: ModelKey, bitmap: Bitmap): Boolean =
        executeAsync(key) { runConfiguredModel(key, bitmap) }

    internal fun execute(
        key: ModelKey,
        operation: () -> ModelResult,
    ): ModelResult {
        val lock = executionLock(key)
        return synchronized(lock) {
            if (!activeModels.containsKey(key)) return@synchronized unavailableModelResult(key)
            val result = runCatching(operation)
                .getOrElse {
                    ModelFailureResult(
                        modelId = key.value,
                        message = "${key.value} inference failed: ${it.message.orEmpty()}",
                    )
                }
                .withResultSequence(resultSequence.incrementAndGet())
            resultFlow(key).value = result
            result
        }
    }

    internal fun executeDetection(
        key: ModelKey,
        operation: () -> RknnObjectDetectionResult,
    ): RknnObjectDetectionResult {
        val lock = executionLock(key)
        return synchronized(lock) { executeDetectionLocked(key, operation) }
    }

    private fun executeDetectionLocked(
        key: ModelKey,
        operation: () -> RknnObjectDetectionResult,
    ): RknnObjectDetectionResult {
        if (!activeModels.containsKey(key)) return unavailableResult(key)
        val result: RknnObjectDetectionResult = runCatching(operation)
            .getOrElse { failedResult(key, "${key.value} detection failed: ${it.message.orEmpty()}") }
            .copy(resultSequence = resultSequence.incrementAndGet())
        resultFlow(key).value = result
        return result
    }

    internal fun executeDetectionAsync(
        key: ModelKey,
        operation: () -> RknnObjectDetectionResult,
    ): Boolean {
        if (!activeModels.containsKey(key)) return false
        val modelMutex = busy.computeIfAbsent(key) { Mutex() }
        if (!modelMutex.tryLock()) return false
        val activeScope = scope
        if (activeScope == null || !activeScope.isActive) {
            modelMutex.unlock()
            return false
        }
        val job = activeScope.launch {
            executeDetection(key, operation)
        }
        job.invokeOnCompletion { modelMutex.unlock() }
        return true
    }

    internal fun executeAsync(
        key: ModelKey,
        operation: () -> ModelResult,
    ): Boolean {
        if (!activeModels.containsKey(key)) return false
        val modelMutex = busy.computeIfAbsent(key) { Mutex() }
        if (!modelMutex.tryLock()) return false
        val activeScope = scope
        if (activeScope == null || !activeScope.isActive) {
            modelMutex.unlock()
            return false
        }
        val job = activeScope.launch { execute(key, operation) }
        job.invokeOnCompletion { modelMutex.unlock() }
        return true
    }

    /** Cancels inference, unregisters every model, and releases native runtime resources. */
    fun release() {
        releaseResources()
        _state.value = MultimodalState(lifecycle = MultimodalLifecycle.RELEASED)
    }

    /** Releases this API and all native model resources. */
    override fun close() = release()

    private fun releaseResources() {
        scope?.cancel()
        scope = null
        synchronized(runtimeLock) {
            activeModels.keys.toList().forEach { runtime.unregisterModel(it.value) }
            activeModels.clear()
        }
        runtime.release()
        busy.clear()
        modelExecutionLocks.clear()
        runMode = RunMode.SERIAL
        resultFlows.values.forEach { it.value = null }
    }

    private fun resolveLabels(root: File, config: DetectorModel): List<String> {
        if (config.labels.isNotEmpty()) {
            return config.labels.map(String::trim).filter(String::isNotEmpty)
        }
        val labelFileName = config.labelFileName ?: return emptyList()
        val labelFile = File(root, labelFileName)
        if (!labelFile.isFile) return emptyList()
        return labelFile.readLines().map(String::trim).filter(String::isNotEmpty)
    }

    private fun unavailableResult(key: ModelKey): RknnObjectDetectionResult {
        val readiness = _state.value.readiness(key)
        val lifecycle = _state.value.lifecycle
        val message = when {
            lifecycle == MultimodalLifecycle.RELEASED -> "ModelApi released"
            lifecycle == MultimodalLifecycle.UNINITIALIZED -> "ModelApi not initialized"
            lifecycle == MultimodalLifecycle.INITIALIZING -> "ModelApi initializing"
            !readiness.enabled -> "${key.value} model disabled"
            else -> readiness.message ?: "${key.value} model not ready"
        }
        return failedResult(key, message)
    }

    private fun failedResult(key: ModelKey, message: String): RknnObjectDetectionResult {
        return RknnObjectDetectionResult(
            success = false,
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = key.value,
            detections = emptyList(),
            durationMs = 0,
            message = message,
        )
    }

    private fun resultFlow(key: ModelKey): MutableStateFlow<ModelResult?> =
        resultFlows.computeIfAbsent(key) { MutableStateFlow(null) }

    private fun runConfiguredModel(key: ModelKey, bitmap: Bitmap): ModelResult =
        when (activeModels[key]?.type) {
            RknnModelType.REID_EMBEDDING -> runtime.extractEmbedding(key.value, bitmap)
            else -> runtime.detectObjects(key.value, bitmap)
        }

    private fun executionLock(key: ModelKey): Any = when (runMode) {
        RunMode.SERIAL -> runtimeLock
        RunMode.PARALLEL -> modelExecutionLocks.computeIfAbsent(key) { Any() }
    }

    private fun unavailableModelResult(key: ModelKey): ModelFailureResult {
        val detectionFailure = unavailableResult(key)
        return ModelFailureResult(
            backend = detectionFailure.backend,
            modelId = detectionFailure.modelId,
            durationMs = detectionFailure.durationMs,
            message = detectionFailure.message,
        )
    }

    private fun requiresLabels(type: RknnModelType): Boolean =
        type != RknnModelType.REID_EMBEDDING && type != RknnModelType.CUSTOM

    private fun mapOfAllEnabled(
        configs: Map<ModelKey, DetectorModel>,
        message: String,
    ): Map<ModelKey, ModelReadiness> = configs.mapValues { (_, config) ->
        ModelReadiness(enabled = config.enabled, message = if (config.enabled) message else "disabled")
    }

    private fun publishInitializationState(
        configs: Map<ModelKey, DetectorModel>,
        readiness: Map<ModelKey, ModelReadiness>,
        overrides: Map<ModelKey, ModelReadiness>,
        lifecycle: MultimodalLifecycle,
        messages: List<String> = overrides.values.mapNotNull { it.message }.distinct(),
    ): MultimodalState {
        val modelStates = configs.mapValues { (key, config) ->
            overrides[key] ?: readiness[key] ?: ModelReadiness(enabled = config.enabled)
        }
        return MultimodalState(
            lifecycle = lifecycle,
            models = modelStates,
            messages = messages,
        ).also { _state.value = it }
    }
}
