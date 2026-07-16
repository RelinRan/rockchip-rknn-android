package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import java.io.Closeable
import java.io.File
import java.util.concurrent.ConcurrentHashMap
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
 * 多模型业务入口，分别管理人员追踪、洗手动作和帽子口罩检测模型。
 * 同一模型同时只执行一项异步推理，避免摄像头帧堆积。
 */
class ModelApi internal constructor(
    private val runtime: ModelRuntime,
    private val inferenceDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : Closeable {
    constructor() : this(DefaultModelRuntime())

    private val resultFlows = ConcurrentHashMap<ModelKey, MutableStateFlow<RknnObjectDetectionResult?>>()
    private val _state = MutableStateFlow(MultimodalState())

    val state: StateFlow<MultimodalState> = _state.asStateFlow()

    private val runtimeLock = Any()
    private val activeModels = ConcurrentHashMap<ModelKey, RknnModelConfig>()
    private val busy = ConcurrentHashMap<ModelKey, Mutex>()
    private val modelExecutionLocks = ConcurrentHashMap<ModelKey, Any>()
    @Volatile
    private var executionMode = ModelExecutionMode.SERIAL
    private var scope: CoroutineScope? = null

    fun initialize(context: Context, modelRoot: File, config: ModelConfig): MultimodalState {
        releaseResources()
        val runtimeState = runtime.initialize(
            context.applicationContext,
            RknnOptions(modelRoot = modelRoot, debug = config.debug),
        )
        return initializeModelRoot(modelRoot, config, runtimeState)
    }

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
        executionMode = config.executionMode
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
                labels.isEmpty() -> "${key.value} labels missing or empty"
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

    /** 获取指定模型的结果流；自定义 key 在初始化前也可以订阅。 */
    fun result(key: ModelKey): StateFlow<RknnObjectDetectionResult?> = resultFlow(key).asStateFlow()

    fun detect(key: ModelKey, bitmap: Bitmap): RknnObjectDetectionResult =
        executeDetection(key) { runtime.detectObjects(key.value, bitmap) }

    fun detectAsync(key: ModelKey, bitmap: Bitmap): Boolean =
        executeDetectionAsync(key) { runtime.detectObjects(key.value, bitmap) }

    internal fun executeDetection(
        key: ModelKey,
        operation: () -> RknnObjectDetectionResult,
    ): RknnObjectDetectionResult {
        val lock = when (executionMode) {
            ModelExecutionMode.SERIAL -> runtimeLock
            ModelExecutionMode.PARALLEL -> modelExecutionLocks.computeIfAbsent(key) { Any() }
        }
        return synchronized(lock) { executeDetectionLocked(key, operation) }
    }

    private fun executeDetectionLocked(
        key: ModelKey,
        operation: () -> RknnObjectDetectionResult,
    ): RknnObjectDetectionResult {
        if (!activeModels.containsKey(key)) return unavailableResult(key)
        return runCatching(operation)
            .getOrElse { failedResult(key, "${key.value} detection failed: ${it.message.orEmpty()}") }
            .also { resultFlow(key).value = it }
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

    fun release() {
        releaseResources()
        _state.value = MultimodalState(lifecycle = MultimodalLifecycle.RELEASED)
    }

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
        executionMode = ModelExecutionMode.SERIAL
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

    private fun resultFlow(key: ModelKey): MutableStateFlow<RknnObjectDetectionResult?> =
        resultFlows.computeIfAbsent(key) { MutableStateFlow(null) }

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
