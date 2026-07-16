package androidx.runtime.rknn

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import androidx.runtime.rknn.internal.ImagePreprocessor
import androidx.runtime.rknn.internal.NoopNativeBridge
import androidx.runtime.rknn.internal.NativeTensorOutput
import androidx.runtime.rknn.internal.RknnNativeBridge
import androidx.runtime.rknn.internal.RknnLogFormat
import androidx.runtime.rknn.internal.RknnEnvironmentProbe
import androidx.runtime.rknn.internal.RockchipRknnNativeBridge
import androidx.runtime.rknn.internal.RockchipSessionManager
import androidx.runtime.rknn.decoder.RknnDecoderRegistry
import androidx.runtime.rknn.decoder.RknnDecodedResult
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.data.RknnImageClassificationResult
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.data.RknnHandLandmarkResult
import androidx.runtime.rknn.data.RknnPoseLandmarkResult
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * RKNN 模型运行时，负责环境探测、模型注册、Session 生命周期和推理调度。
 * 每个实例独立维护模型及 Session，调用结束后应执行 [close]。
 */
class RknnRuntime internal constructor(
    private val bridgeFactory: (RknnBackend) -> RknnNativeBridge,
) : AutoCloseable {

    constructor() : this({ RockchipRknnNativeBridge() })

    private val TAG = "RknnSDK"

    @Volatile
    private var appContext: Context? = null

    @Volatile
    private var options: RknnOptions = RknnOptions()

    @Volatile
    var state: RknnState = RknnState.UNINITIALIZED
        private set

    private val models = ConcurrentHashMap<String, RknnModelConfig>()
    private var nativeBridge: RknnNativeBridge = NoopNativeBridge()
    private var sessionManager: RockchipSessionManager = RockchipSessionManager(nativeBridge)
    private val lifecycleLock = Any()

    fun initialize(context: Context, options: RknnOptions = RknnOptions()): RknnState {
        if (options.debug) {
            Log.i(
                TAG,
                "initialize start backend=${options.backend} modelRoot=${options.modelRoot?.absolutePath} " +
                    "project=${options.project} modelDir=${options.modelDir}",
            )
        }
        appContext = context.applicationContext
        this.options = options
        try {
            nativeBridge = bridgeFactory(options.backend)
        } catch (error: Throwable) {
            Log.e(TAG, "initialize bridge creation failed backend=${options.backend}", error)
            throw error
        }
        if (!nativeBridge.isAvailable()) {
            Log.w(TAG, "initialize runtime unavailable backend=${options.backend}; using noop bridge")
            nativeBridge = NoopNativeBridge()
        }
        Log.i(
            TAG,
            RknnLogFormat.environment(
                chip = RknnEnvironmentProbe.chip(),
                driverVersion = RknnEnvironmentProbe.driverVersion(),
            ),
        )
        sessionManager.closeAll()
        sessionManager = RockchipSessionManager(nativeBridge, options.debug)
        val deviceInfo = deviceInfo()
        state = if (deviceInfo.supported) RknnState.READY else RknnState.UNSUPPORTED
        if (options.debug) {
            Log.i(TAG, "initialize state=$state backend=${options.backend} reasons=${deviceInfo.reasons}")
        }
        if (state != RknnState.READY) {
            Log.w(TAG, "initialize unsupported backend=${options.backend} reasons=${deviceInfo.reasons}")
        }
        return state
    }

    fun deviceInfo(): RknnDeviceInfo {
        val reasons = mutableListOf<String>()
        val markerSupported = RockchipProbe.isRockchipNpuAvailable(reasons)
        val runtimeSupported = nativeBridge.isAvailable().also {
            if (!it) reasons += "rknn runtime unavailable"
        }
        val supported = markerSupported && runtimeSupported
        return RknnDeviceInfo(
            backend = options.backend,
            supported = supported,
            reasons = reasons
        )
    }

    fun registerModel(config: RknnModelConfig): Boolean {
        val context = appContext
        if (context == null) {
            Log.w(TAG, "registerModel rejected id=${config.id}: SDK not initialized")
            return false
        }
        val modelFile = File(resolveModelDir(), config.fileName)
        if (!modelFile.exists()) {
            Log.w(TAG, "registerModel missing id=${config.id} file=${modelFile.absolutePath}")
            return false
        }
        synchronized(lifecycleLock) {
            sessionManager.closeSession(config.id)
            models[config.id] = config
            if (options.debug) {
                Log.i(
                    TAG,
                    "registerModel id=${config.id} modelType=${config.type.name} " +
                        "file=${modelFile.absolutePath} fileBytes=${modelFile.length()} " +
                        "input=${config.inputWidth}x${config.inputHeight} " +
                        "inputType=${config.inputType.name} inputLayout=${config.inputLayout.name} " +
                        "normalizationMean=${config.normalization.mean.contentToString()} " +
                        "normalizationStd=${config.normalization.std.contentToString()} " +
                        "decoder=${config.decoderType.name} mediaPipeModel=${config.mediaPipeModel.name} " +
                        "classifierModel=${config.classifierModel?.name ?: "NONE"} " +
                        "classifierScoreType=${config.classifierScoreType.name} " +
                        "poseLandmarkModel=${config.poseLandmarkModel?.name ?: "NONE"} " +
                        "handLandmarkModel=${config.handLandmarkModel?.name ?: "NONE"} " +
                        "labels=${config.labels.size} threshold=${config.scoreThreshold} maxResults=${config.maxResults}",
                )
            }
            val handle = sessionManager.ensureSession(config, modelFile)
            if (handle == 0L) {
                Log.e(TAG, "registerModel session warmup failed id=${config.id} file=${modelFile.absolutePath}")
                return false
            }
            if (options.debug) Log.i(TAG, "registerModel session ready id=${config.id}")
        }
        return true
    }

    fun retryModel(modelId: String): Boolean {
        synchronized(lifecycleLock) {
            if (state != RknnState.READY) return false
            val config = models[modelId] ?: return false
            val modelFile = File(resolveModelDir(), config.fileName)
            if (!modelFile.exists()) return false
            sessionManager.resetFailure(modelId)
            val handle = sessionManager.ensureSession(config, modelFile)
            if (handle == 0L) {
                Log.e(TAG, "retryModel failed id=$modelId file=${modelFile.absolutePath}")
                return false
            }
            if (options.debug) Log.i(TAG, "retryModel session ready id=$modelId")
            return true
        }
    }

    fun registeredModels(): List<RknnModelConfig> = models.values.sortedBy { it.id }

    fun unregisterModel(modelId: String) {
        synchronized(lifecycleLock) {
            val removed = models.remove(modelId)
            sessionManager.closeSession(modelId)
            if (removed == null) {
                Log.w(TAG, "unregisterModel ignored id=$modelId: model not registered")
            } else if (options.debug) {
                Log.i(TAG, "unregisterModel id=$modelId")
            } else {

            }
        }
    }

    fun clearModels() {
        synchronized(lifecycleLock) {
            val count = models.size
            models.clear()
            sessionManager.closeAll()
            if (options.debug) Log.i(TAG, "clearModels count=$count")
        }
    }

    fun run(modelId: String, bitmap: Bitmap, extras: Map<String, Any?> = emptyMap()): RknnInferenceResult {
        val startedAt = SystemClock.elapsedRealtime()
        if (options.debug) {
            Log.d(TAG, "run start model=$modelId image=${bitmap.width}x${bitmap.height} extras=${extras.keys.sorted()}")
        }
        val result = runInternal(modelId, bitmap, extras)
        if (options.debug) {
            Log.d(TAG, RknnLogFormat.completion(modelId, result.success,
                SystemClock.elapsedRealtime() - startedAt, result.raw?.tensors?.size ?: 0, result.message))
        }
        return result
    }

    private fun runInternal(modelId: String, bitmap: Bitmap, extras: Map<String, Any?>): RknnInferenceResult {
        val start = SystemClock.elapsedRealtime()
        val backend = options.backend
        val config = models[modelId]
        if (state != RknnState.READY) {
            Log.w(TAG, "run rejected model=$modelId state=$state")
            return RknnInferenceResult(
                success = false,
                backend = backend,
                modelId = modelId,
                durationMs = SystemClock.elapsedRealtime() - start,
                message = "NPU not ready: $state"
            )
        }
        if (config == null) {
            Log.w(TAG, "run rejected model=$modelId: model not registered")
            return RknnInferenceResult(
                success = false,
                backend = backend,
                modelId = modelId,
                durationMs = SystemClock.elapsedRealtime() - start,
                message = "Model not registered: $modelId"
            )
        }
        val context = appContext
        if (context == null) {
            Log.w(TAG, "run rejected model=$modelId: context not initialized")
            return RknnInferenceResult(
                success = false,
                backend = backend,
                modelId = modelId,
                durationMs = SystemClock.elapsedRealtime() - start,
                message = "Context not initialized"
            )
        }
        val modelFile = File(resolveModelDir(), config.fileName)
        if (!modelFile.exists()) {
            Log.w(TAG, "run missing model=$modelId file=${modelFile.absolutePath}")
            return RknnInferenceResult(
                success = false,
                backend = backend,
                modelId = modelId,
                durationMs = SystemClock.elapsedRealtime() - start,
                message = "Model file missing: ${modelFile.absolutePath}"
            )
        }
        val result = sessionManager.withSession(config, modelFile) { handle ->
            val input = ImagePreprocessor.toRgbImage(bitmap, config.inputWidth, config.inputHeight)
            nativeBridge.run(handle, input, config, extras + mapOf("modelId" to modelId))
                .let { it.copy(raw = it.raw?.copy(inputImage = input)) }
        }
        if (result == null) {
            Log.e(TAG, "run session open failed model=$modelId file=${modelFile.absolutePath}")
            return RknnInferenceResult(
                success = false,
                backend = backend,
                modelId = modelId,
                durationMs = SystemClock.elapsedRealtime() - start,
                message = "Failed to open RKNN session"
            )
        }
        val rawOutputs = result.raw?.tensors.orEmpty()
        if (!result.success) {
            Log.e(TAG, "run native failure model=$modelId durationMs=${result.durationMs} message=${result.message}")
        } else if (options.debug) {
            Log.d(
                TAG,
                "run native model=$modelId durationMs=${result.durationMs} api=${result.raw?.apiVersion} " +
                    "driver=${result.raw?.driverVersion} ${RknnLogFormat.tensors(rawOutputs)}",
            )
        }
        return result
    }

    fun detectObjects(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnObjectDetectionResult {
        val result = run(modelId, bitmap, extras)
        if (!result.success) {
            Log.w(TAG, "detectObjects aborted model=$modelId message=${result.message}")
            return RknnObjectDetectionResult(false, result.backend, modelId, emptyList(), result.durationMs, result.message)
        }
        val config = models[modelId]
        if (config == null) {
            Log.w(TAG, "detectObjects model removed during inference model=$modelId")
            return RknnObjectDetectionResult(false, result.backend, modelId, emptyList(), result.durationMs, "Model unregistered during inference")
        }
        val outputs = result.raw?.tensors.orEmpty()
        if (options.debug) {
            Log.d(TAG, "detectObjects selected model=$modelId ${RknnLogFormat.tensors(outputs)}")
        }
        val input = result.raw?.inputImage
        if (input == null) {
            Log.e(TAG, "detectObjects input metadata missing model=$modelId")
            return RknnObjectDetectionResult(false, result.backend, modelId, emptyList(), result.durationMs, "Input transform metadata missing")
        }
        val decoded = runCatching {
            RknnDecoderRegistry.decode(config.decoderType, outputs, input, config)
        }
            .getOrElse { error ->
                Log.e(TAG, "detectObjects decode failed model=$modelId", error)
                return RknnObjectDetectionResult(false, result.backend, modelId, emptyList(), result.durationMs, error.message)
            }
        val detections = (decoded as? RknnDecodedResult.Detection)?.detections
            ?: return RknnObjectDetectionResult(
                false, result.backend, modelId, emptyList(), result.durationMs,
                "Decoder did not return an object detection result",
            )
        if (options.debug) {
            Log.d(TAG, "detectObjects complete model=$modelId detections=${detections.size} durationMs=${result.durationMs}")
        }
        return RknnObjectDetectionResult(
            success = result.success,
            backend = result.backend,
            modelId = result.modelId,
            detections = detections,
            durationMs = result.durationMs,
            message = result.message
        )
    }

    /** 执行图像分类，并返回经过概率处理、阈值过滤和 Top-K 排序的类别。 */
    fun classifyImage(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnImageClassificationResult {
        val result = run(modelId, bitmap, extras)
        if (!result.success) {
            return RknnImageClassificationResult(
                false, result.backend, modelId, emptyList(), result.durationMs, result.message,
            )
        }
        val config = models[modelId]
            ?: return RknnImageClassificationResult(
                false, result.backend, modelId, emptyList(), result.durationMs,
                "Model unregistered during inference",
            )
        if (config.type != RknnModelType.IMAGE_CLASSIFIER) {
            return RknnImageClassificationResult(
                false, result.backend, modelId, emptyList(), result.durationMs,
                "Model is not an image classifier: $modelId",
            )
        }
        val input = result.raw?.inputImage
            ?: return RknnImageClassificationResult(
                false, result.backend, modelId, emptyList(), result.durationMs,
                "Input transform metadata missing",
            )
        val decoded = runCatching {
            RknnDecoderRegistry.decode(config.decoderType, result.raw?.tensors.orEmpty(), input, config)
        }
            .getOrElse { error ->
                Log.e(TAG, "classifyImage decode failed model=$modelId", error)
                return RknnImageClassificationResult(
                    false, result.backend, modelId, emptyList(), result.durationMs, error.message,
                )
            }
        val categories = (decoded as? RknnDecodedResult.Classification)?.categories
            ?: return RknnImageClassificationResult(
                false, result.backend, modelId, emptyList(), result.durationMs,
                "Decoder did not return an image classification result",
            )
        return RknnImageClassificationResult(
            true, result.backend, modelId, categories, result.durationMs, result.message,
        )
    }

    /** 解码已按人体 ROI 裁剪后的 Pose Landmark 模型输出。 */
    fun detectPoseLandmarks(modelId: String, bitmap: Bitmap): RknnPoseLandmarkResult {
        val result = run(modelId, bitmap)
        if (!result.success) return failedPoseResult(result, modelId, result.message)
        val config = models[modelId]
            ?: return failedPoseResult(result, modelId, "Model unregistered during inference")
        if (config.type != RknnModelType.POSE_LANDMARKER) {
            return failedPoseResult(result, modelId, "Model is not a pose landmark model: $modelId")
        }
        val input = result.raw?.inputImage
            ?: return failedPoseResult(result, modelId, "Input transform metadata missing")
        val decodedResult = runCatching {
            RknnDecoderRegistry.decode(config.decoderType, result.raw?.tensors.orEmpty(), input, config)
        }
            .getOrElse { error ->
                Log.e(TAG, "detectPoseLandmarks decode failed model=$modelId", error)
                return failedPoseResult(result, modelId, error.message)
            }
        val decoded = (decodedResult as? RknnDecodedResult.PoseLandmark)?.pose
            ?: return failedPoseResult(result, modelId, "Decoder did not return a pose landmark result")
        return RknnPoseLandmarkResult(
            success = true,
            backend = result.backend,
            modelId = modelId,
            landmarks = decoded.landmarks,
            worldLandmarks = decoded.worldLandmarks,
            posePresence = decoded.posePresence,
            segmentationMask = decoded.segmentationMask,
            segmentationMaskWidth = decoded.segmentationMaskWidth,
            segmentationMaskHeight = decoded.segmentationMaskHeight,
            durationMs = result.durationMs,
            message = result.message,
        )
    }

    /** 解码已按手部 ROI 裁剪后的 Hand Landmark 模型输出。 */
    fun detectHandLandmarks(modelId: String, bitmap: Bitmap): RknnHandLandmarkResult {
        val result = run(modelId, bitmap)
        if (!result.success) return failedHandResult(result, modelId, result.message)
        val config = models[modelId]
            ?: return failedHandResult(result, modelId, "Model unregistered during inference")
        if (config.type != RknnModelType.HAND_LANDMARKER) {
            return failedHandResult(result, modelId, "Model is not a hand landmark model: $modelId")
        }
        val input = result.raw?.inputImage
            ?: return failedHandResult(result, modelId, "Input transform metadata missing")
        val decodedResult = runCatching {
            RknnDecoderRegistry.decode(config.decoderType, result.raw?.tensors.orEmpty(), input, config)
        }
            .getOrElse { error ->
                Log.e(TAG, "detectHandLandmarks decode failed model=$modelId", error)
                return failedHandResult(result, modelId, error.message)
            }
        val decoded = (decodedResult as? RknnDecodedResult.HandLandmark)?.hand
            ?: return failedHandResult(result, modelId, "Decoder did not return a hand landmark result")
        return RknnHandLandmarkResult(
            success = true,
            backend = result.backend,
            modelId = modelId,
            landmarks = decoded.landmarks,
            worldLandmarks = decoded.worldLandmarks,
            handPresence = decoded.handPresence,
            handedness = decoded.handedness,
            durationMs = result.durationMs,
            message = result.message,
        )
    }

    private fun failedPoseResult(
        result: RknnInferenceResult,
        modelId: String,
        message: String?,
    ) = RknnPoseLandmarkResult(
        false, result.backend, modelId, emptyList(), emptyList(), 0f,
        durationMs = result.durationMs, message = message,
    )

    private fun failedHandResult(
        result: RknnInferenceResult,
        modelId: String,
        message: String?,
    ) = RknnHandLandmarkResult(
        false, result.backend, modelId, emptyList(), emptyList(), 0f,
        RknnCategory(-1, "Unknown", 0f), result.durationMs, message,
    )

    private fun resolveModelDir(): File {
        return options.resolveModelRoot(Environment.getExternalStorageDirectory())
    }

    override fun close() {
        clearModels()
        appContext = null
        state = RknnState.UNINITIALIZED
    }
}
