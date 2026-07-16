package androidx.runtime.rknn

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import androidx.runtime.rknn.data.RknnImageClassificationResult
import androidx.runtime.rknn.data.RknnHandLandmarkResult
import androidx.runtime.rknn.data.RknnPoseLandmarkResult

/** 基于进程级默认运行时的兼容入口，新代码优先直接持有 [RknnRuntime]。 */
object RknnApi {
    private val runtime = RknnRuntime()

    val state: RknnState get() = runtime.state

    fun initialize(context: Context, options: RknnOptions = RknnOptions()): RknnState =
        runtime.initialize(context, options)

    fun deviceInfo(): RknnDeviceInfo = runtime.deviceInfo()

    fun registerModel(config: RknnModelConfig): Boolean = runtime.registerModel(config)

    fun retryModel(modelId: String): Boolean = runtime.retryModel(modelId)

    fun registeredModels(): List<RknnModelConfig> = runtime.registeredModels()

    fun unregisterModel(modelId: String) = runtime.unregisterModel(modelId)

    fun clearModels() = runtime.clearModels()

    fun run(modelId: String, bitmap: Bitmap, extras: Map<String, Any?> = emptyMap()): RknnInferenceResult = runtime.run(modelId, bitmap, extras)

    fun detectObjects(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnObjectDetectionResult = runtime.detectObjects(modelId, bitmap, extras)

    fun classifyImage(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnImageClassificationResult = runtime.classifyImage(modelId, bitmap, extras)

    fun detectPoseLandmarks(modelId: String, bitmap: Bitmap): RknnPoseLandmarkResult =
        runtime.detectPoseLandmarks(modelId, bitmap)

    fun detectHandLandmarks(modelId: String, bitmap: Bitmap): RknnHandLandmarkResult =
        runtime.detectHandLandmarks(modelId, bitmap)
}
