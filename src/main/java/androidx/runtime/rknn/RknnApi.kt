package androidx.runtime.rknn

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import androidx.runtime.rknn.data.RknnImageClassificationResult
import androidx.runtime.rknn.data.RknnHandLandmarkResult
import androidx.runtime.rknn.data.RknnPoseLandmarkResult

/**
 * Provides the `RknnApi` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnApi` where its surrounding API requires this contract.
 */
object RknnApi {
    private val runtime = RknnRuntime()

    val state: RknnState get() = runtime.state

    /**
     * Executes `initialize` for the RKNN runtime contract.
     * @param context Android context used to access storage and native resources.
     * @param options Runtime initialization options.
     */
    fun initialize(context: Context, options: RknnOptions = RknnOptions()): RknnState =
        runtime.initialize(context, options)

    /**
     * Executes `deviceInfo` for the RKNN runtime contract.
     */
    fun deviceInfo(): RknnDeviceInfo = runtime.deviceInfo()

    /**
     * Executes `registerModel` for the RKNN runtime contract.
     * @param config Model or runtime configuration used by the operation.
     */
    fun registerModel(config: RknnModelConfig): Boolean = runtime.registerModel(config)

    /**
     * Executes `retryModel` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     */
    fun retryModel(modelId: String): Boolean = runtime.retryModel(modelId)

    /**
     * Executes `registeredModels` for the RKNN runtime contract.
     */
    fun registeredModels(): List<RknnModelConfig> = runtime.registeredModels()

    /**
     * Executes `unregisterModel` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     */
    fun unregisterModel(modelId: String) = runtime.unregisterModel(modelId)

    /**
     * Executes `clearModels` for the RKNN runtime contract.
     */
    fun clearModels() = runtime.clearModels()

    /**
     * Executes `run` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param extras Optional backend-specific inference values.
     */
    fun run(modelId: String, bitmap: Bitmap, extras: Map<String, Any?> = emptyMap()): RknnInferenceResult = runtime.run(modelId, bitmap, extras)

    /**
     * Executes `detectObjects` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param extras Optional backend-specific inference values.
     */
    fun detectObjects(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnObjectDetectionResult = runtime.detectObjects(modelId, bitmap, extras)

    /**
     * Executes `classifyImage` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param extras Optional backend-specific inference values.
     */
    fun classifyImage(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnImageClassificationResult = runtime.classifyImage(modelId, bitmap, extras)

    /**
     * Executes `detectPoseLandmarks` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     */
    fun detectPoseLandmarks(modelId: String, bitmap: Bitmap): RknnPoseLandmarkResult =
        runtime.detectPoseLandmarks(modelId, bitmap)

    /**
     * Executes `detectHandLandmarks` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     */
    fun detectHandLandmarks(modelId: String, bitmap: Bitmap): RknnHandLandmarkResult =
        runtime.detectHandLandmarks(modelId, bitmap)
}
