package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnRuntime
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult

/**
 * Provides the `ModelRuntime` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ModelRuntime` where its surrounding API requires this contract.
 */
internal interface ModelRuntime {
    /**
     * Executes `initialize` for the RKNN runtime contract.
     * @param context Android context used to access storage and native resources.
     * @param options Runtime initialization options.
     */
    fun initialize(context: Context, options: RknnOptions): RknnState = RknnState.READY
    /**
     * Executes `registerModel` for the RKNN runtime contract.
     * @param config Model or runtime configuration used by the operation.
     */
    fun registerModel(config: RknnModelConfig): Boolean
    /**
     * Executes `unregisterModel` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     */
    fun unregisterModel(modelId: String)
    /**
     * Executes `release` for the RKNN runtime contract.
     */
    fun release() = Unit
    /**
     * Executes `detectObjects` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     */
    fun detectObjects(modelId: String, bitmap: Bitmap): RknnObjectDetectionResult =
        error("Detection is not configured for this runtime")
}

/**
 * Provides the `DefaultModelRuntime` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `DefaultModelRuntime` where its surrounding API requires this contract.
 */
internal class DefaultModelRuntime(
    private val runtime: RknnRuntime = RknnRuntime(),
) : ModelRuntime {
    override fun initialize(context: Context, options: RknnOptions): RknnState =
        runtime.initialize(context, options)

    override fun registerModel(config: RknnModelConfig): Boolean = runtime.registerModel(config)

    override fun unregisterModel(modelId: String) = runtime.unregisterModel(modelId)

    override fun detectObjects(modelId: String, bitmap: Bitmap): RknnObjectDetectionResult =
        runtime.detectObjects(modelId, bitmap)

    override fun release() = runtime.close()
}
