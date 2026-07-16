package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnRuntime
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult

/** 为多模型 API 提供可替换、可测试的运行时能力。 */
internal interface ModelRuntime {
    fun initialize(context: Context, options: RknnOptions): RknnState = RknnState.READY
    fun registerModel(config: RknnModelConfig): Boolean
    fun unregisterModel(modelId: String)
    fun release() = Unit
    fun detectObjects(modelId: String, bitmap: Bitmap): RknnObjectDetectionResult =
        error("Detection is not configured for this runtime")
}

/** 基于 [RknnRuntime] 的默认多模型运行时实现。 */
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
