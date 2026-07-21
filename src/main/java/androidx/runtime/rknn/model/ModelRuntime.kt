package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnEmbeddingResult
import androidx.runtime.rknn.RknnRuntime
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult

/** Replaceable runtime contract used by the multi-model API and unit tests. */
internal interface ModelRuntime {
    fun initialize(context: Context, options: RknnOptions): RknnState = RknnState.READY
    fun registerModel(config: RknnModelConfig): Boolean
    fun unregisterModel(modelId: String)
    fun release() = Unit
    fun detectObjects(modelId: String, bitmap: Bitmap): RknnObjectDetectionResult =
        error("Detection is not configured for this runtime")
    fun extractEmbedding(modelId: String, bitmap: Bitmap): RknnEmbeddingResult =
        error("Embedding extraction is not configured for this runtime")
}

/** Default multi-model runtime implementation backed by [RknnRuntime]. */
internal class DefaultModelRuntime(
    private val runtime: RknnRuntime = RknnRuntime(),
) : ModelRuntime {
    override fun initialize(context: Context, options: RknnOptions): RknnState =
        runtime.initialize(context, options)

    override fun registerModel(config: RknnModelConfig): Boolean = runtime.registerModel(config)

    override fun unregisterModel(modelId: String) = runtime.unregisterModel(modelId)

    override fun detectObjects(modelId: String, bitmap: Bitmap): RknnObjectDetectionResult =
        runtime.detectObjects(modelId, bitmap)

    override fun extractEmbedding(modelId: String, bitmap: Bitmap): RknnEmbeddingResult =
        runtime.extractEmbedding(modelId, bitmap)

    override fun release() = runtime.close()
}
