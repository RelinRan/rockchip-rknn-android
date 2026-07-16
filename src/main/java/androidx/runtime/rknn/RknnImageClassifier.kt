package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnImageClassificationResult

/** 保留进程级默认运行时调用方式的图像分类入口。 */
object RknnImageClassifier {
    fun classify(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnImageClassificationResult = RknnApi.classifyImage(modelId, bitmap, extras)
}
