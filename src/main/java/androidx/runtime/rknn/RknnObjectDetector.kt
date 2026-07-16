package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnObjectDetectionResult

/** 保留旧调用方式的目标检测兼容入口。 */
object RknnObjectDetector {

    fun detect(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnObjectDetectionResult {
        return RknnApi.detectObjects(modelId, bitmap, extras)
    }
}
