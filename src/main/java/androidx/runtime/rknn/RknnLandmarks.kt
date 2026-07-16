package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnHandLandmarkResult
import androidx.runtime.rknn.data.RknnPoseLandmarkResult

/** 使用进程级默认运行时执行 Pose Landmark。 */
object RknnPoseLandmark {
    fun detect(modelId: String, bitmap: Bitmap): RknnPoseLandmarkResult =
        RknnApi.detectPoseLandmarks(modelId, bitmap)
}

/** 使用进程级默认运行时执行 Hand Landmark。 */
object RknnHandLandmark {
    fun detect(modelId: String, bitmap: Bitmap): RknnHandLandmarkResult =
        RknnApi.detectHandLandmarks(modelId, bitmap)
}
