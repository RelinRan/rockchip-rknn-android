package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnObjectDetectionResult

/**
 * Provides the `RknnObjectDetector` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnObjectDetector` where its surrounding API requires this contract.
 */
object RknnObjectDetector {

    /**
     * Executes `detect` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param extras Optional backend-specific inference values.
     */
    fun detect(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnObjectDetectionResult {
        return RknnApi.detectObjects(modelId, bitmap, extras)
    }
}
