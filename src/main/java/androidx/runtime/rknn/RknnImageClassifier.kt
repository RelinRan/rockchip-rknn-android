package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnImageClassificationResult

/**
 * Provides the `RknnImageClassifier` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnImageClassifier` where its surrounding API requires this contract.
 */
object RknnImageClassifier {
    /**
     * Executes `classify` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     * @param extras Optional backend-specific inference values.
     */
    fun classify(
        modelId: String,
        bitmap: Bitmap,
        extras: Map<String, Any?> = emptyMap(),
    ): RknnImageClassificationResult = RknnApi.classifyImage(modelId, bitmap, extras)
}
