package androidx.runtime.rknn

import android.graphics.Bitmap
import androidx.runtime.rknn.data.RknnHandLandmarkResult
import androidx.runtime.rknn.data.RknnPoseLandmarkResult

/**
 * Provides the `RknnPoseLandmark` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnPoseLandmark` where its surrounding API requires this contract.
 */
object RknnPoseLandmark {
    /**
     * Executes `detect` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     */
    fun detect(modelId: String, bitmap: Bitmap): RknnPoseLandmarkResult =
        RknnApi.detectPoseLandmarks(modelId, bitmap)
}

/**
 * Provides the `RknnHandLandmark` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnHandLandmark` where its surrounding API requires this contract.
 */
object RknnHandLandmark {
    /**
     * Executes `detect` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param bitmap Source bitmap to preprocess and run through the model.
     */
    fun detect(modelId: String, bitmap: Bitmap): RknnHandLandmarkResult =
        RknnApi.detectHandLandmarks(modelId, bitmap)
}
