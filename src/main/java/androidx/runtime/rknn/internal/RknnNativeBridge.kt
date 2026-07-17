package androidx.runtime.rknn.internal

import androidx.runtime.rknn.RknnInferenceResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnImage

/**
 * Provides the `RknnNativeBridge` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnNativeBridge` where its surrounding API requires this contract.
 */
internal interface RknnNativeBridge {
    /**
     * Executes `isAvailable` for the RKNN runtime contract.
     * @param model Value supplied for `model`.
     * @param modelPath Absolute path to the RKNN model file.
     * @param debug Whether verbose native diagnostics are enabled.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param model Value supplied for `model`.
     * @param extras Optional backend-specific inference values.
     */
    fun isAvailable(): Boolean
    /**
     * Executes `backendName` for the RKNN runtime contract.
     * @param model Value supplied for `model`.
     * @param modelPath Absolute path to the RKNN model file.
     * @param debug Whether verbose native diagnostics are enabled.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param model Value supplied for `model`.
     * @param extras Optional backend-specific inference values.
     */
    fun backendName(): String
    /**
     * Executes `openSession` for the RKNN runtime contract.
     * @param model Value supplied for `model`.
     * @param modelPath Absolute path to the RKNN model file.
     * @param debug Whether verbose native diagnostics are enabled.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param model Value supplied for `model`.
     * @param extras Optional backend-specific inference values.
     */
    fun openSession(model: RknnModelConfig, modelPath: String, debug: Boolean): Long
    /**
     * Executes `closeSession` for the RKNN runtime contract.
     * @param handle Non-zero native RKNN resource handle.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param model Value supplied for `model`.
     * @param extras Optional backend-specific inference values.
     */
    fun closeSession(handle: Long)
    /**
     * Executes `run` for the RKNN runtime contract.
     * @param handle Non-zero native RKNN resource handle.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param model Value supplied for `model`.
     * @param extras Optional backend-specific inference values.
     */
    fun run(handle: Long, image: RknnImage, model: RknnModelConfig, extras: Map<String, Any?> = emptyMap()): RknnInferenceResult
}
