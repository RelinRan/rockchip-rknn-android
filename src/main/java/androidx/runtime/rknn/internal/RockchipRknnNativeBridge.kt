package androidx.runtime.rknn.internal

import android.util.Log
import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnInferenceResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnRawInference
import androidx.runtime.rknn.data.RknnImage

/**
 * Provides the `RockchipRknnNativeBridge` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RockchipRknnNativeBridge` where its surrounding API requires this contract.
 */
internal class RockchipRknnNativeBridge : RknnNativeBridge {

    override fun isAvailable(): Boolean = runCatching { RknnJni.nativeIsRuntimeAvailable() }.getOrDefault(false)

    override fun backendName(): String = runCatching { RknnJni.nativeRuntimeName() }.getOrDefault("rknn")

    override fun openSession(model: RknnModelConfig, modelPath: String, debug: Boolean): Long {
        return runCatching { RknnJni.nativeOpenSession(modelPath, debug) }
            .getOrElse { error ->
                Log.e(
                    "RknnSDK",
                    "nativeOpenSession exception model=${model.id} file=$modelPath",
                    error,
                )
                0L
            }
    }

    override fun closeSession(handle: Long) {
        runCatching { RknnJni.nativeCloseSession(handle) }
    }

    override fun run(handle: Long, image: RknnImage, model: RknnModelConfig, extras: Map<String, Any?>): RknnInferenceResult {
        val native = runCatching {
            RknnJni.nativeRun(
                handle, image.bytes, image.width, image.height, image.channels,
                model.inputType.nativeCode, model.inputLayout.nativeCode,
                model.normalization.mean, model.normalization.std,
            )
        }.getOrElse {
            return RknnInferenceResult(
                success = false,
                backend = RknnBackend.ROCKCHIP_RKNN,
                modelId = extras["modelId"] as? String ?: "",
                durationMs = 0L,
                message = it.message ?: "nativeRun failed"
            )
        }
        return RknnInferenceResult(
            success = native.success,
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = extras["modelId"] as? String ?: "",
            durationMs = native.durationMs,
            message = native.message,
            raw = RknnRawInference(
                tensors = native.outputs,
                apiVersion = native.apiVersion,
                driverVersion = native.driverVersion,
            ),
        )
    }
}
