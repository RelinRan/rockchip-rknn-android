package androidx.runtime.rknn.internal

import android.os.SystemClock
import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnInferenceResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnImage

/**
 * Provides the `NoopNativeBridge` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `NoopNativeBridge` where its surrounding API requires this contract.
 */
internal class NoopNativeBridge : RknnNativeBridge {
    override fun isAvailable(): Boolean = false

    override fun backendName(): String = "noop"

    override fun openSession(model: RknnModelConfig, modelPath: String, debug: Boolean): Long = 0L

    override fun closeSession(handle: Long) = Unit

    override fun run(handle: Long, image: RknnImage, model: RknnModelConfig, extras: Map<String, Any?>): RknnInferenceResult {
        val start = SystemClock.elapsedRealtime()
        return RknnInferenceResult(
            success = false,
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = extras["modelId"] as? String ?: "",
            durationMs = SystemClock.elapsedRealtime() - start,
            message = "RKNN native bridge not loaded"
        )
    }
}
