package androidx.runtime.rknn.internal

import androidx.runtime.rknn.RknnInferenceResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnImage

/** 隔离 Kotlin 运行时与 RKNN JNI 实现的内部接口。 */
internal interface RknnNativeBridge {
    fun isAvailable(): Boolean
    fun backendName(): String
    fun openSession(model: RknnModelConfig, modelPath: String, debug: Boolean): Long
    fun closeSession(handle: Long)
    fun run(handle: Long, image: RknnImage, model: RknnModelConfig, extras: Map<String, Any?> = emptyMap()): RknnInferenceResult
}
