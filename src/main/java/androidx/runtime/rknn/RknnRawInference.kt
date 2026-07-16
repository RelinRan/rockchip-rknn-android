package androidx.runtime.rknn

import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/** JNI 返回的原始推理信息，保留输入变换和各输出张量。 */
data class RknnRawInference(
    val tensors: List<NativeTensorOutput>,
    val inputImage: RknnImage? = null,
    val apiVersion: String? = null,
    val driverVersion: String? = null,
)
