package androidx.runtime.rknn

import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/**
 * Raw JNI inference payload with tensor and input-transform metadata.
 *
 * Decoder implementations consume [tensors] together with [inputImage] to restore
 * model-space coordinates to the source image.
 *
 * @property tensors Output tensors converted to floating-point arrays.
 * @property inputImage Preprocessed image metadata, including scale and padding.
 * @property apiVersion RKNN Runtime API version reported by the native library.
 * @property driverVersion RKNPU driver version reported by the device.
 */
data class RknnRawInference(
    val tensors: List<NativeTensorOutput>,
    val inputImage: RknnImage? = null,
    val apiVersion: String? = null,
    val driverVersion: String? = null,
)
