package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/** EfficientDet 的兼容入口，新接入模型统一使用 [MediaPipeSsdDecoder]。 */
internal object EfficientDetDecoder {
    fun decode(
        boxes: NativeTensorOutput,
        scores: NativeTensorOutput,
        image: RknnImage,
        config: RknnModelConfig,
    ): List<RknnDetection> = MediaPipeSsdDecoder.decode(boxes, scores, image, config)
}
