package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnDetection
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/**
 * Provides the `EfficientDetDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `EfficientDetDecoder` where its surrounding API requires this contract.
 */
internal object EfficientDetDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param boxes Value supplied for `boxes`.
     * @param scores Value supplied for `scores`.
     * @param image Preprocessed image and coordinate-transform metadata.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(
        boxes: NativeTensorOutput,
        scores: NativeTensorOutput,
        image: RknnImage,
        config: RknnModelConfig,
    ): List<RknnDetection> = MediaPipeSsdDecoder.decode(boxes, scores, image, config)
}
