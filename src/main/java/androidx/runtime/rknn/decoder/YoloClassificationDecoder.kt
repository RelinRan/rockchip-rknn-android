package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.internal.NativeTensorOutput

/**
 * Provides the `YoloClassificationDecoder` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `YoloClassificationDecoder` where its surrounding API requires this contract.
 */
internal object YoloClassificationDecoder {
    /**
     * Executes `decode` for the RKNN runtime contract.
     * @param output Native output tensor to decode.
     * @param config Model or runtime configuration used by the operation.
     */
    fun decode(output: NativeTensorOutput, config: RknnModelConfig): List<RknnCategory> {
        require(output.data.size == config.labels.size) {
            "YOLO classification output ${output.name}${output.dims.contentToString()} does not match " +
                "${config.labels.size} labels"
        }
        return ImageClassifierDecoder.decode(output, config)
    }
}
