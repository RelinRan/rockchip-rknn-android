package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.data.RknnCategory
import androidx.runtime.rknn.internal.NativeTensorOutput

/** 解码 Ultralytics YOLO Classification 的类别向量。 */
internal object YoloClassificationDecoder {
    fun decode(output: NativeTensorOutput, config: RknnModelConfig): List<RknnCategory> {
        require(output.data.size == config.labels.size) {
            "YOLO classification output ${output.name}${output.dims.contentToString()} does not match " +
                "${config.labels.size} labels"
        }
        return ImageClassifierDecoder.decode(output, config)
    }
}
