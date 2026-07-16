package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.internal.NativeTensorOutput

/** 将 YOLO 的候选优先或通道优先张量统一为按候选行读取的只读视图。 */
internal class YoloTensorView private constructor(
    private val tensor: NativeTensorOutput,
    val rows: Int,
    val channels: Int,
    private val channelFirst: Boolean,
) {
    operator fun get(row: Int, channel: Int): Float {
        require(row in 0 until rows) { "Row index out of range: $row" }
        require(channel in 0 until channels) { "Channel index out of range: $channel" }
        val index = if (channelFirst) channel * rows + row else row * channels + channel
        return tensor.data[index]
    }

    companion object {
        fun create(
            tensor: NativeTensorOutput,
            expectedChannels: Set<Int>,
        ): YoloTensorView {
            require(expectedChannels.isNotEmpty()) { "Expected channels must not be empty" }
            val shape = tensor.dims.toMutableList().apply {
                while (size > 2) {
                    when {
                        first() == 1 -> removeAt(0)
                        last() == 1 -> removeAt(lastIndex)
                        else -> break
                    }
                }
            }
            require(shape.size == 2) {
                "YOLO tensor ${tensor.name} must be two-dimensional after batch removal, got ${format(tensor)}"
            }
            require(shape[0] > 0 && shape[1] > 0 && shape[0] * shape[1] == tensor.data.size) {
                "YOLO tensor ${tensor.name} data length does not match ${format(tensor)}"
            }
            val firstIsChannel = shape[0] in expectedChannels
            val secondIsChannel = shape[1] in expectedChannels
            require(firstIsChannel.xor(secondIsChannel)) {
                "YOLO tensor ${tensor.name} layout is ambiguous for channels " +
                    "${expectedChannels.sorted()}: ${format(tensor)}"
            }
            return if (firstIsChannel) {
                YoloTensorView(tensor, rows = shape[1], channels = shape[0], channelFirst = true)
            } else {
                YoloTensorView(tensor, rows = shape[0], channels = shape[1], channelFirst = false)
            }
        }

        private fun format(tensor: NativeTensorOutput): String =
            tensor.dims.joinToString(prefix = "[", postfix = "]")
    }
}
