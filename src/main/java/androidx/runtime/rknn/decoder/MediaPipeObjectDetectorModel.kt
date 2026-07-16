package androidx.runtime.rknn.decoder

/** MediaPipe Model Maker 支持的目标检测模型规格。 */
enum class MediaPipeObjectDetectorModel(
    val inputSize: Int,
    internal val minLevel: Int,
    internal val maxLevel: Int,
) {
    AUTO(0, 0, 0),
    MOBILENET_V2(256, 3, 7),
    MOBILENET_V2_I320(320, 3, 6),
    MOBILENET_MULTI_AVG(256, 3, 7),
    MOBILENET_MULTI_AVG_I384(384, 3, 7),
    ;

    val anchorCount: Int
        get() {
            require(this != AUTO) { "AUTO does not have a fixed anchor count" }
            return (minLevel..maxLevel).sumOf { level ->
                val featureSize = inputSize / (1 shl level)
                featureSize * featureSize * ANCHORS_PER_LOCATION
            }
        }

    companion object {
        private const val ANCHORS_PER_LOCATION = 9

        /** 根据输入尺寸和锚框数量自动匹配模型规格。 */
        internal fun resolve(inputWidth: Int, inputHeight: Int, anchorCount: Int): MediaPipeObjectDetectorModel {
            require(inputWidth == inputHeight) { "MediaPipe object detector input must be square" }
            return entries.firstOrNull {
                it != AUTO && it.inputSize == inputWidth && it.anchorCount == anchorCount
            } ?: error("Unsupported MediaPipe object detector input ${inputWidth}x$inputHeight with $anchorCount anchors")
        }
    }
}
