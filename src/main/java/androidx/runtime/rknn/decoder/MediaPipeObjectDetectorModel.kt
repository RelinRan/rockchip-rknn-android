package androidx.runtime.rknn.decoder

/**
 * Provides the `MediaPipeObjectDetectorModel` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `MediaPipeObjectDetectorModel` where its surrounding API requires this contract.
 */
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

        /** Resolves a model specification from its input dimensions and anchor count. */
        /**
         * Executes `resolve` for the RKNN runtime contract.
         * @param inputWidth Value supplied for `inputWidth`.
         * @param inputHeight Value supplied for `inputHeight`.
         * @param anchorCount Value supplied for `anchorCount`.
         */
        internal fun resolve(inputWidth: Int, inputHeight: Int, anchorCount: Int): MediaPipeObjectDetectorModel {
            require(inputWidth == inputHeight) { "MediaPipe object detector input must be square" }
            return entries.firstOrNull {
                it != AUTO && it.inputSize == inputWidth && it.anchorCount == anchorCount
            } ?: error("Unsupported MediaPipe object detector input ${inputWidth}x$inputHeight with $anchorCount anchors")
        }
    }
}
