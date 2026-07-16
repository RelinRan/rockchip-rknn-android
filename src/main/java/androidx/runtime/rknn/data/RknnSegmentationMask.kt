package androidx.runtime.rknn.data

/** 单个实例在模型输入空间中的概率掩码。 */
class RknnSegmentationMask(
    val width: Int,
    val height: Int,
    val probabilities: FloatArray,
) {
    init {
        require(width > 0 && height > 0) { "Mask dimensions must be positive" }
        require(probabilities.size == width * height) { "Mask data length does not match dimensions" }
    }

    fun toBinary(threshold: Float = 0.5f): ByteArray {
        require(threshold in 0f..1f) { "Mask threshold must be between 0 and 1" }
        return ByteArray(probabilities.size) { if (probabilities[it] >= threshold) 1 else 0 }
    }

    fun resize(targetWidth: Int, targetHeight: Int): RknnSegmentationMask {
        require(targetWidth > 0 && targetHeight > 0) { "Target dimensions must be positive" }
        val resized = FloatArray(targetWidth * targetHeight)
        repeat(targetHeight) { y ->
            val sourceY = (y * height / targetHeight).coerceAtMost(height - 1)
            repeat(targetWidth) { x ->
                val sourceX = (x * width / targetWidth).coerceAtMost(width - 1)
                resized[y * targetWidth + x] = probabilities[sourceY * width + sourceX]
            }
        }
        return RknnSegmentationMask(targetWidth, targetHeight, resized)
    }
}
