package androidx.runtime.rknn.data

/** 预处理后的 RGB 图像数据以及恢复原图坐标所需的缩放信息。 */
data class RknnImage(
    val width: Int,
    val height: Int,
    val channels: Int = 3,
    val bytes: ByteArray,
    val originalWidth: Int = width,
    val originalHeight: Int = height,
    val scale: Float = 1f,
    val paddingLeft: Int = 0,
    val paddingTop: Int = 0,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as RknnImage

        if (width != other.width) return false
        if (height != other.height) return false
        if (channels != other.channels) return false
        if (originalWidth != other.originalWidth) return false
        if (originalHeight != other.originalHeight) return false
        if (scale != other.scale) return false
        if (paddingLeft != other.paddingLeft) return false
        if (paddingTop != other.paddingTop) return false
        if (!bytes.contentEquals(other.bytes)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + channels
        result = 31 * result + originalWidth
        result = 31 * result + originalHeight
        result = 31 * result + scale.hashCode()
        result = 31 * result + paddingLeft
        result = 31 * result + paddingTop
        result = 31 * result + bytes.contentHashCode()
        return result
    }
}
