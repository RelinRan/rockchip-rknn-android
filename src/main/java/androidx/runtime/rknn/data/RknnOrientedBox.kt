package androidx.runtime.rknn.data

import kotlin.math.cos
import kotlin.math.sin

data class RknnPoint(val x: Float, val y: Float)

/** 以中心点、尺寸和弧度角表示的旋转边界框。 */
data class RknnOrientedBox(
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val rotationRadians: Float,
) {
    init {
        require(width > 0f && height > 0f) { "Oriented box dimensions must be positive" }
    }

    val corners: List<RknnPoint> by lazy {
        val cosine = cos(rotationRadians)
        val sine = sin(rotationRadians)
        listOf(
            -width / 2f to -height / 2f,
            width / 2f to -height / 2f,
            width / 2f to height / 2f,
            -width / 2f to height / 2f,
        ).map { (x, y) ->
            RknnPoint(centerX + x * cosine - y * sine, centerY + x * sine + y * cosine)
        }
    }

    fun boundingBox(): RknnBoundingBox = RknnBoundingBox(
        corners.minOf { it.x }, corners.minOf { it.y },
        corners.maxOf { it.x }, corners.maxOf { it.y },
    )
}
