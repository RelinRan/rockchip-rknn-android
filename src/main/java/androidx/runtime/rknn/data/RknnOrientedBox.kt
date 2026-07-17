package androidx.runtime.rknn.data

import kotlin.math.cos
import kotlin.math.sin

data class RknnPoint(val x: Float, val y: Float)

/**
 * Provides the `RknnOrientedBox` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnOrientedBox` where its surrounding API requires this contract.
 */
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

    /**
     * Executes `boundingBox` for the RKNN runtime contract.
     */
    fun boundingBox(): RknnBoundingBox = RknnBoundingBox(
        corners.minOf { it.x }, corners.minOf { it.y },
        corners.maxOf { it.x }, corners.maxOf { it.y },
    )
}
