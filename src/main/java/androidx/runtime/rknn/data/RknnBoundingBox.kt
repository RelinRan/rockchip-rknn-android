package androidx.runtime.rknn.data

/**
 * Axis-aligned detection bounds in source-image pixel coordinates.
 *
 * Use `RknnBoundingBox(10f, 20f, 110f, 220f)` and read [width], [height],
 * [centerX], or [centerY] for derived geometry.
 *
 * @property left Left edge in pixels.
 * @property top Top edge in pixels.
 * @property right Right edge in pixels.
 * @property bottom Bottom edge in pixels.
 */
data class RknnBoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    /** Box width in pixels. */
    val width: Float get() = right - left
    /** Box height in pixels. */
    val height: Float get() = bottom - top
    /** Horizontal center in source-image pixels. */
    val centerX: Float get() = (left + right) / 2f
    /** Vertical center in source-image pixels. */
    val centerY: Float get() = (top + bottom) / 2f
}
