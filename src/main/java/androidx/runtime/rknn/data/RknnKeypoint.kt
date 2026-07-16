package androidx.runtime.rknn.data

/** YOLO Pose 在原始图像坐标系中的关键点。 */
data class RknnKeypoint(
    val index: Int,
    val name: String,
    val x: Float,
    val y: Float,
    val score: Float,
)
