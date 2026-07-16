package androidx.runtime.rknn.data

/** 图像坐标关键点；x、y、z 均按模型输入尺寸归一化。 */
data class RknnLandmark(
    val x: Float,
    val y: Float,
    val z: Float,
    val visibility: Float = 1f,
    val presence: Float = 1f,
)

/** 以米为单位的三维世界坐标关键点。 */
data class RknnWorldLandmark(
    val x: Float,
    val y: Float,
    val z: Float,
)
