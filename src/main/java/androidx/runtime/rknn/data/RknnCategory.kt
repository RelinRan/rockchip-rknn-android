package androidx.runtime.rknn.data

/** 单个检测类别及其置信度。 */
data class RknnCategory(
    val index: Int,
    val name: String,
    val score: Float,
)
