package androidx.runtime.rknn.internal

/** 已转换为 FloatArray 的 RKNN 输出张量及其属性。 */
data class NativeTensorOutput(
    val index: Int,
    val name: String,
    val dims: IntArray,
    val type: Int,
    val format: Int,
    val data: FloatArray,
)
