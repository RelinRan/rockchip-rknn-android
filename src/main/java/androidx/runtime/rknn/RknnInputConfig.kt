package androidx.runtime.rknn

/** 输入张量的数据类型；AUTO 表示采用模型声明的类型。 */
enum class RknnInputType(internal val nativeCode: Int) {
    AUTO(0),
    UINT8(1),
    INT8(2),
    FLOAT16(3),
    FLOAT32(4),
}

/** 输入张量的数据布局；AUTO 表示采用模型声明的布局。 */
enum class RknnInputLayout(internal val nativeCode: Int) {
    AUTO(0),
    NHWC(1),
    NCHW(2),
}

/** 浮点输入的归一化参数，计算公式为 `(像素值 - mean) / std`。 */
data class RknnNormalization(
    val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
    val std: FloatArray = floatArrayOf(255f, 255f, 255f),
) {
    init {
        require(mean.size == 3 && std.size == 3) { "Normalization requires three RGB channels" }
        require(mean.all(Float::isFinite) && std.all { it.isFinite() && it != 0f }) {
            "Normalization values must be finite and standard deviations must be nonzero"
        }
    }
}
