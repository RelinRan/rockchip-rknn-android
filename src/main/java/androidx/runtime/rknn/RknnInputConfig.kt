package androidx.runtime.rknn

/**
 * Provides the `RknnInputType` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnInputType` where its surrounding API requires this contract.
 */
enum class RknnInputType(internal val nativeCode: Int) {
    AUTO(0),
    UINT8(1),
    INT8(2),
    FLOAT16(3),
    FLOAT32(4),
}

/**
 * Provides the `RknnInputLayout` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnInputLayout` where its surrounding API requires this contract.
 */
enum class RknnInputLayout(internal val nativeCode: Int) {
    AUTO(0),
    NHWC(1),
    NCHW(2),
}

/**
 * Provides the `RknnNormalization` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnNormalization` where its surrounding API requires this contract.
 */
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
