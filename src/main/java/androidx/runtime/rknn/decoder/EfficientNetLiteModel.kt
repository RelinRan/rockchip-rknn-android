package androidx.runtime.rknn.decoder

/**
 * Provides the `EfficientNetLiteModel` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `EfficientNetLiteModel` where its surrounding API requires this contract.
 */
enum class EfficientNetLiteModel(
    val inputSize: Int,
    val quantized: Boolean,
) {
    LITE0_FLOAT32(224, false),
    LITE0_INT8(224, true),
    LITE2_FLOAT32(224, false),
    LITE2_INT8(224, true),
}

/**
 * Provides the `ClassificationScoreType` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `ClassificationScoreType` where its surrounding API requires this contract.
 */
enum class ClassificationScoreType {
    /** Infers whether values are probabilities or logits from their range and sum. */
    AUTO,
    /** Values have already passed through Softmax and can be used as probabilities. */
    PROBABILITIES,
    /** Values are logits and require Softmax before thresholding. */
    LOGITS,
}
