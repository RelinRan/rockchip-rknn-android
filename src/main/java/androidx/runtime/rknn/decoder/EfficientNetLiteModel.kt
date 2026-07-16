package androidx.runtime.rknn.decoder

/** MediaPipe Image Classifier 提供的 EfficientNet-Lite 模型规格。 */
enum class EfficientNetLiteModel(
    val inputSize: Int,
    val quantized: Boolean,
) {
    LITE0_FLOAT32(224, false),
    LITE0_INT8(224, true),
    LITE2_FLOAT32(224, false),
    LITE2_INT8(224, true),
}

/** 分类输出张量中分数的含义。 */
enum class ClassificationScoreType {
    /** 根据数值范围和总和自动判断概率或 logits。 */
    AUTO,
    /** 输出已经过 Softmax，可直接作为概率使用。 */
    PROBABILITIES,
    /** 输出是 logits，需要先执行 Softmax。 */
    LOGITS,
}
