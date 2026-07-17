package androidx.runtime.rknn.data

/**
 * One classification candidate associated with an inference result.
 *
 * Use [name] for display and [score] for thresholding or ordering.
 *
 * @property index Zero-based class index in the model output.
 * @property name Human-readable label, or a generated label when none was configured.
 * @property score Confidence or probability, normally in the `0..1` range.
 */
data class RknnCategory(
    val index: Int,
    val name: String,
    val score: Float,
)
