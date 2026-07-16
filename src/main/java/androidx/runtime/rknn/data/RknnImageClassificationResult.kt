package androidx.runtime.rknn.data

import androidx.runtime.rknn.RknnBackend

/** 图像分类结果，类别按置信度从高到低排列。 */
data class RknnImageClassificationResult(
    val success: Boolean,
    val backend: RknnBackend,
    val modelId: String,
    val categories: List<RknnCategory>,
    val durationMs: Long,
    val message: String? = null,
)
