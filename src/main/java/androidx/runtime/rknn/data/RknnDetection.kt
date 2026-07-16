package androidx.runtime.rknn.data

/** 一个目标的边界框和候选类别。 */
data class RknnDetection(
    val categories: List<RknnCategory>,
    val boundingBox: RknnBoundingBox,
    val keyPoints: List<RknnKeypoint> = emptyList(),
    val segmentationMask: RknnSegmentationMask? = null,
    val orientedBox: RknnOrientedBox? = null,
)
