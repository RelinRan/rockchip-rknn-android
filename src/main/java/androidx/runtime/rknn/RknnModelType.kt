package androidx.runtime.rknn

/** Semantic model purpose used to distinguish inference result types. */
enum class RknnModelType {
    OBJECT_DETECTOR,
    IMAGE_CLASSIFIER,
    POSE_LANDMARKER,
    POSE_DETECTOR,
    HAND_LANDMARKER,
    IMAGE_SEGMENTER,
    OBB_DETECTOR,
    MASK_HAT_DETECTOR,
    REID_EMBEDDING,
    CUSTOM,
}
