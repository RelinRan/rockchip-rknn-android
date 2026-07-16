package androidx.runtime.rknn

/** 模型的业务用途，用于区分不同推理结果。 */
enum class RknnModelType {
    OBJECT_DETECTOR,
    IMAGE_CLASSIFIER,
    POSE_LANDMARKER,
    POSE_DETECTOR,
    HAND_LANDMARKER,
    IMAGE_SEGMENTER,
    OBB_DETECTOR,
    MASK_HAT_DETECTOR,
    CUSTOM,
}
