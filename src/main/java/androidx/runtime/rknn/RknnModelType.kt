package androidx.runtime.rknn

/**
 * Provides the `RknnModelType` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnModelType` where its surrounding API requires this contract.
 */
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
