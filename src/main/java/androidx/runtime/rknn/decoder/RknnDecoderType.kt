package androidx.runtime.rknn.decoder

/**
 * Provides the `RknnDecoderType` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnDecoderType` where its surrounding API requires this contract.
 */
enum class RknnDecoderType {
    AUTO,
    YOLO_END_TO_END,
    YOLO_DETECT_RAW,
    YOLO_DETECT_HEADS,
    YOLO_POSE_LANDMARK,
    YOLO_POSE_RAW,
    YOLO_SEGMENT,
    YOLO_CLASSIFY,
    YOLO_OBB,
    YOLO_POSE_HEADS,
    MEDIA_PIPE_SSD,
    MEDIA_PIPE_IMAGE_CLASSIFIER,
    MEDIA_PIPE_POSE_LANDMARK,
    MEDIA_PIPE_HAND_LANDMARK,
}
