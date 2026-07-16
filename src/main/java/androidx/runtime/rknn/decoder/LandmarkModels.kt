package androidx.runtime.rknn.decoder

/** MediaPipe Pose Landmark 的三种精度/性能规格，输出协议完全一致。 */
enum class PoseLandmarkModel(val inputSize: Int) {
    LITE(256),
    FULL(256),
    HEAVY(256),
}

/** MediaPipe Hand Landmark 模型规格。 */
enum class HandLandmarkModel(val inputSize: Int) {
    FULL(224),
}
