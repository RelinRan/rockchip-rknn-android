package androidx.runtime.rknn

/** RKNN 运行时生命周期状态。 */
enum class RknnState {
    UNINITIALIZED,
    READY,
    UNSUPPORTED,
    ERROR,
}
