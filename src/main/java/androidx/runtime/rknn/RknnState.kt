package androidx.runtime.rknn

/**
 * Provides the `RknnState` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnState` where its surrounding API requires this contract.
 */
enum class RknnState {
    UNINITIALIZED,
    READY,
    UNSUPPORTED,
    ERROR,
}
