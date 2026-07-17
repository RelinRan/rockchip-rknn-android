package androidx.runtime.rknn

/**
 * Provides the `RknnBackend` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnBackend` where its surrounding API requires this contract.
 */
enum class RknnBackend {
    ROCKCHIP_RKNN,
}
