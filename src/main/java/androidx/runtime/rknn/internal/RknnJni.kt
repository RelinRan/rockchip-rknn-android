package androidx.runtime.rknn.internal

import java.nio.ByteBuffer

/**
 * Native boundary between Kotlin and `librknn_jni.so`.
 *
 * Usage: higher-level runtime classes call these methods after this object loads the JNI
 * library. Native handles are opaque, non-zero values and must be released by their owner.
 */
internal object RknnJni {

    init {
        System.loadLibrary("rknn_jni")
    }

    external fun nativeIsRuntimeAvailable(): Boolean

    external fun nativeRuntimeName(): String

    /** Opens [modelPath] and returns an opaque session handle, or zero on failure. */
    external fun nativeOpenSession(modelPath: String, debug: Boolean): Long

    /** Releases the native session identified by [handle]. */
    external fun nativeCloseSession(handle: Long)

    external fun nativeDuplicateSession(handle: Long): Long
    external fun nativeSetBatchCoreNumber(handle: Long, coreNumber: Int): Int
    external fun nativeSetCoreMask(handle: Long, coreMask: Int): Int
    external fun nativeWait(handle: Long, frameId: Long): Int
    external fun nativeExecute(handle: Long, nonBlocking: Boolean, frameId: Long): Int
    external fun nativeCreateMemory(handle: Long, size: Long, allocationFlags: Long): Long
    external fun nativeCreateMemoryFromFd(handle: Long, fd: Int, size: Int, offset: Int): Long
    external fun nativeCreateMemoryFromPhysical(handle: Long, physicalAddress: Long, buffer: ByteBuffer?, size: Int): Long
    external fun nativeCreateMemoryFromMbBlock(handle: Long, blockHandle: Long, offset: Int): Long
    external fun nativeGetMemoryBuffer(memoryHandle: Long): ByteBuffer?
    external fun nativeGetMemorySize(memoryHandle: Long): Long
    external fun nativeDestroyMemory(handle: Long, memoryHandle: Long): Int
    external fun nativeSetWeightMemory(handle: Long, memoryHandle: Long): Int
    external fun nativeSetInternalMemory(handle: Long, memoryHandle: Long): Int
    external fun nativeSetIoMemory(handle: Long, memoryHandle: Long, tensorIndex: Int, input: Boolean): Int
    external fun nativeSetInputShape(handle: Long, tensorIndex: Int, dimensions: IntArray): Int
    external fun nativeSetInputShapes(handle: Long, tensorIndices: IntArray, dimensions: Array<IntArray>): Int
    external fun nativeSynchronizeMemory(handle: Long, memoryHandle: Long, mode: Int): Int

    /**
     * Runs synchronous inference for one RGB image.
     *
     * @param handle Open native session handle.
     * @param imageBytes Packed image bytes in RGB channel order.
     * @param width Input image width in pixels.
     * @param height Input image height in pixels.
     * @param channels Number of packed channels, normally three.
     * @param inputType Native RKNN tensor-type code.
     * @param inputLayout Native RKNN tensor-layout code.
     * @param mean Per-channel normalization means.
     * @param std Per-channel normalization divisors.
     * @return Native status and output tensors.
     */
    external fun nativeRun(
        handle: Long,
        imageBytes: ByteArray,
        width: Int,
        height: Int,
        channels: Int,
        inputType: Int,
        inputLayout: Int,
        mean: FloatArray,
        std: FloatArray,
    ): NativeRunResult
}
