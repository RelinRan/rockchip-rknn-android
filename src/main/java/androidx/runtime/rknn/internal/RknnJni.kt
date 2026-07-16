package androidx.runtime.rknn.internal

import java.nio.ByteBuffer

/** Kotlin 与 librknn_jni.so 之间的原生方法声明。 */
internal object RknnJni {

    init {
        System.loadLibrary("rknn_jni")
    }

    external fun nativeIsRuntimeAvailable(): Boolean

    external fun nativeRuntimeName(): String

    external fun nativeOpenSession(modelPath: String, debug: Boolean): Long

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
