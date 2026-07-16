package androidx.runtime.rknn

import androidx.runtime.rknn.internal.RknnJni
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer

/** RK3588 NPU 核心选择掩码。 */
enum class RknnCoreMask(internal val value: Int) {
    AUTO(0), CORE_0(1), CORE_1(2), CORE_2(4), CORE_0_1(3), CORE_0_1_2(7), ALL(0xffff),
}

/** CPU 与 NPU 之间同步张量内存的方向。 */
enum class RknnMemorySyncMode(internal val value: Int) {
    TO_DEVICE(1), FROM_DEVICE(2), BIDIRECTIONAL(3),
}

/** 对原生 RKNN Context 的受控封装，关闭后不可继续调用。 */
class RknnRuntimeSession private constructor(private var handle: Long) : Closeable {
    private val memories = linkedSetOf<RknnTensorMemory>()

    val isOpen: Boolean get() = handle != 0L

    fun duplicate(): RknnRuntimeSession {
        checkOpen()
        return RknnRuntimeSession(RknnJni.nativeDuplicateSession(handle)).also {
            check(it.isOpen) { "Failed to duplicate RKNN context" }
        }
    }

    fun setBatchCoreNumber(coreNumber: Int): Int {
        require(coreNumber > 0)
        return RknnJni.nativeSetBatchCoreNumber(requireHandle(), coreNumber)
    }

    fun setCoreMask(mask: RknnCoreMask): Int = RknnJni.nativeSetCoreMask(requireHandle(), mask.value)

    fun execute(nonBlocking: Boolean = false, frameId: Long = 0): Int =
        RknnJni.nativeExecute(requireHandle(), nonBlocking, frameId)

    fun waitFor(frameId: Long = 0): Int = RknnJni.nativeWait(requireHandle(), frameId)

    fun createMemory(size: Long, allocationFlags: Long = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemory(requireHandle(), size, allocationFlags))

    fun createMemoryFromFd(fd: Int, size: Int, offset: Int = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemoryFromFd(requireHandle(), fd, size, offset))

    fun createMemoryFromPhysical(address: Long, buffer: ByteBuffer?, size: Int): RknnTensorMemory {
        require(buffer == null || buffer.isDirect) { "Physical memory buffer must be direct" }
        return own(RknnJni.nativeCreateMemoryFromPhysical(requireHandle(), address, buffer, size))
    }

    fun createMemoryFromMbBlock(blockHandle: Long, offset: Int = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemoryFromMbBlock(requireHandle(), blockHandle, offset))

    fun setWeightMemory(memory: RknnTensorMemory): Int =
        RknnJni.nativeSetWeightMemory(requireHandle(), memory.requireHandle(this))

    fun setInternalMemory(memory: RknnTensorMemory): Int =
        RknnJni.nativeSetInternalMemory(requireHandle(), memory.requireHandle(this))

    fun setIoMemory(memory: RknnTensorMemory, tensorIndex: Int, input: Boolean): Int =
        RknnJni.nativeSetIoMemory(requireHandle(), memory.requireHandle(this), tensorIndex, input)

    fun setInputShape(tensorIndex: Int, vararg dimensions: Int): Int =
        RknnJni.nativeSetInputShape(requireHandle(), tensorIndex, dimensions)

    fun setInputShapes(shapes: Map<Int, IntArray>): Int = RknnJni.nativeSetInputShapes(
        requireHandle(), shapes.keys.toIntArray(), shapes.values.toTypedArray(),
    )

    override fun close() {
        if (handle == 0L) return
        memories.toList().forEach { it.close() }
        RknnJni.nativeCloseSession(handle)
        handle = 0L
    }

    private fun own(memoryHandle: Long): RknnTensorMemory {
        check(memoryHandle != 0L) { "Failed to create RKNN tensor memory" }
        return RknnTensorMemory(this, memoryHandle).also(memories::add)
    }

    private fun checkOpen() = check(handle != 0L) { "RKNN session is closed" }
    internal fun requireHandle(): Long = handle.also { check(it != 0L) { "RKNN session is closed" } }
    internal fun release(memory: RknnTensorMemory, memoryHandle: Long) {
        if (handle != 0L) RknnJni.nativeDestroyMemory(handle, memoryHandle)
        memories.remove(memory)
    }

    companion object {
        fun open(modelFile: File, debug: Boolean = false): RknnRuntimeSession {
            require(modelFile.isFile) { "RKNN model file not found: ${modelFile.absolutePath}" }
            return RknnRuntimeSession(RknnJni.nativeOpenSession(modelFile.absolutePath, debug)).also {
                check(it.isOpen) { "Failed to open RKNN model: ${modelFile.absolutePath}" }
            }
        }
    }
}

/** 由 RKNN Runtime 分配的原生张量内存，必须通过 [close] 释放。 */
class RknnTensorMemory internal constructor(
    private val owner: RknnRuntimeSession,
    private var handle: Long,
) : Closeable {
    val size: Long get() = RknnJni.nativeGetMemorySize(requireHandle(owner))
    val buffer: ByteBuffer? get() = RknnJni.nativeGetMemoryBuffer(requireHandle(owner))

    fun synchronize(mode: RknnMemorySyncMode): Int =
        RknnJni.nativeSynchronizeMemory(owner.requireHandle(), requireHandle(owner), mode.value)

    override fun close() {
        if (handle == 0L) return
        val value = handle
        handle = 0L
        owner.release(this, value)
    }

    internal fun requireHandle(expectedOwner: RknnRuntimeSession): Long {
        require(owner === expectedOwner) { "Tensor memory belongs to another RKNN session" }
        return handle.also { check(it != 0L) { "RKNN tensor memory is closed" } }
    }
}
