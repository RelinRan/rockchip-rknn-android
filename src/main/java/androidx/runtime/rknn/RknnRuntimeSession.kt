package androidx.runtime.rknn

import androidx.runtime.rknn.internal.RknnJni
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer

/**
 * Provides the `RknnCoreMask` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnCoreMask` where its surrounding API requires this contract.
 */
enum class RknnCoreMask(internal val value: Int) {
    AUTO(0), CORE_0(1), CORE_1(2), CORE_2(4), CORE_0_1(3), CORE_0_1_2(7), ALL(0xffff),
}

/**
 * Provides the `RknnMemorySyncMode` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnMemorySyncMode` where its surrounding API requires this contract.
 */
enum class RknnMemorySyncMode(internal val value: Int) {
    TO_DEVICE(1), FROM_DEVICE(2), BIDIRECTIONAL(3),
}

/**
 * Provides the `RknnRuntimeSession` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnRuntimeSession` where its surrounding API requires this contract.
 */
class RknnRuntimeSession private constructor(private var handle: Long) : Closeable {
    private val memories = linkedSetOf<RknnTensorMemory>()

    val isOpen: Boolean get() = handle != 0L

    /**
     * Executes `duplicate` for the RKNN runtime contract.
     */
    fun duplicate(): RknnRuntimeSession {
        checkOpen()
        return RknnRuntimeSession(RknnJni.nativeDuplicateSession(handle)).also {
            check(it.isOpen) { "Failed to duplicate RKNN context" }
        }
    }

    /**
     * Executes `setBatchCoreNumber` for the RKNN runtime contract.
     * @param coreNumber Value supplied for `coreNumber`.
     */
    fun setBatchCoreNumber(coreNumber: Int): Int {
        require(coreNumber > 0)
        return RknnJni.nativeSetBatchCoreNumber(requireHandle(), coreNumber)
    }

    /**
     * Executes `setCoreMask` for the RKNN runtime contract.
     * @param mask Value supplied for `mask`.
     */
    fun setCoreMask(mask: RknnCoreMask): Int = RknnJni.nativeSetCoreMask(requireHandle(), mask.value)

    /**
     * Executes `execute` for the RKNN runtime contract.
     * @param nonBlocking Value supplied for `nonBlocking`.
     * @param frameId Value supplied for `frameId`.
     */
    fun execute(nonBlocking: Boolean = false, frameId: Long = 0): Int =
        RknnJni.nativeExecute(requireHandle(), nonBlocking, frameId)

    /**
     * Executes `waitFor` for the RKNN runtime contract.
     * @param frameId Value supplied for `frameId`.
     */
    fun waitFor(frameId: Long = 0): Int = RknnJni.nativeWait(requireHandle(), frameId)

    /**
     * Executes `createMemory` for the RKNN runtime contract.
     * @param size Value supplied for `size`.
     * @param allocationFlags Value supplied for `allocationFlags`.
     */
    fun createMemory(size: Long, allocationFlags: Long = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemory(requireHandle(), size, allocationFlags))

    /**
     * Executes `createMemoryFromFd` for the RKNN runtime contract.
     * @param fd Value supplied for `fd`.
     * @param size Value supplied for `size`.
     * @param offset Value supplied for `offset`.
     */
    fun createMemoryFromFd(fd: Int, size: Int, offset: Int = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemoryFromFd(requireHandle(), fd, size, offset))

    /**
     * Executes `createMemoryFromPhysical` for the RKNN runtime contract.
     * @param address Value supplied for `address`.
     * @param buffer Value supplied for `buffer`.
     * @param size Value supplied for `size`.
     */
    fun createMemoryFromPhysical(address: Long, buffer: ByteBuffer?, size: Int): RknnTensorMemory {
        require(buffer == null || buffer.isDirect) { "Physical memory buffer must be direct" }
        return own(RknnJni.nativeCreateMemoryFromPhysical(requireHandle(), address, buffer, size))
    }

    /**
     * Executes `createMemoryFromMbBlock` for the RKNN runtime contract.
     * @param blockHandle Value supplied for `blockHandle`.
     * @param offset Value supplied for `offset`.
     */
    fun createMemoryFromMbBlock(blockHandle: Long, offset: Int = 0): RknnTensorMemory =
        own(RknnJni.nativeCreateMemoryFromMbBlock(requireHandle(), blockHandle, offset))

    /**
     * Executes `setWeightMemory` for the RKNN runtime contract.
     * @param memory Value supplied for `memory`.
     */
    fun setWeightMemory(memory: RknnTensorMemory): Int =
        RknnJni.nativeSetWeightMemory(requireHandle(), memory.requireHandle(this))

    /**
     * Executes `setInternalMemory` for the RKNN runtime contract.
     * @param memory Value supplied for `memory`.
     */
    fun setInternalMemory(memory: RknnTensorMemory): Int =
        RknnJni.nativeSetInternalMemory(requireHandle(), memory.requireHandle(this))

    /**
     * Executes `setIoMemory` for the RKNN runtime contract.
     * @param memory Value supplied for `memory`.
     * @param tensorIndex Value supplied for `tensorIndex`.
     * @param input Value supplied for `input`.
     */
    fun setIoMemory(memory: RknnTensorMemory, tensorIndex: Int, input: Boolean): Int =
        RknnJni.nativeSetIoMemory(requireHandle(), memory.requireHandle(this), tensorIndex, input)

    /**
     * Executes `setInputShape` for the RKNN runtime contract.
     * @param tensorIndex Value supplied for `tensorIndex`.
     * @param dimensions Value supplied for `dimensions`.
     */
    fun setInputShape(tensorIndex: Int, vararg dimensions: Int): Int =
        RknnJni.nativeSetInputShape(requireHandle(), tensorIndex, dimensions)

    /**
     * Executes `setInputShapes` for the RKNN runtime contract.
     * @param shapes Value supplied for `shapes`.
     */
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
    /**
     * Executes `requireHandle` for the RKNN runtime contract.
     */
    internal fun requireHandle(): Long = handle.also { check(it != 0L) { "RKNN session is closed" } }
    /**
     * Executes `release` for the RKNN runtime contract.
     * @param memory Value supplied for `memory`.
     * @param memoryHandle Value supplied for `memoryHandle`.
     */
    internal fun release(memory: RknnTensorMemory, memoryHandle: Long) {
        if (handle != 0L) RknnJni.nativeDestroyMemory(handle, memoryHandle)
        memories.remove(memory)
    }

    companion object {
        /**
         * Executes `open` for the RKNN runtime contract.
         * @param modelFile RKNN model file to open.
         * @param debug Whether verbose native diagnostics are enabled.
         */
        fun open(modelFile: File, debug: Boolean = false): RknnRuntimeSession {
            require(modelFile.isFile) { "RKNN model file not found: ${modelFile.absolutePath}" }
            return RknnRuntimeSession(RknnJni.nativeOpenSession(modelFile.absolutePath, debug)).also {
                check(it.isOpen) { "Failed to open RKNN model: ${modelFile.absolutePath}" }
            }
        }
    }
}

/**
 * Provides the `RknnTensorMemory` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnTensorMemory` where its surrounding API requires this contract.
 */
class RknnTensorMemory internal constructor(
    private val owner: RknnRuntimeSession,
    private var handle: Long,
) : Closeable {
    val size: Long get() = RknnJni.nativeGetMemorySize(requireHandle(owner))
    val buffer: ByteBuffer? get() = RknnJni.nativeGetMemoryBuffer(requireHandle(owner))

    /**
     * Executes `synchronize` for the RKNN runtime contract.
     * @param mode Value supplied for `mode`.
     */
    fun synchronize(mode: RknnMemorySyncMode): Int =
        RknnJni.nativeSynchronizeMemory(owner.requireHandle(), requireHandle(owner), mode.value)

    override fun close() {
        if (handle == 0L) return
        val value = handle
        handle = 0L
        owner.release(this, value)
    }

    /**
     * Executes `requireHandle` for the RKNN runtime contract.
     * @param expectedOwner Value supplied for `expectedOwner`.
     */
    internal fun requireHandle(expectedOwner: RknnRuntimeSession): Long {
        require(owner === expectedOwner) { "Tensor memory belongs to another RKNN session" }
        return handle.also { check(it != 0L) { "RKNN tensor memory is closed" } }
    }
}
