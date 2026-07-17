package androidx.runtime.rknn.internal

import android.util.Log
import androidx.runtime.rknn.RknnModelConfig
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Provides the `RockchipSessionManager` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RockchipSessionManager` where its surrounding API requires this contract.
 */
internal class RockchipSessionManager(
    private val nativeBridge: RknnNativeBridge,
    private val debug: Boolean = false,
) {

    private val sessionHandles = ConcurrentHashMap<String, Long>()
    private val failedModelIds = ConcurrentHashMap.newKeySet<String>()
    private val sessionLocks = ConcurrentHashMap<String, Any>()

    @Synchronized
    /**
     * Executes `ensureSession` for the RKNN runtime contract.
     * @param model Value supplied for `model`.
     * @param modelFile RKNN model file to open.
     */
    fun ensureSession(model: RknnModelConfig, modelFile: File): Long {
        sessionHandles[model.id]?.let { return it }
        if (failedModelIds.contains(model.id)) return 0L
        val handle = nativeBridge.openSession(model, modelFile.absolutePath, debug)
        if (handle != 0L) {
            sessionHandles[model.id] = handle
        } else {
            failedModelIds += model.id
            if (debug) {
                Log.w("RknnSDK", "openSession failed model=${model.id} file=${modelFile.absolutePath}; retry blocked")
            }
        }
        return handle
    }

    /**
     * Executes `resetFailure` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     */
    fun resetFailure(modelId: String) {
        failedModelIds.remove(modelId)
    }

    /**
     * Executes `withSession` for the RKNN runtime contract.
     * @param model Value supplied for `model`.
     * @param modelFile RKNN model file to open.
     * @param operation Value supplied for `operation`.
     */
    fun <T> withSession(model: RknnModelConfig, modelFile: File, operation: (Long) -> T): T? {
        val lock = sessionLocks.computeIfAbsent(model.id) { Any() }
        return synchronized(lock) {
            val handle = ensureSession(model, modelFile)
            if (handle == 0L) null else operation(handle)
        }
    }

    /**
     * Executes `closeSession` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     */
    fun closeSession(modelId: String) {
        val lock = sessionLocks.computeIfAbsent(modelId) { Any() }
        synchronized(lock) {
            failedModelIds.remove(modelId)
            sessionHandles.remove(modelId)?.let(nativeBridge::closeSession)
        }
        sessionLocks.remove(modelId, lock)
    }

    /**
     * Executes `closeAll` for the RKNN runtime contract.
     */
    fun closeAll() {
        sessionHandles.entries.forEach { (_, handle) ->
            nativeBridge.closeSession(handle)
        }
        sessionHandles.clear()
        failedModelIds.clear()
        sessionLocks.clear()
    }
}
