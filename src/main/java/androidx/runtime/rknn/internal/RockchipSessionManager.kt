package androidx.runtime.rknn.internal

import android.util.Log
import androidx.runtime.rknn.RknnModelConfig
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * 管理模型 Session 及每个 Session 的推理锁。
 * 首次创建失败后会阻止逐帧重试，只有重置失败状态后才允许重新创建。
 */
internal class RockchipSessionManager(
    private val nativeBridge: RknnNativeBridge,
    private val debug: Boolean = false,
) {

    private val sessionHandles = ConcurrentHashMap<String, Long>()
    private val failedModelIds = ConcurrentHashMap.newKeySet<String>()
    private val sessionLocks = ConcurrentHashMap<String, Any>()

    @Synchronized
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

    fun resetFailure(modelId: String) {
        failedModelIds.remove(modelId)
    }

    fun <T> withSession(model: RknnModelConfig, modelFile: File, operation: (Long) -> T): T? {
        val lock = sessionLocks.computeIfAbsent(model.id) { Any() }
        return synchronized(lock) {
            val handle = ensureSession(model, modelFile)
            if (handle == 0L) null else operation(handle)
        }
    }

    fun closeSession(modelId: String) {
        val lock = sessionLocks.computeIfAbsent(modelId) { Any() }
        synchronized(lock) {
            failedModelIds.remove(modelId)
            sessionHandles.remove(modelId)?.let(nativeBridge::closeSession)
        }
        sessionLocks.remove(modelId, lock)
    }

    fun closeAll() {
        sessionHandles.entries.forEach { (_, handle) ->
            nativeBridge.closeSession(handle)
        }
        sessionHandles.clear()
        failedModelIds.clear()
        sessionLocks.clear()
    }
}
