package androidx.runtime.rknn.internal

import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnInferenceResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

class RockchipSessionManagerTest {

    @Test
    fun nonZeroUnsignedContextIsAcceptedWhenSignedLongIsNegative() {
        val unsignedContext = Long.MIN_VALUE + 42
        val bridge = RecordingBridge(openResults = ArrayDeque(listOf(unsignedContext)))
        val manager = RockchipSessionManager(bridge)

        assertEquals(unsignedContext, manager.ensureSession(model(), modelFile()))
        assertEquals(unsignedContext, manager.ensureSession(model(), modelFile()))
        assertEquals(1, bridge.openCount)
    }

    @Test
    fun failedSessionIsNotOpenedAgainUntilFailureIsReset() {
        val bridge = RecordingBridge(openResults = ArrayDeque(listOf(0L, 42L)))
        val manager = RockchipSessionManager(bridge)

        assertEquals(0L, manager.ensureSession(model(), modelFile()))
        assertEquals(0L, manager.ensureSession(model(), modelFile()))
        assertEquals(1, bridge.openCount)

        manager.resetFailure("action")

        assertEquals(42L, manager.ensureSession(model(), modelFile()))
        assertEquals(2, bridge.openCount)
    }

    @Test
    fun closingModelClearsFailedStateForReregistration() {
        val bridge = RecordingBridge(openResults = ArrayDeque(listOf(0L, 24L)))
        val manager = RockchipSessionManager(bridge)

        assertEquals(0L, manager.ensureSession(model(), modelFile()))
        manager.closeSession("action")

        assertEquals(24L, manager.ensureSession(model(), modelFile()))
        assertEquals(2, bridge.openCount)
    }

    private fun model() = RknnModelConfig(
        id = "action",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "action.rknn",
        inputWidth = 640,
        inputHeight = 640,
    )

    private fun modelFile() = File("action.rknn")

    private class RecordingBridge(
        private val openResults: ArrayDeque<Long>,
    ) : RknnNativeBridge {
        var openCount = 0

        override fun isAvailable() = true
        override fun backendName() = "rknn"
        override fun openSession(model: RknnModelConfig, modelPath: String, debug: Boolean): Long {
            openCount += 1
            return openResults.removeFirst()
        }

        override fun closeSession(handle: Long) = Unit

        override fun run(
            handle: Long,
            image: RknnImage,
            model: RknnModelConfig,
            extras: Map<String, Any?>,
        ) = RknnInferenceResult(
            success = true,
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = "action",
            durationMs = 0L,
        )
    }
}
