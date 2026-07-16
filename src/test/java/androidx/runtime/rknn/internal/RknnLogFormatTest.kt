package androidx.runtime.rknn.internal

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RknnLogFormatTest {
    @Test
    fun environment_includesChipDriverAndRuntimeVersions() {
        val summary = RknnLogFormat.environment(
            chip = "RK3588",
            driverVersion = "0.8.2",
            runtimeVersion = "2.3.2",
        )

        assertTrue(summary.contains("chip=RK3588"))
        assertTrue(summary.contains("rknpuDriver=0.8.2"))
        assertTrue(summary.contains("deviceRuntime=2.3.2"))
    }

    @Test
    fun tensors_includesShapeMetadataWithoutTensorValues() {
        val tensor = NativeTensorOutput(
            index = 2,
            name = "detections",
            dims = intArrayOf(1, 10, 6),
            type = 3,
            format = 1,
            data = floatArrayOf(123.456f, 9f),
        )

        val summary = RknnLogFormat.tensors(listOf(tensor))

        assertTrue(summary.contains("index=2"))
        assertTrue(summary.contains("name=detections"))
        assertTrue(summary.contains("dims=1x10x6"))
        assertTrue(summary.contains("elements=2"))
        assertFalse(summary.contains("123.456"))
    }

    @Test
    fun completion_includesModelOutcomeAndDuration() {
        val summary = RknnLogFormat.completion(
            modelId = "action",
            success = true,
            durationMs = 18,
            outputCount = 2,
            message = "ok",
        )

        assertTrue(summary.contains("model=action"))
        assertTrue(summary.contains("success=true"))
        assertTrue(summary.contains("durationMs=18"))
        assertTrue(summary.contains("outputs=2"))
        assertTrue(summary.contains("message=ok"))
    }
}
