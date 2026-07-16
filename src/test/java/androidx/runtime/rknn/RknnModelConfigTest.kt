package androidx.runtime.rknn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class RknnModelConfigTest {
    @Test
    fun `uses stable yolo postprocessing defaults`() {
        val config = config()

        assertEquals(0.5f, config.nmsThreshold)
        assertEquals(17, config.poseKeyPointCount)
    }

    @Test
    fun `rejects invalid nms threshold`() {
        assertThrows(IllegalArgumentException::class.java) { config(nmsThreshold = 1.1f) }
    }

    @Test
    fun `rejects non-positive pose key point count`() {
        assertThrows(IllegalArgumentException::class.java) { config(poseKeyPointCount = 0) }
    }

    private fun config(
        nmsThreshold: Float = 0.5f,
        poseKeyPointCount: Int = 17,
    ) = RknnModelConfig(
        id = "model",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "model.rknn",
        inputWidth = 640,
        inputHeight = 640,
        nmsThreshold = nmsThreshold,
        poseKeyPointCount = poseKeyPointCount,
    )
}
