package androidx.runtime.rknn.internal

import androidx.runtime.rknn.decoder.Yolo26Decoder
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class Yolo26DecoderTest {
    private val config = RknnModelConfig(
        id = "wash",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "wash.rknn",
        inputWidth = 640,
        inputHeight = 640,
        labels = listOf("1X", "2L"),
        scoreThreshold = 0.25f,
        maxResults = 10,
    )
    private val image = RknnImage(
        width = 640,
        height = 640,
        bytes = ByteArray(640 * 640 * 3),
        originalWidth = 1920,
        originalHeight = 1080,
        scale = 1f / 3f,
        paddingLeft = 0,
        paddingTop = 140,
    )

    @Test
    fun decodesAndRestoresLetterboxCoordinates() {
        val output = NativeTensorOutput(
            index = 0,
            name = "output0",
            dims = intArrayOf(1, 2, 6),
            type = 0,
            format = 0,
            data = floatArrayOf(
                64f, 176f, 320f, 320f, 0.9f, 1f,
                10f, 10f, 20f, 20f, 0.1f, 0f,
            ),
        )

        val detections = Yolo26Decoder.decode(output, image, config)

        assertEquals(1, detections.size)
        assertEquals("2L", detections.single().categories.single().name)
        assertEquals(0.9f, detections.single().categories.single().score)
        assertEquals(192f, detections.single().boundingBox.left)
        assertEquals(108f, detections.single().boundingBox.top)
        assertEquals(960f, detections.single().boundingBox.right)
        assertEquals(540f, detections.single().boundingBox.bottom)
    }

    @Test
    fun decodesChannelFirstEndToEndOutput() {
        val output = NativeTensorOutput(
            0, "output0", intArrayOf(1, 6, 2), 0, 0,
            floatArrayOf(
                64f, 10f, 176f, 10f, 320f, 20f,
                320f, 20f, 0.9f, 0.1f, 1f, 0f,
            ),
        )

        val detections = Yolo26Decoder.decode(output, image, config)

        assertEquals(1, detections.size)
        assertEquals("2L", detections.single().categories.single().name)
        assertEquals(192f, detections.single().boundingBox.left)
    }

    @Test
    fun rejectsNonYolo26OutputShape() {
        val output = NativeTensorOutput(0, "output0", intArrayOf(1, 28, 8400), 0, 0, FloatArray(28 * 8400))

        try {
            Yolo26Decoder.decode(output, image, config)
            fail("Expected invalid YOLO26 shape to be rejected")
        } catch (_: IllegalArgumentException) {
            // Expected.
        }
    }

    @Test
    fun sortsByScoreAndLimitsResults() {
        val limited = config.copy(maxResults = 1)
        val output = NativeTensorOutput(
            0, "output0", intArrayOf(1, 2, 6), 0, 0,
            floatArrayOf(
                10f, 150f, 20f, 160f, 0.5f, 0f,
                30f, 150f, 40f, 160f, 0.8f, 1f,
            ),
        )

        val detections = Yolo26Decoder.decode(output, image, limited)

        assertEquals(1, detections.size)
        assertEquals("2L", detections.single().categories.single().name)
    }
}
