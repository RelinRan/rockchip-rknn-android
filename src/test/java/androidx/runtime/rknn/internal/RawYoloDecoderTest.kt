package androidx.runtime.rknn.internal

import androidx.runtime.rknn.decoder.RawYoloDecoder
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import org.junit.Assert.assertEquals
import org.junit.Test

class RawYoloDecoderTest {
    @Test
    fun `decodes channel first xywh output`() {
        val data = floatArrayOf(
            100f, 300f,
            120f, 300f,
            40f, 20f,
            20f, 20f,
            0.9f, 0.1f,
            0.1f, 0.8f,
        )
        val tensor = NativeTensorOutput(0, "output0", intArrayOf(1, 6, 2, 1), 1, 0, data)
        val config = RknnModelConfig("action", RknnModelType.OBJECT_DETECTOR, "action.rknn", 640, 640, listOf("A", "B"), 0.4f, 10)
        val image = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)

        val result = RawYoloDecoder.decode(tensor, image, config)

        assertEquals(2, result.size)
        assertEquals("A", result[0].categories.single().name)
        assertEquals(80f, result[0].boundingBox.left, 0.01f)
        assertEquals(110f, result[0].boundingBox.top, 0.01f)
    }

    @Test
    fun `decodes candidate first output`() {
        val tensor = NativeTensorOutput(
            0, "output0", intArrayOf(1, 2, 6), 1, 0,
            floatArrayOf(
                100f, 120f, 40f, 20f, 0.9f, 0.1f,
                300f, 300f, 20f, 20f, 0.1f, 0.8f,
            ),
        )

        val result = RawYoloDecoder.decode(tensor, image(), config())

        assertEquals(2, result.size)
        assertEquals("A", result[0].categories.single().name)
    }

    @Test
    fun `multiplies objectness by class score`() {
        val tensor = NativeTensorOutput(
            0, "output0", intArrayOf(1, 1, 7), 1, 0,
            floatArrayOf(100f, 120f, 40f, 20f, 0.5f, 0.8f, 0.1f),
        )

        val result = RawYoloDecoder.decode(tensor, image(), config(threshold = 0.39f))

        assertEquals(1, result.size)
        assertEquals(0.4f, result.single().categories.single().score, 0.0001f)
    }

    private fun config(threshold: Float = 0.4f) = RknnModelConfig(
        "action", RknnModelType.OBJECT_DETECTOR, "action.rknn", 640, 640,
        labels = listOf("A", "B"), scoreThreshold = threshold, maxResults = 10,
    )

    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
}
