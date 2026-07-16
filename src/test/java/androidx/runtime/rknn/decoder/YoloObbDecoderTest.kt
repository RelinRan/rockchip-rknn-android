package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloObbDecoderTest {
    @Test
    fun `decodes candidate first oriented box`() {
        val tensor = output(intArrayOf(1, 1, 6), floatArrayOf(100f, 100f, 40f, 20f, 0.9f, 0f))

        val result = YoloObbDecoder.decode(tensor, image(), config())

        val oriented = result.single().orientedBox!!
        assertEquals(100f, oriented.centerX, 0.001f)
        assertEquals(40f, oriented.width, 0.001f)
        assertEquals(80f, result.single().boundingBox.left, 0.001f)
    }

    @Test
    fun `decodes channel first oriented box`() {
        val tensor = output(intArrayOf(1, 6, 1), floatArrayOf(100f, 100f, 40f, 20f, 0.9f, 0f))

        assertEquals(1, YoloObbDecoder.decode(tensor, image(), config()).size)
    }

    private fun config() = RknnModelConfig(
        "obb", RknnModelType.OBJECT_DETECTOR, "obb.rknn", 640, 640,
        labels = listOf("person"), scoreThreshold = 0.25f,
    )
    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
    private fun output(dims: IntArray, data: FloatArray) = NativeTensorOutput(0, "obb", dims, 0, 0, data)
}
