package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloDflDecoderTest {
    @Test
    fun `decodes combined nchw dfl head`() {
        val channels = 4 * 2 + 1
        val data = FloatArray(channels)
        repeat(4) { side ->
            data[side * 2] = -10f
            data[side * 2 + 1] = 10f
        }
        data[8] = 0.9f

        val result = YoloDflDecoder.decode(
            listOf(tensor("head", intArrayOf(1, channels, 1, 1), data)), image(), config(),
        )

        assertEquals(1, result.size)
        assertEquals(0.9f, result.single().categories.single().score, 0.001f)
        assertEquals(0f, result.single().boundingBox.left, 0.001f)
        assertEquals(640f, result.single().boundingBox.right, 0.001f)
    }

    @Test
    fun `pairs separate regression and class heads by grid`() {
        val regression = FloatArray(8) { if (it % 2 == 0) -10f else 10f }
        val classification = floatArrayOf(0.8f)

        val result = YoloDflDecoder.decode(
            listOf(
                tensor("box", intArrayOf(1, 8, 1, 1), regression),
                tensor("score", intArrayOf(1, 1, 1, 1), classification),
            ),
            image(), config(),
        )

        assertEquals(1, result.size)
        assertEquals(0.8f, result.single().categories.single().score, 0.001f)
    }

    @Test
    fun `preserves multiple classes from dfl head`() {
        val data = FloatArray(10)
        repeat(4) { side -> data[side * 2 + 1] = 10f }
        data[8] = 0.9f
        data[9] = 0.7f
        val multiConfig = config().copy(labels = listOf("a", "b"), multiLabel = true)

        val result = YoloDflDecoder.decode(
            listOf(tensor("head", intArrayOf(1, 10, 1, 1), data)), image(), multiConfig,
        )

        assertEquals(listOf("a", "b"), result.single().categories.map { it.name })
    }

    private fun config() = RknnModelConfig(
        "model", RknnModelType.OBJECT_DETECTOR, "model.rknn", 640, 640,
        labels = listOf("person"), scoreThreshold = 0.25f,
    )

    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)

    private fun tensor(name: String, dims: IntArray, data: FloatArray) =
        NativeTensorOutput(0, name, dims, 0, 0, data)
}
