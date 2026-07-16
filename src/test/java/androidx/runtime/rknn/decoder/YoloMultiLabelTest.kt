package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloMultiLabelTest {
    @Test
    fun `returns all passing categories when enabled`() {
        val result = RawYoloDecoder.decode(tensor(), image(), config(multiLabel = true))

        assertEquals(listOf("a", "b"), result.single().categories.map { it.name })
        assertEquals(listOf(0.9f, 0.7f), result.single().categories.map { it.score })
    }

    @Test
    fun `returns highest category when disabled`() {
        val result = RawYoloDecoder.decode(tensor(), image(), config(multiLabel = false))

        assertEquals(listOf("a"), result.single().categories.map { it.name })
    }

    private fun tensor() = NativeTensorOutput(
        0, "output", intArrayOf(1, 1, 6), 0, 0,
        floatArrayOf(100f, 100f, 20f, 20f, 0.9f, 0.7f),
    )

    private fun config(multiLabel: Boolean) = RknnModelConfig(
        "model", RknnModelType.OBJECT_DETECTOR, "model.rknn", 640, 640,
        labels = listOf("a", "b"), scoreThreshold = 0.5f, multiLabel = multiLabel,
    )

    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
}
