package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class YoloTensorViewTest {
    @Test
    fun `reads candidate first tensor`() {
        val view = YoloTensorView.create(
            tensor(intArrayOf(1, 2, 6), FloatArray(12) { it.toFloat() }),
            expectedChannels = setOf(6),
        )

        assertEquals(2, view.rows)
        assertEquals(6, view.channels)
        assertEquals(4f, view[0, 4])
        assertEquals(10f, view[1, 4])
    }

    @Test
    fun `reads channel first tensor`() {
        val channelFirst = floatArrayOf(
            0f, 6f, 1f, 7f, 2f, 8f, 3f, 9f, 4f, 10f, 5f, 11f,
        )
        val view = YoloTensorView.create(
            tensor(intArrayOf(1, 6, 2), channelFirst),
            expectedChannels = setOf(6),
        )

        assertEquals(2, view.rows)
        assertEquals(6, view.channels)
        assertEquals(4f, view[0, 4])
        assertEquals(10f, view[1, 4])
    }

    @Test
    fun `rejects ambiguous layout`() {
        assertThrows(IllegalArgumentException::class.java) {
            YoloTensorView.create(
                tensor(intArrayOf(1, 6, 6), FloatArray(36)),
                expectedChannels = setOf(6),
            )
        }
    }

    @Test
    fun `rejects inconsistent data length`() {
        assertThrows(IllegalArgumentException::class.java) {
            YoloTensorView.create(
                tensor(intArrayOf(1, 2, 6), FloatArray(11)),
                expectedChannels = setOf(6),
            )
        }
    }

    private fun tensor(dims: IntArray, data: FloatArray) = NativeTensorOutput(
        index = 0,
        name = "output0",
        dims = dims,
        type = 0,
        format = 0,
        data = data,
    )
}
