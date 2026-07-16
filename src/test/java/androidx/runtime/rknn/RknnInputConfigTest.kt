package androidx.runtime.rknn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class RknnInputConfigTest {
    @Test
    fun `defaults preserve automatic input selection`() {
        val config = model()

        assertEquals(RknnInputType.AUTO, config.inputType)
        assertEquals(RknnInputLayout.AUTO, config.inputLayout)
        assertEquals(listOf(0f, 0f, 0f), config.normalization.mean.toList())
        assertEquals(listOf(255f, 255f, 255f), config.normalization.std.toList())
    }

    @Test
    fun `normalization requires three nonzero channel divisors`() {
        assertThrows(IllegalArgumentException::class.java) {
            RknnNormalization(floatArrayOf(0f), floatArrayOf(255f))
        }
        assertThrows(IllegalArgumentException::class.java) {
            RknnNormalization(floatArrayOf(0f, 0f, 0f), floatArrayOf(255f, 0f, 255f))
        }
    }

    private fun model() = RknnModelConfig(
        id = "action",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "action.rknn",
        inputWidth = 640,
        inputHeight = 640,
    )
}
