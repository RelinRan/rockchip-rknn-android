package androidx.runtime.rknn.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class RknnAdvancedDataTest {
    @Test
    fun `thresholds and resizes segmentation mask`() {
        val mask = RknnSegmentationMask(2, 2, floatArrayOf(0.1f, 0.8f, 0.6f, 0.2f))

        assertArrayEquals(byteArrayOf(0, 1, 1, 0), mask.toBinary(0.5f))
        assertArrayEquals(
            floatArrayOf(0.1f, 0.1f, 0.8f, 0.8f),
            mask.resize(4, 1).probabilities,
            0.0001f,
        )
    }

    @Test
    fun `computes oriented box corners and bounds`() {
        val box = RknnOrientedBox(50f, 50f, 40f, 20f, 0f)

        assertEquals(4, box.corners.size)
        assertEquals(RknnBoundingBox(30f, 40f, 70f, 60f), box.boundingBox())
    }
}
