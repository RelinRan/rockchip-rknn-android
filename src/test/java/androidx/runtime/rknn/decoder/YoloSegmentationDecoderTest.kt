package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloSegmentationDecoderTest {
    @Test
    fun `combines nchw prototype with mask coefficients`() {
        val detection = tensor("detect", intArrayOf(1, 1, 6), floatArrayOf(320f, 320f, 640f, 640f, 0.9f, 1f))
        val prototype = tensor("proto", intArrayOf(1, 1, 2, 2), floatArrayOf(10f, -10f, -10f, 10f))

        val result = YoloSegmentationDecoder.decode(listOf(detection, prototype), image(), config())

        val mask = result.single().segmentationMask!!
        assertEquals(2, mask.width)
        assertEquals(2, mask.height)
        assertTrue(mask.probabilities[0] > 0.99f)
        assertTrue(mask.probabilities[1] < 0.01f)
    }

    @Test
    fun `supports nhwc prototype`() {
        val detection = tensor("detect", intArrayOf(1, 1, 6), floatArrayOf(320f, 320f, 640f, 640f, 0.9f, 1f))
        val prototype = tensor("proto", intArrayOf(1, 2, 2, 1), floatArrayOf(10f, -10f, -10f, 10f), format = 1)

        val result = YoloSegmentationDecoder.decode(listOf(detection, prototype), image(), config())

        assertTrue(result.single().segmentationMask!!.probabilities[0] > 0.99f)
    }

    @Test
    fun `clears prototype pixels outside detection box`() {
        val detection = tensor("detect", intArrayOf(1, 1, 6), floatArrayOf(160f, 320f, 320f, 640f, 0.9f, 1f))
        val prototype = tensor("proto", intArrayOf(1, 1, 2, 2), FloatArray(4) { 10f })

        val mask = YoloSegmentationDecoder.decode(listOf(detection, prototype), image(), config())
            .single().segmentationMask!!

        assertTrue(mask.probabilities[0] > 0.99f)
        assertEquals(0f, mask.probabilities[1], 0f)
        assertEquals(0f, mask.probabilities[3], 0f)
    }

    private fun config() = RknnModelConfig(
        "segment", RknnModelType.OBJECT_DETECTOR, "segment.rknn", 640, 640,
        labels = listOf("person"), scoreThreshold = 0.25f,
    )
    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
    private fun tensor(name: String, dims: IntArray, data: FloatArray, format: Int = 0) =
        NativeTensorOutput(0, name, dims, 0, format, data)
}
