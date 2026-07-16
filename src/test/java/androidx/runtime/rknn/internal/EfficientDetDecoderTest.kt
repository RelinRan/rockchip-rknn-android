package androidx.runtime.rknn.internal

import androidx.runtime.rknn.decoder.EfficientDetDecoder
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EfficientDetDecoderTest {
    private val config = RknnModelConfig(
        id = "action",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "action.rknn",
        inputWidth = 384,
        inputHeight = 384,
        labels = listOf("1X", "2L"),
        scoreThreshold = 0.4f,
        maxResults = 10,
    )
    private val image = RknnImage(
        width = 384,
        height = 384,
        channels = 3,
        bytes = ByteArray(384 * 384 * 3),
        originalWidth = 384,
        originalHeight = 384,
        scale = 1f,
        paddingLeft = 0,
        paddingTop = 0,
    )

    @Test
    fun `decodes efficientdet anchors and skips background class`() {
        val boxes = FloatArray(27_621 * 4)
        val scores = FloatArray(27_621 * 3) { -20f }
        scores[1] = 8f

        val detections = EfficientDetDecoder.decode(
            boxes = tensor(0, intArrayOf(1, 27_621, 4, 1), boxes),
            scores = tensor(1, intArrayOf(1, 27_621, 3, 1), scores),
            image = image,
            config = config,
        )

        assertEquals(1, detections.size)
        assertEquals("1X", detections.single().categories.single().name)
        assertTrue(detections.single().categories.single().score > 0.99f)
        assertEquals(0f, detections.single().boundingBox.left, 0.01f)
        assertEquals(12.49f, detections.single().boundingBox.right, 0.01f)
    }

    @Test
    fun `suppresses overlapping detections of the same class`() {
        val boxes = FloatArray(27_621 * 4)
        val scores = FloatArray(27_621 * 3) { -20f }
        scores[1] = 8f
        scores[4] = 7f

        val detections = EfficientDetDecoder.decode(
            boxes = tensor(0, intArrayOf(1, 27_621, 4, 1), boxes),
            scores = tensor(1, intArrayOf(1, 27_621, 3, 1), scores),
            image = image,
            config = config,
        )

        assertEquals(1, detections.size)
    }

    @Test
    fun `does not apply sigmoid when model already returns probabilities`() {
        val boxes = FloatArray(27_621 * 4)
        val scores = FloatArray(27_621 * 3)
        scores[1] = 0.39f

        val detections = EfficientDetDecoder.decode(
            boxes = tensor(0, intArrayOf(1, 27_621, 4, 1), boxes),
            scores = tensor(1, intArrayOf(1, 27_621, 3, 1), scores),
            image = image,
            config = config,
        )

        assertTrue(detections.isEmpty())
    }

    private fun tensor(index: Int, dims: IntArray, data: FloatArray) = NativeTensorOutput(
        index = index,
        name = "output_$index",
        dims = dims,
        type = 0,
        format = 0,
        data = data,
    )
}
