package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class MediaPipeSsdDecoderTest {
    @Test
    fun `all supported model variants generate the expected number of anchors`() {
        assertEquals(12_276, MediaPipeObjectDetectorModel.MOBILENET_V2.anchorCount)
        assertEquals(19_125, MediaPipeObjectDetectorModel.MOBILENET_V2_I320.anchorCount)
        assertEquals(12_276, MediaPipeObjectDetectorModel.MOBILENET_MULTI_AVG.anchorCount)
        assertEquals(27_621, MediaPipeObjectDetectorModel.MOBILENET_MULTI_AVG_I384.anchorCount)
    }

    @Test
    fun `I384 anchor 6255 matches MediaPipe Model Maker anchor`() {
        val anchor = MediaPipeSsdDecoder.generateAnchors(
            MediaPipeObjectDetectorModel.MOBILENET_MULTI_AVG_I384,
        )[6255]

        assertEquals(99.029434f, anchor.centerY - anchor.height / 2f, 0.0001f)
        assertEquals(179.514725f, anchor.centerX - anchor.width / 2f, 0.0001f)
        assertEquals(132.970566f, anchor.centerY + anchor.height / 2f, 0.0001f)
        assertEquals(196.485275f, anchor.centerX + anchor.width / 2f, 0.0001f)
    }

    @Test
    fun `background is ignored and foreground class index is shifted`() {
        val model = MediaPipeObjectDetectorModel.MOBILENET_V2
        val boxes = FloatArray(model.anchorCount * 4)
        val scores = FloatArray(model.anchorCount * 3)
        scores[0] = 0.99f
        scores[2] = 0.8f

        val detections = MediaPipeSsdDecoder.decode(
            tensor(boxes, intArrayOf(1, model.anchorCount, 4)),
            tensor(scores, intArrayOf(1, model.anchorCount, 3)),
            image(256),
            config(model),
        )

        assertEquals(1, detections.size)
        assertEquals(1, detections.single().categories.single().index)
        assertEquals("second", detections.single().categories.single().name)
    }

    @Test
    fun `explicit model rejects a mismatched input size`() {
        val model = MediaPipeObjectDetectorModel.MOBILENET_V2
        assertThrows(IllegalArgumentException::class.java) {
            MediaPipeSsdDecoder.decode(
                tensor(FloatArray(model.anchorCount * 4), intArrayOf(1, model.anchorCount, 4)),
                tensor(FloatArray(model.anchorCount * 3), intArrayOf(1, model.anchorCount, 3)),
                image(320),
                config(model).copy(inputWidth = 320, inputHeight = 320),
            )
        }
    }

    private fun config(model: MediaPipeObjectDetectorModel) = RknnModelConfig(
        id = "test",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "test.rknn",
        inputWidth = model.inputSize,
        inputHeight = model.inputSize,
        labels = listOf("first", "second"),
        scoreThreshold = 0.4f,
        mediaPipeModel = model,
    )

    private fun image(size: Int) = RknnImage(
        width = size,
        height = size,
        channels = 3,
        bytes = ByteArray(size * size * 3),
        originalWidth = size,
        originalHeight = size,
        scale = 1f,
        paddingLeft = 0,
        paddingTop = 0,
    )

    private fun tensor(data: FloatArray, dims: IntArray) = NativeTensorOutput(
        index = 0,
        name = "output",
        dims = dims,
        type = 0,
        format = 0,
        data = data,
    )
}
