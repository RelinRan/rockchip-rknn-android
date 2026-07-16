package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageClassifierDecoderTest {
    @Test
    fun `four EfficientNet Lite variants use official input size and quantization`() {
        EfficientNetLiteModel.entries.forEach { model ->
            assertEquals(224, model.inputSize)
        }
        assertEquals(false, EfficientNetLiteModel.LITE0_FLOAT32.quantized)
        assertEquals(true, EfficientNetLiteModel.LITE0_INT8.quantized)
        assertEquals(false, EfficientNetLiteModel.LITE2_FLOAT32.quantized)
        assertEquals(true, EfficientNetLiteModel.LITE2_INT8.quantized)
    }

    @Test
    fun `keeps probability output and returns top K categories`() {
        val result = ImageClassifierDecoder.decode(
            tensor(floatArrayOf(0.1f, 0.7f, 0.2f)),
            config(maxResults = 2),
        )

        assertEquals(2, result.size)
        assertEquals("two", result[0].name)
        assertEquals(0.7f, result[0].score, 0.0001f)
        assertEquals("three", result[1].name)
    }

    @Test
    fun `applies softmax to logits`() {
        val result = ImageClassifierDecoder.decode(
            tensor(floatArrayOf(1f, 3f, 2f)),
            config(scoreType = ClassificationScoreType.LOGITS),
        )

        assertEquals("two", result.first().name)
        assertEquals(0.66524f, result.first().score, 0.0001f)
    }

    @Test
    fun `filters scores below threshold`() {
        val result = ImageClassifierDecoder.decode(
            tensor(floatArrayOf(0.2f, 0.3f, 0.5f)),
            config(threshold = 0.6f),
        )

        assertTrue(result.isEmpty())
    }

    private fun config(
        threshold: Float = 0f,
        maxResults: Int = 3,
        scoreType: ClassificationScoreType = ClassificationScoreType.AUTO,
    ) = RknnModelConfig(
        id = "classifier",
        type = RknnModelType.IMAGE_CLASSIFIER,
        fileName = "classifier.rknn",
        inputWidth = 224,
        inputHeight = 224,
        labels = listOf("one", "two", "three"),
        scoreThreshold = threshold,
        maxResults = maxResults,
        classifierScoreType = scoreType,
    )

    private fun tensor(data: FloatArray) = NativeTensorOutput(
        index = 0,
        name = "probability",
        dims = intArrayOf(1, data.size),
        type = 0,
        format = 0,
        data = data,
    )
}
