package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloClassificationDecoderTest {
    @Test
    fun `decodes probabilities with top k`() {
        val result = YoloClassificationDecoder.decode(
            tensor(floatArrayOf(0.1f, 0.7f, 0.2f)), config(maxResults = 2),
        )

        assertEquals(listOf("b", "c"), result.map { it.name })
    }

    @Test
    fun `applies softmax to logits`() {
        val result = YoloClassificationDecoder.decode(
            tensor(floatArrayOf(1f, 3f, 2f)),
            config(scoreType = ClassificationScoreType.LOGITS),
        )

        assertEquals("b", result.first().name)
        assertEquals(0.66524f, result.first().score, 0.0001f)
    }

    private fun config(
        maxResults: Int = 3,
        scoreType: ClassificationScoreType = ClassificationScoreType.AUTO,
    ) = RknnModelConfig(
        "classify", RknnModelType.IMAGE_CLASSIFIER, "classify.rknn", 224, 224,
        labels = listOf("a", "b", "c"), scoreThreshold = 0f, maxResults = maxResults,
        classifierScoreType = scoreType,
    )

    private fun tensor(data: FloatArray) = NativeTensorOutput(
        0, "output", intArrayOf(1, data.size), 0, 0, data,
    )
}
