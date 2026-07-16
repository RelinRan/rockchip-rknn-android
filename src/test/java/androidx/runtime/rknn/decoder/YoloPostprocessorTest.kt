package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import org.junit.Assert.assertEquals
import org.junit.Test
import androidx.runtime.rknn.data.RknnOrientedBox

class YoloPostprocessorTest {
    @Test
    fun `suppresses overlapping boxes only within same class`() {
        val candidates = listOf(
            candidate(0, 0.9f, 10f, 10f, 110f, 110f),
            candidate(0, 0.8f, 12f, 12f, 108f, 108f),
            candidate(1, 0.7f, 12f, 12f, 108f, 108f),
        )

        val result = YoloPostprocessor.process(candidates, image(), config(), applyNms = true)

        assertEquals(2, result.size)
        assertEquals(listOf(0, 1), result.map { it.categories.single().index })
    }

    @Test
    fun `restores letterbox coordinates and limits results`() {
        val result = YoloPostprocessor.process(
            listOf(
                candidate(0, 0.9f, 20f, 30f, 220f, 230f),
                candidate(1, 0.8f, 40f, 50f, 80f, 90f),
            ),
            image(),
            config(maxResults = 1),
            applyNms = false,
        )

        assertEquals(1, result.size)
        assertEquals(5f, result.single().boundingBox.left, 0.001f)
        assertEquals(10f, result.single().boundingBox.top, 0.001f)
        assertEquals(105f, result.single().boundingBox.right, 0.001f)
    }

    @Test
    fun `uses rotated overlap for oriented box nms`() {
        val firstBox = RknnOrientedBox(100f, 100f, 160f, 20f, (Math.PI / 4).toFloat())
        val secondBox = RknnOrientedBox(100f, 100f, 160f, 20f, (-Math.PI / 4).toFloat())
        val firstBounds = firstBox.boundingBox()
        val secondBounds = secondBox.boundingBox()
        val horizontal = YoloCandidate(
            firstBounds.left, firstBounds.top, firstBounds.right, firstBounds.bottom, 0, 0.9f,
            orientedBox = firstBox,
        )
        val vertical = YoloCandidate(
            secondBounds.left, secondBounds.top, secondBounds.right, secondBounds.bottom, 0, 0.8f,
            orientedBox = secondBox,
        )

        val result = YoloPostprocessor.process(
            listOf(horizontal, vertical), image(), config().copy(nmsThreshold = 0.3f), applyNms = true,
        )

        assertEquals(2, result.size)
    }

    private fun candidate(classId: Int, score: Float, l: Float, t: Float, r: Float, b: Float) =
        YoloCandidate(l, t, r, b, classId, score)

    private fun image() = RknnImage(640, 640, 3, ByteArray(1), 320, 300, 2f, 10, 10)

    private fun config(maxResults: Int = 100) = RknnModelConfig(
        "model", RknnModelType.OBJECT_DETECTOR, "model.rknn", 640, 640,
        labels = listOf("a", "b"), maxResults = maxResults, nmsThreshold = 0.5f,
    )
}
