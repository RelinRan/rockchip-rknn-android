package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloPoseDecoderTest {
    @Test
    fun `decodes end to end box and 17 key points`() {
        val row = FloatArray(57)
        row[0] = 100f
        row[1] = 120f
        row[2] = 300f
        row[3] = 400f
        row[4] = 0.9f
        row[5] = 0f
        row[6] = 160f
        row[7] = 180f
        row[8] = 0.8f

        val detections = YoloPoseDecoder.decode(tensor(row), image(), config())

        assertEquals(1, detections.size)
        assertEquals("person", detections.single().categories.single().name)
        assertEquals(17, detections.single().keyPoints.size)
        assertEquals("nose", detections.single().keyPoints.first().name)
        assertEquals(160f, detections.single().keyPoints.first().x, 0.001f)
        assertEquals(180f, detections.single().keyPoints.first().y, 0.001f)
        assertEquals(0.8f, detections.single().keyPoints.first().score, 0.001f)
    }

    @Test
    fun `auto registry recognizes 57 value pose rows`() {
        val output = tensor(FloatArray(300 * 57), intArrayOf(1, 300, 57))
        assertEquals(
            RknnDecoderType.YOLO_POSE_LANDMARK,
            RknnDecoderRegistry.resolve(RknnDecoderType.AUTO, listOf(output), config()),
        )
    }

    @Test
    fun `decodes transposed pose rows`() {
        val values = FloatArray(57)
        values[0] = 100f
        values[1] = 120f
        values[2] = 300f
        values[3] = 400f
        values[4] = 0.9f
        values[5] = 0f
        values[6] = 160f
        values[7] = 180f
        values[8] = 0.8f

        val result = YoloPoseDecoder.decode(tensor(values, intArrayOf(1, 57, 1)), image(), config())

        assertEquals(1, result.size)
        assertEquals(17, result.single().keyPoints.size)
    }

    @Test
    fun `supports configured key point count`() {
        val values = FloatArray(6 + 5 * 3)
        values[0] = 10f
        values[1] = 10f
        values[2] = 100f
        values[3] = 100f
        values[4] = 0.9f
        values[5] = 0f

        val result = YoloPoseDecoder.decode(
            tensor(values, intArrayOf(1, 1, values.size)), image(),
            config().copy(poseKeyPointCount = 5),
        )

        assertEquals(5, result.single().keyPoints.size)
    }

    private fun config() = RknnModelConfig(
        id = "pose",
        type = RknnModelType.POSE_DETECTOR,
        fileName = "pose.rknn",
        inputWidth = 640,
        inputHeight = 640,
        labels = listOf("person"),
        scoreThreshold = 0.5f,
    )

    private fun image() = RknnImage(
        640, 640, 3, ByteArray(640 * 640 * 3), 640, 640, 1f, 0, 0,
    )

    private fun tensor(data: FloatArray, dims: IntArray = intArrayOf(1, 1, 57)) = NativeTensorOutput(
        0, "output0", dims, 1, 3, data,
    )
}
