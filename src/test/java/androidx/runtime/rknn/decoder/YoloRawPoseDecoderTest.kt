package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloRawPoseDecoderTest {
    @Test
    fun `decodes raw pose with objectness`() {
        val values = FloatArray(5 + 1 + 2 * 3)
        values[0] = 100f
        values[1] = 100f
        values[2] = 40f
        values[3] = 20f
        values[4] = 0.5f
        values[5] = 0.8f
        values[6] = 90f
        values[7] = 95f
        values[8] = 0.7f

        val result = YoloRawPoseDecoder.decode(
            tensor(intArrayOf(1, 1, values.size), values), image(), config(),
        )

        assertEquals(1, result.size)
        assertEquals(0.4f, result.single().categories.single().score, 0.0001f)
        assertEquals(2, result.single().keyPoints.size)
        assertEquals(90f, result.single().keyPoints.first().x, 0.001f)
    }

    private fun config() = RknnModelConfig(
        "pose", RknnModelType.POSE_DETECTOR, "pose.rknn", 640, 640,
        labels = listOf("person"), scoreThreshold = 0.25f, poseKeyPointCount = 2,
    )

    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
    private fun tensor(dims: IntArray, data: FloatArray) = NativeTensorOutput(0, "pose", dims, 0, 0, data)
}
