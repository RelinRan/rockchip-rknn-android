package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloPoseHeadsDecoderTest {
    @Test
    fun `decodes matched dfl class and key point heads`() {
        val regression = FloatArray(8) { if (it % 2 == 0) -10f else 10f }
        val classes = floatArrayOf(0.9f)
        val points = floatArrayOf(0.5f, 0.5f, 10f, 0.25f, 0.25f, -10f)

        val result = YoloPoseHeadsDecoder.decode(
            listOf(
                tensor("box", 8, regression), tensor("score", 1, classes), tensor("pose", 6, points),
            ), image(), config(),
        )

        assertEquals(1, result.size)
        assertEquals(2, result.single().keyPoints.size)
        assertEquals(0, result.single().keyPoints.first().index)
    }

    private fun config() = RknnModelConfig(
        "pose", RknnModelType.POSE_DETECTOR, "pose.rknn", 640, 640,
        labels = listOf("person"), poseKeyPointCount = 2,
    )
    private fun image() = RknnImage(640, 640, 3, ByteArray(0), 640, 640, 1f, 0, 0)
    private fun tensor(name: String, channels: Int, data: FloatArray) =
        NativeTensorOutput(0, name, intArrayOf(1, channels, 1, 1), 0, 0, data)
}
