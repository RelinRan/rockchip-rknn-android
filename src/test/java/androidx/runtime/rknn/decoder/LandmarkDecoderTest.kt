package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LandmarkDecoderTest {
    @Test
    fun `pose variants share official input and output contract`() {
        PoseLandmarkModel.entries.forEach { assertEquals(256, it.inputSize) }
    }

    @Test
    fun `decodes first 33 pose landmarks world landmarks and mask`() {
        val landmarks = FloatArray(39 * 5)
        landmarks[0] = 128f
        landmarks[1] = 64f
        landmarks[2] = -25.6f
        landmarks[3] = 2f
        landmarks[4] = -2f
        val world = FloatArray(39 * 3)
        world[0] = 0.1f
        world[1] = 0.2f
        world[2] = -0.3f
        val result = PoseLandmarkDecoder.decode(
            listOf(
                tensor(0, landmarks),
                tensor(1, floatArrayOf(0.8f)),
                tensor(2, FloatArray(256 * 256), intArrayOf(1, 1, 256, 256)),
                tensor(3, FloatArray(64 * 64 * 39)),
                tensor(4, world),
            ),
            poseConfig(),
        )

        assertEquals(33, result.landmarks.size)
        assertEquals(33, result.worldLandmarks.size)
        assertEquals(0.5f, result.landmarks[0].x, 0.0001f)
        assertEquals(0.25f, result.landmarks[0].y, 0.0001f)
        assertEquals(-0.1f, result.landmarks[0].z, 0.0001f)
        assertTrue(result.landmarks[0].visibility > 0.88f)
        assertTrue(result.landmarks[0].presence < 0.12f)
        assertEquals(0.1f, result.worldLandmarks[0].x, 0.0001f)
        assertEquals(0.8f, result.posePresence, 0.0001f)
        assertEquals(256 * 256, result.segmentationMask?.size)
        assertEquals(256, result.segmentationMaskWidth)
        assertEquals(256, result.segmentationMaskHeight)
    }

    @Test
    fun `decodes hand landmarks presence handedness and world coordinates`() {
        val landmarks = FloatArray(21 * 3)
        landmarks[0] = 112f
        landmarks[1] = 56f
        landmarks[2] = -22.4f
        val world = FloatArray(21 * 3)
        world[0] = 0.01f
        val result = HandLandmarkDecoder.decode(
            listOf(
                tensor(0, landmarks),
                tensor(1, floatArrayOf(0.9f)),
                tensor(2, floatArrayOf(0.8f)),
                tensor(3, world),
            ),
            handConfig(),
        )

        assertEquals(21, result.landmarks.size)
        assertEquals(0.5f, result.landmarks[0].x, 0.0001f)
        assertEquals(0.25f, result.landmarks[0].y, 0.0001f)
        assertEquals(-0.1f, result.landmarks[0].z, 0.0001f)
        assertEquals(0.9f, result.handPresence, 0.0001f)
        assertEquals("Right", result.handedness.name)
        assertEquals(0.8f, result.handedness.score, 0.0001f)
        assertEquals(0.01f, result.worldLandmarks[0].x, 0.0001f)
    }

    private fun poseConfig() = RknnModelConfig(
        "pose", RknnModelType.POSE_LANDMARKER, "pose.rknn", 256, 256,
        poseLandmarkModel = PoseLandmarkModel.LITE,
    )

    private fun handConfig() = RknnModelConfig(
        "hand", RknnModelType.HAND_LANDMARKER, "hand.rknn", 224, 224,
        handLandmarkModel = HandLandmarkModel.FULL,
    )

    private fun tensor(index: Int, data: FloatArray, dims: IntArray = intArrayOf(1, data.size)) =
        NativeTensorOutput(index, "Identity${if (index == 0) "" else "_$index"}", dims, 0, 0, data)
}
