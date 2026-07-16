package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class RknnDecoderRegistryTest {
    @Test
    fun `auto selects image classifier from model type`() {
        assertAutoDecoder(RknnModelType.IMAGE_CLASSIFIER, RknnDecoderType.MEDIA_PIPE_IMAGE_CLASSIFIER)
    }

    @Test
    fun `auto selects pose landmark from model type`() {
        assertAutoDecoder(RknnModelType.POSE_LANDMARKER, RknnDecoderType.MEDIA_PIPE_POSE_LANDMARK)
    }

    @Test
    fun `auto selects hand landmark from model type`() {
        assertAutoDecoder(RknnModelType.HAND_LANDMARKER, RknnDecoderType.MEDIA_PIPE_HAND_LANDMARK)
    }

    @Test
    fun `auto selects yolo pose landmark from 57 value rows`() {
        assertEquals(
            RknnDecoderType.YOLO_POSE_LANDMARK,
            RknnDecoderRegistry.resolve(
                RknnDecoderType.AUTO,
                listOf(tensor(intArrayOf(1, 300, 57))),
                config(RknnModelType.POSE_DETECTOR),
            ),
        )
    }

    @Test
    fun `auto selects transposed end to end output`() {
        assertEquals(
            RknnDecoderType.YOLO_END_TO_END,
            RknnDecoderRegistry.resolve(
                RknnDecoderType.AUTO,
                listOf(tensor(intArrayOf(1, 6, 300))),
                config(RknnModelType.OBJECT_DETECTOR),
            ),
        )
    }

    @Test
    fun `auto selects raw output with objectness`() {
        assertEquals(
            RknnDecoderType.YOLO_DETECT_RAW,
            RknnDecoderRegistry.resolve(
                RknnDecoderType.AUTO,
                listOf(tensor(intArrayOf(1, 8400, 7))),
                config(RknnModelType.OBJECT_DETECTOR).copy(labels = listOf("person", "other")),
            ),
        )
    }

    @Test
    fun `auto selects rank four dfl heads`() {
        assertEquals(
            RknnDecoderType.YOLO_DETECT_HEADS,
            RknnDecoderRegistry.resolve(
                RknnDecoderType.AUTO,
                listOf(tensor(intArrayOf(1, 9, 80, 80))),
                config(RknnModelType.OBJECT_DETECTOR),
            ),
        )
    }

    private fun assertAutoDecoder(modelType: RknnModelType, expected: RknnDecoderType) {
        assertEquals(
            expected,
            RknnDecoderRegistry.resolve(
                RknnDecoderType.AUTO,
                listOf(tensor(intArrayOf(1, 57))),
                config(modelType),
            ),
        )
    }

    private fun config(type: RknnModelType) = RknnModelConfig(
        id = "model",
        type = type,
        fileName = "model.rknn",
        inputWidth = 640,
        inputHeight = 640,
        labels = listOf("person"),
    )

    private fun tensor(dims: IntArray) = NativeTensorOutput(
        index = 0,
        name = "output0",
        dims = dims,
        type = 0,
        format = 0,
        data = FloatArray(dims.reduce(Int::times)),
    )
}
