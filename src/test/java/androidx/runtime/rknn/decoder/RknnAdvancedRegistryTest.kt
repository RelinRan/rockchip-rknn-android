package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.internal.NativeTensorOutput
import org.junit.Assert.assertEquals
import org.junit.Test

class RknnAdvancedRegistryTest {
    @Test
    fun `auto selects segmentation from model type`() {
        assertEquals(RknnDecoderType.YOLO_SEGMENT, resolve(RknnModelType.IMAGE_SEGMENTER, listOf(tensor(6))))
    }

    @Test
    fun `auto selects obb from model type before six channel detection`() {
        assertEquals(RknnDecoderType.YOLO_OBB, resolve(RknnModelType.OBB_DETECTOR, listOf(tensor(6))))
    }

    @Test
    fun `auto selects pose heads from rank four branches`() {
        val outputs = listOf(head(8), head(1), head(51))
        assertEquals(RknnDecoderType.YOLO_POSE_HEADS, resolve(RknnModelType.POSE_DETECTOR, outputs))
    }

    @Test
    fun `explicit classification decoder is preserved`() {
        assertEquals(
            RknnDecoderType.YOLO_CLASSIFY,
            RknnDecoderRegistry.resolve(RknnDecoderType.YOLO_CLASSIFY, listOf(tensor(3)), config(RknnModelType.IMAGE_CLASSIFIER)),
        )
    }

    private fun resolve(type: RknnModelType, outputs: List<NativeTensorOutput>) =
        RknnDecoderRegistry.resolve(RknnDecoderType.AUTO, outputs, config(type))

    private fun config(type: RknnModelType) = RknnModelConfig(
        "model", type, "model.rknn", 640, 640, labels = listOf("person"),
    )

    private fun tensor(channels: Int) = NativeTensorOutput(
        0, "output", intArrayOf(1, 1, channels), 0, 0, FloatArray(channels),
    )

    private fun head(channels: Int) = NativeTensorOutput(
        0, "head_$channels", intArrayOf(1, channels, 1, 1), 0, 0, FloatArray(channels),
    )
}
