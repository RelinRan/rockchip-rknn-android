package androidx.runtime.rknn.model

import android.content.Context
import android.graphics.Bitmap
import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnEmbeddingResult
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.RknnOptions
import androidx.runtime.rknn.RknnState
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelApiEmbeddingTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun initialize_registersReIdModelWithoutLabels() {
        temporaryFolder.newFile("osnet.rknn")
        val runtime = FakeEmbeddingRuntime()
        val api = ModelApi(runtime)

        val state = api.initializeModelRoot(
            temporaryFolder.root,
            ModelConfig(
                mapOf(
                    ModelKey.REID to DetectorModel(
                        fileName = "osnet.rknn",
                        inputWidth = 128,
                        inputHeight = 256,
                        modelType = RknnModelType.REID_EMBEDDING,
                    ),
                ),
            ),
            RknnState.READY,
        )

        assertTrue(state.readiness(ModelKey.REID).ready)
        assertEquals(RknnModelType.REID_EMBEDDING, runtime.registered.single().type)
        assertTrue(runtime.registered.single().labels.isEmpty())
    }

    @Test
    fun detect_dispatchesReIdModelAndPublishesEmbedding() {
        temporaryFolder.newFile("osnet.rknn")
        val runtime = FakeEmbeddingRuntime()
        val api = ModelApi(runtime)
        api.initializeModelRoot(
            temporaryFolder.root,
            ModelConfig(
                mapOf(
                    ModelKey.REID to DetectorModel(
                        fileName = "osnet.rknn",
                        modelType = RknnModelType.REID_EMBEDDING,
                    ),
                ),
            ),
            RknnState.READY,
        )

        val result = api.execute(ModelKey.REID) {
            RknnEmbeddingResult(
                success = true,
                backend = RknnBackend.ROCKCHIP_RKNN,
                modelId = ModelKey.REID.value,
                embedding = floatArrayOf(0.6f, 0.8f),
                durationMs = 1,
            )
        }

        assertTrue(result is RknnEmbeddingResult)
        assertEquals(result, api.result(ModelKey.REID).value)
        assertTrue(result.resultSequence > 0)
    }

    @Test
    fun embeddingResult_rejectsNonFiniteOrNonNormalizedData() {
        val result = RknnEmbeddingResult.normalized(
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = "osnet",
            values = floatArrayOf(3f, 4f),
            expectedSize = 2,
            durationMs = 2,
        )

        assertTrue(result.success)
        assertEquals(0.6f, result.embedding[0], 0.0001f)
        assertEquals(0.8f, result.embedding[1], 0.0001f)

        val invalid = RknnEmbeddingResult.normalized(
            backend = RknnBackend.ROCKCHIP_RKNN,
            modelId = "osnet",
            values = floatArrayOf(Float.NaN),
            expectedSize = 2,
            durationMs = 2,
        )
        assertFalse(invalid.success)
        assertTrue(invalid.message.orEmpty().contains("Expected 2"))
    }
}

private class FakeEmbeddingRuntime : ModelRuntime {
    val registered = mutableListOf<RknnModelConfig>()

    override fun initialize(context: Context, options: RknnOptions): RknnState = RknnState.READY

    override fun registerModel(config: RknnModelConfig): Boolean {
        registered += config
        return true
    }

    override fun unregisterModel(modelId: String) = Unit

    override fun release() = Unit

    override fun extractEmbedding(modelId: String, bitmap: Bitmap): RknnEmbeddingResult =
        error("Not used by these unit tests")
}
