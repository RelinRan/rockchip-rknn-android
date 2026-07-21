package androidx.runtime.rknn.model

import androidx.runtime.rknn.RknnBackend
import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnInputLayout
import androidx.runtime.rknn.RknnInputType
import androidx.runtime.rknn.RknnNormalization
import androidx.runtime.rknn.RknnState
import androidx.runtime.rknn.data.RknnObjectDetectionResult
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelApiTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun detectForRole_beforeInitializationReportsUninitialized() {
        val sdk = ModelApi(FakeModelRuntime())

        val result = sdk.executeDetection(ModelKey.TARGET) { success("target") }

        assertFalse(result.success)
        assertTrue(result.message.orEmpty().contains("not initialized"))
        assertNull(sdk.result(ModelKey.TARGET).value)
    }

    @Test
    fun initialize_prefersDirectLabelsOverLabelFile() {
        createModel("target.rknn")
        temporaryFolder.newFile("target.labels").writeText("file-label\n")
        val runtime = FakeModelRuntime()
        val sdk = ModelApi(runtime)

        sdk.initializeModelRoot(
            temporaryFolder.root,
            config(target = DetectorModel(fileName = "target.rknn", labels = listOf("direct-label"), labelFileName = "target.labels")),
            RknnState.READY,
        )

        assertEquals(listOf("direct-label"), runtime.registered.single().labels)
    }

    @Test
    fun initialize_passesDetectorInputConfigurationToRegisteredModel() {
        createModel("target.rknn")
        val runtime = FakeModelRuntime()
        val sdk = ModelApi(runtime)
        val normalization = RknnNormalization(
            mean = floatArrayOf(1f, 2f, 3f),
            std = floatArrayOf(4f, 5f, 6f),
        )

        sdk.initializeModelRoot(
            temporaryFolder.root,
            config(
                target = DetectorModel(
                    fileName = "target.rknn",
                    labels = listOf("head"),
                    inputWidth = 384,
                    inputHeight = 320,
                    scoreThreshold = 0.45f,
                    maxResults = 12,
                    inputType = RknnInputType.FLOAT16,
                    inputLayout = RknnInputLayout.NCHW,
                    normalization = normalization,
                ),
            ),
            RknnState.READY,
        )

        val model = runtime.registered.single()
        assertEquals(384, model.inputWidth)
        assertEquals(320, model.inputHeight)
        assertEquals(0.45f, model.scoreThreshold)
        assertEquals(12, model.maxResults)
        assertEquals(RknnInputType.FLOAT16, model.inputType)
        assertEquals(RknnInputLayout.NCHW, model.inputLayout)
        assertEquals(normalization, model.normalization)
    }

    @Test
    fun initialize_readsTrimmedNonblankLabelsFromFile() {
        createModel("target.rknn")
        temporaryFolder.newFile("target.labels").writeText("head\n\n person \n")
        val runtime = FakeModelRuntime()
        val sdk = ModelApi(runtime)

        sdk.initializeModelRoot(
            temporaryFolder.root,
            config(target = DetectorModel(fileName = "target.rknn", labelFileName = "target.labels")),
            RknnState.READY,
        )

        assertEquals(listOf("head", "person"), runtime.registered.single().labels)
    }

    @Test
    fun initialize_reportsReadyWhenAllEnabledModelsRegister() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())

        val state = sdk.initializeModelRoot(
            temporaryFolder.root,
            config(target = DetectorModel(fileName = "target.rknn", labels = listOf("head"))),
            RknnState.READY,
        )

        assertEquals(MultimodalLifecycle.READY, state.lifecycle)
        assertTrue(state.readiness(ModelKey.TARGET).ready)
        assertFalse(state.readiness(ModelKey.ACTION).enabled)
    }

    @Test
    fun initialize_doesNotRegisterDisabledModels() {
        createModel("target.rknn")
        createModel("action.rknn")
        createModel("mask.rknn")
        val runtime = FakeModelRuntime()
        val sdk = ModelApi(runtime)

        val state = sdk.initializeModelRoot(
            temporaryFolder.root,
            config(
                action = DetectorModel(
                    enabled = true,
                    fileName = "action.rknn",
                    labels = listOf("action"),
                ),
            ),
            RknnState.READY,
        )

        assertEquals(listOf("action"), runtime.registered.map { it.id })
        assertFalse(state.readiness(ModelKey.TARGET).enabled)
        assertFalse(state.readiness(ModelKey.MASK).enabled)
    }

    @Test
    fun initialize_reportsPartialWhenSomeEnabledModelsAreInvalid() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())

        val state = sdk.initializeModelRoot(
            temporaryFolder.root,
            config(
                target = DetectorModel(fileName = "target.rknn", labels = listOf("head")),
                action = DetectorModel(enabled = true, fileName = "missing.rknn", labels = listOf("1X")),
            ),
            RknnState.READY,
        )

        assertEquals(MultimodalLifecycle.PARTIALLY_READY, state.lifecycle)
        assertTrue(state.readiness(ModelKey.TARGET).ready)
        assertFalse(state.readiness(ModelKey.ACTION).ready)
    }

    @Test
    fun initialize_reportsErrorWhenRuntimeIsUnavailable() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())

        val state = sdk.initializeModelRoot(
            temporaryFolder.root,
            config(target = DetectorModel(fileName = "target.rknn", labels = listOf("head"))),
            RknnState.UNSUPPORTED,
        )

        assertEquals(MultimodalLifecycle.ERROR, state.lifecycle)
        assertTrue(state.messages.any { it.contains("UNSUPPORTED") })
    }

    @Test
    fun detectForRole_publishesSynchronousResult() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(target = enabledTarget()), RknnState.READY)
        val expected = success("target")

        val actual = sdk.executeDetection(ModelKey.TARGET) { expected }

        assertEquals(expected.copy(resultSequence = actual.resultSequence), actual)
        assertEquals(actual, sdk.result(ModelKey.TARGET).value)
    }

    @Test
    fun detectForRole_publishesRepeatedEqualResultsWithIncreasingSequence() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(target = enabledTarget()), RknnState.READY)
        val expected = success("target")

        val first = sdk.executeDetection(ModelKey.TARGET) { expected }
        val second = sdk.executeDetection(ModelKey.TARGET) { expected }

        assertTrue(second.resultSequence > first.resultSequence)
        assertEquals(second, sdk.result(ModelKey.TARGET).value)
    }

    @Test
    fun detectForRole_convertsRuntimeExceptionToPublishedFailure() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(target = enabledTarget()), RknnState.READY)

        val result = sdk.executeDetection(ModelKey.TARGET) { error("native failure") }

        assertFalse(result.success)
        assertTrue(result.message.orEmpty().contains("native failure"))
        assertEquals(result, sdk.result(ModelKey.TARGET).value)
    }

    @Test
    fun detectForRole_disabledModelDoesNotRunOrPublish() {
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(), RknnState.READY)
        var calls = 0

        val result = sdk.executeDetection(ModelKey.ACTION) {
            calls++
            success("action")
        }

        assertFalse(result.success)
        assertEquals(0, calls)
        assertNull(sdk.result(ModelKey.ACTION).value)
    }

    @Test
    fun detectAsyncForRole_rejectsBusyFrameAndPublishesCompletion() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(target = enabledTarget()), RknnState.READY)
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val expected = success("target")

        assertTrue(sdk.executeDetectionAsync(ModelKey.TARGET) {
            started.countDown()
            release.await(2, TimeUnit.SECONDS)
            expected
        })
        assertTrue(started.await(2, TimeUnit.SECONDS))
        assertFalse(sdk.executeDetectionAsync(ModelKey.TARGET) { expected })
        release.countDown()

        assertTrue(waitUntil { sdk.result(ModelKey.TARGET).value?.modelId == expected.modelId })
    }

    @Test
    fun serialExecutionMode_runsDifferentModelsOneAtATime() {
        val sdk = initializedTwoModelSdk(RunMode.SERIAL)
        val firstStarted = CountDownLatch(1)
        val allowFirstToFinish = CountDownLatch(1)
        val secondStarted = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)

        executor.submit {
            sdk.executeDetection(ModelKey.TARGET) {
                firstStarted.countDown()
                allowFirstToFinish.await(2, TimeUnit.SECONDS)
                success("target")
            }
        }
        assertTrue(firstStarted.await(1, TimeUnit.SECONDS))
        executor.submit {
            sdk.executeDetection(ModelKey.ACTION) {
                secondStarted.countDown()
                success("action")
            }
        }

        assertFalse(secondStarted.await(150, TimeUnit.MILLISECONDS))
        allowFirstToFinish.countDown()
        assertTrue(secondStarted.await(1, TimeUnit.SECONDS))
        executor.shutdownNow()
    }

    @Test
    fun parallelExecutionMode_runsDifferentModelsConcurrently() {
        val sdk = initializedTwoModelSdk(RunMode.PARALLEL)
        val bothStarted = CountDownLatch(2)
        val allowFinish = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)

        listOf(ModelKey.TARGET, ModelKey.ACTION).forEach { key ->
            executor.submit {
                sdk.executeDetection(key) {
                    bothStarted.countDown()
                    allowFinish.await(2, TimeUnit.SECONDS)
                    success(key.value)
                }
            }
        }

        assertTrue(bothStarted.await(1, TimeUnit.SECONDS))
        allowFinish.countDown()
        executor.shutdownNow()
    }

    @Test
    fun release_clearsResultsAndRejectsFurtherDetection() {
        createModel("target.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(temporaryFolder.root, config(target = enabledTarget()), RknnState.READY)
        sdk.executeDetection(ModelKey.TARGET) { success("target") }

        sdk.release()

        assertEquals(MultimodalLifecycle.RELEASED, sdk.state.value.lifecycle)
        assertNull(sdk.result(ModelKey.TARGET).value)
        assertFalse(sdk.executeDetectionAsync(ModelKey.TARGET) { success("target") })

    }

    @Test
    fun customKey_registersAndPublishesIndependentResult() {
        val custom = ModelKey("ppe")
        createModel("ppe.rknn")
        val sdk = ModelApi(FakeModelRuntime())
        sdk.initializeModelRoot(
            temporaryFolder.root,
            ModelConfig(mapOf(custom to DetectorModel(fileName = "ppe.rknn", labels = listOf("helmet")))),
            RknnState.READY,
        )
        val expected = success("ppe")

        val actual = sdk.executeDetection(custom) { expected }

        assertTrue(sdk.state.value.readiness(custom).ready)
        assertEquals(expected.copy(resultSequence = actual.resultSequence), sdk.result(custom).value)
    }

    private fun createModel(name: String): File = temporaryFolder.newFile(name)

    private fun enabledTarget() = DetectorModel(fileName = "target.rknn", labels = listOf("head"))

    private fun initializedTwoModelSdk(mode: RunMode): ModelApi {
        createModel("target.rknn")
        createModel("action.rknn")
        return ModelApi(FakeModelRuntime()).also { sdk ->
            sdk.initializeModelRoot(
                temporaryFolder.root,
                config(
                    target = enabledTarget(),
                    action = DetectorModel(fileName = "action.rknn", labels = listOf("action")),
                    runMode = mode,
                ),
                RknnState.READY,
            )
        }
    }

    private fun config(
        target: DetectorModel = DetectorModel(enabled = false, fileName = "target.rknn"),
        action: DetectorModel = DetectorModel(enabled = false, fileName = "action.rknn"),
        mask: DetectorModel = DetectorModel(enabled = false, fileName = "mask.rknn"),
        runMode: RunMode = RunMode.SERIAL,
    ) = ModelConfig(
        models = mapOf(
            ModelKey.TARGET to target,
            ModelKey.ACTION to action,
            ModelKey.MASK to mask,
        ),
        runMode = runMode,
    )

    private fun success(id: String) = RknnObjectDetectionResult(
        success = true,
        backend = RknnBackend.ROCKCHIP_RKNN,
        modelId = id,
        detections = emptyList(),
        durationMs = 1,
    )

    private fun waitUntil(condition: () -> Boolean): Boolean {
        repeat(100) {
            if (condition()) return true
            Thread.sleep(10)
        }
        return false
    }
}

private class FakeModelRuntime : ModelRuntime {
    val registered = mutableListOf<RknnModelConfig>()

    override fun registerModel(config: RknnModelConfig): Boolean {
        registered += config
        return true
    }

    override fun unregisterModel(modelId: String) = Unit
}
