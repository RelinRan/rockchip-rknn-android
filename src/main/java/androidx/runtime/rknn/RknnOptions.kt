package androidx.runtime.rknn

import java.io.File

/**
 * Runtime initialization and model-directory options.
 *
 * Example: `RknnRuntime().initialize(context, RknnOptions(modelRoot = filesDir))`.
 *
 * @property project Directory name used below external storage when [modelRoot] is absent.
 * @property modelDir Model subdirectory below [project].
 * @property modelRoot Explicit model directory; overrides [project] and [modelDir].
 * @property backend Preferred inference backend.
 * @property allowCpuFallback Whether an unavailable NPU may fall back to a CPU implementation.
 * @property debug Whether verbose runtime, tensor, and timing logs are enabled.
 */
data class RknnOptions(
    val project: String = "AiHandHygiene",
    val modelDir: String = "model",
    val modelRoot: File? = null,
    val backend: RknnBackend = RknnBackend.ROCKCHIP_RKNN,
    val allowCpuFallback: Boolean = false,
    val debug: Boolean = false,
) {
    /**
     * Resolves the directory from which model files are loaded.
     *
     * @param externalStorageDirectory Android external-storage root used for the default path.
     * @return [modelRoot], or `<external>/<project>/<modelDir>` when no override is set.
     */
    fun resolveModelRoot(externalStorageDirectory: File): File =
        modelRoot ?: File(externalStorageDirectory, "$project/$modelDir")
}
