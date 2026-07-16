package androidx.runtime.rknn

import java.io.File

/** RKNN 运行时初始化参数以及模型根目录配置。 */
data class RknnOptions(
    val project: String = "AiHandHygiene",
    val modelDir: String = "model",
    val modelRoot: File? = null,
    val backend: RknnBackend = RknnBackend.ROCKCHIP_RKNN,
    val allowCpuFallback: Boolean = false,
    val debug: Boolean = false,
) {
    fun resolveModelRoot(externalStorageDirectory: File): File =
        modelRoot ?: File(externalStorageDirectory, "$project/$modelDir")
}
