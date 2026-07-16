package androidx.runtime.rknn.internal

/** 统一生成 RKNN 诊断日志文本。 */
internal object RknnLogFormat {
    fun environment(
        chip: String,
        driverVersion: String,
        runtimeVersion: String? = null,
    ): String = buildString {
        append("chip=").append(chip)
        append(" rknpuDriver=").append(driverVersion)
        if (!runtimeVersion.isNullOrBlank()) append(" deviceRuntime=").append(runtimeVersion)
    }

    fun tensors(outputs: List<NativeTensorOutput>): String = outputs.joinToString(
        prefix = "tensors=[",
        postfix = "]",
    ) { tensor ->
        "index=${tensor.index} name=${tensor.name} dims=${tensor.dims.joinToString("x")} " +
            "elements=${tensor.data.size} type=${tensor.type} format=${tensor.format}"
    }

    fun completion(
        modelId: String,
        success: Boolean,
        durationMs: Long,
        outputCount: Int,
        message: String?,
    ): String = buildString {
        append("model=").append(modelId)
        append(" success=").append(success)
        append(" durationMs=").append(durationMs)
        append(" outputs=").append(outputCount)
        if (!message.isNullOrBlank()) append(" message=").append(message)
    }
}
