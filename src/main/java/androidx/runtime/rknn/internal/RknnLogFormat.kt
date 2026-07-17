package androidx.runtime.rknn.internal

/**
 * Provides the `RknnLogFormat` contract used by the RKNN Android runtime.
 *
 * Usage: create or reference `RknnLogFormat` where its surrounding API requires this contract.
 */
internal object RknnLogFormat {
    /**
     * Executes `environment` for the RKNN runtime contract.
     * @param chip Value supplied for `chip`.
     * @param driverVersion Value supplied for `driverVersion`.
     * @param runtimeVersion Value supplied for `runtimeVersion`.
     */
    fun environment(
        chip: String,
        driverVersion: String,
        runtimeVersion: String? = null,
    ): String = buildString {
        append("chip=").append(chip)
        append(" rknpuDriver=").append(driverVersion)
        if (!runtimeVersion.isNullOrBlank()) append(" deviceRuntime=").append(runtimeVersion)
    }

    /**
     * Executes `tensors` for the RKNN runtime contract.
     * @param outputs Native output tensors to decode.
     */
    fun tensors(outputs: List<NativeTensorOutput>): String = outputs.joinToString(
        prefix = "tensors=[",
        postfix = "]",
    ) { tensor ->
        "index=${tensor.index} name=${tensor.name} dims=${tensor.dims.joinToString("x")} " +
            "elements=${tensor.data.size} type=${tensor.type} format=${tensor.format}"
    }

    /**
     * Executes `completion` for the RKNN runtime contract.
     * @param modelId Identifier of the registered model.
     * @param success Value supplied for `success`.
     * @param durationMs Value supplied for `durationMs`.
     * @param outputCount Value supplied for `outputCount`.
     * @param message Value supplied for `message`.
     */
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
