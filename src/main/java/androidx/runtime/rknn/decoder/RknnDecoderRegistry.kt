package androidx.runtime.rknn.decoder

import androidx.runtime.rknn.RknnModelConfig
import androidx.runtime.rknn.RknnModelType
import androidx.runtime.rknn.data.RknnImage
import androidx.runtime.rknn.internal.NativeTensorOutput

/** 根据显式配置或输出张量形状选择对应的检测结果解码器。 */
internal object RknnDecoderRegistry {
    fun decode(
        type: RknnDecoderType,
        outputs: List<NativeTensorOutput>,
        image: RknnImage,
        config: RknnModelConfig,
    ): RknnDecodedResult = when (resolve(type, outputs, config)) {
        RknnDecoderType.YOLO_POSE_LANDMARK -> RknnDecodedResult.Detection(
            YoloPoseDecoder.decode(
                outputs.first { hasChannel(it, 6 + config.poseKeyPointCount * 3) }, image, config,
            ),
        )
        RknnDecoderType.YOLO_END_TO_END -> RknnDecodedResult.Detection(
            Yolo26Decoder.decode(outputs.first { hasChannel(it, 6) }, image, config),
        )
        RknnDecoderType.YOLO_DETECT_RAW -> RknnDecodedResult.Detection(
            RawYoloDecoder.decode(outputs.first { isRawYolo(it, config) }, image, config),
        )
        RknnDecoderType.YOLO_DETECT_HEADS -> RknnDecodedResult.Detection(
            YoloDflDecoder.decode(outputs, image, config),
        )
        RknnDecoderType.YOLO_POSE_RAW -> RknnDecodedResult.Detection(
            YoloRawPoseDecoder.decode(outputs.single(), image, config),
        )
        RknnDecoderType.YOLO_SEGMENT -> RknnDecodedResult.Detection(
            YoloSegmentationDecoder.decode(outputs, image, config),
        )
        RknnDecoderType.YOLO_CLASSIFY -> RknnDecodedResult.Classification(
            YoloClassificationDecoder.decode(outputs.single(), config),
        )
        RknnDecoderType.YOLO_OBB -> RknnDecodedResult.Detection(
            YoloObbDecoder.decode(outputs.single(), image, config),
        )
        RknnDecoderType.YOLO_POSE_HEADS -> RknnDecodedResult.Detection(
            YoloPoseHeadsDecoder.decode(outputs, image, config),
        )
        RknnDecoderType.MEDIA_PIPE_SSD -> RknnDecodedResult.Detection(
            MediaPipeSsdDecoder.decode(
                boxes = outputs.first { isMediaPipeBoxes(it, config) },
                scores = outputs.first { isMediaPipeScores(it, config) },
                image = image,
                config = config,
            ),
        )
        RknnDecoderType.MEDIA_PIPE_IMAGE_CLASSIFIER -> RknnDecodedResult.Classification(
            ImageClassifierDecoder.decode(
                outputs.single { it.data.size == config.labels.size }, config,
            ),
        )
        RknnDecoderType.MEDIA_PIPE_POSE_LANDMARK -> RknnDecodedResult.PoseLandmark(
            PoseLandmarkDecoder.decode(outputs, config),
        )
        RknnDecoderType.MEDIA_PIPE_HAND_LANDMARK -> RknnDecodedResult.HandLandmark(
            HandLandmarkDecoder.decode(outputs, config),
        )
        RknnDecoderType.AUTO -> error("AUTO decoder must be resolved")
    }

    fun resolve(
        requested: RknnDecoderType,
        outputs: List<NativeTensorOutput>,
        config: RknnModelConfig,
    ): RknnDecoderType {
        if (requested != RknnDecoderType.AUTO) return requested
        return when {
            config.type == RknnModelType.IMAGE_CLASSIFIER -> RknnDecoderType.MEDIA_PIPE_IMAGE_CLASSIFIER
            config.type == RknnModelType.POSE_LANDMARKER -> RknnDecoderType.MEDIA_PIPE_POSE_LANDMARK
            config.type == RknnModelType.HAND_LANDMARKER -> RknnDecoderType.MEDIA_PIPE_HAND_LANDMARK
            config.type == RknnModelType.IMAGE_SEGMENTER -> RknnDecoderType.YOLO_SEGMENT
            config.type == RknnModelType.OBB_DETECTOR -> RknnDecoderType.YOLO_OBB
            config.type == RknnModelType.POSE_DETECTOR && outputs.count { it.dims.size == 4 } >= 3 ->
                RknnDecoderType.YOLO_POSE_HEADS
            config.type == RknnModelType.POSE_DETECTOR &&
                outputs.any { hasChannel(it, 6 + config.poseKeyPointCount * 3) } ->
                RknnDecoderType.YOLO_POSE_LANDMARK
            outputs.any { hasChannel(it, 6) } -> RknnDecoderType.YOLO_END_TO_END
            outputs.any { isRawYolo(it, config) } -> RknnDecoderType.YOLO_DETECT_RAW
            outputs.any { isDflHead(it, config) } -> RknnDecoderType.YOLO_DETECT_HEADS
            outputs.any { isMediaPipeBoxes(it, config) } &&
                outputs.any { isMediaPipeScores(it, config) } -> RknnDecoderType.MEDIA_PIPE_SSD
            else -> error(
                "Supported decoder not found for tensors: " +
                    outputs.joinToString { "${it.name}${it.dims.contentToString()}" } +
                    "; configure YOLO_END_TO_END, YOLO_DETECT_RAW, YOLO_DETECT_HEADS, " +
                    "YOLO_POSE_LANDMARK or YOLO_POSE_RAW explicitly",
            )
        }
    }

    private fun isRawYolo(output: NativeTensorOutput, config: RknnModelConfig): Boolean {
        return listOf(config.labels.size + 4, config.labels.size + 5).any { channels ->
            output.data.size % channels == 0 && hasChannel(output, channels)
        }
    }

    private fun isDflHead(output: NativeTensorOutput, config: RknnModelConfig): Boolean {
        if (output.dims.size != 4 || output.dims.firstOrNull() != 1) return false
        return output.dims.drop(1).any { channels ->
            channels == config.labels.size || channels % 4 == 0 ||
                ((channels - config.labels.size) % 4 == 0 && channels > config.labels.size)
        }
    }

    private fun hasChannel(output: NativeTensorOutput, channels: Int): Boolean =
        output.dims.drop(1).any { it == channels }

    private fun isMediaPipeBoxes(output: NativeTensorOutput, config: RknnModelConfig): Boolean {
        if (output.data.size % 4 != 0) return false
        val anchors = output.data.size / 4
        return runCatching {
            val model = if (config.mediaPipeModel == MediaPipeObjectDetectorModel.AUTO) {
                MediaPipeObjectDetectorModel.resolve(config.inputWidth, config.inputHeight, anchors)
            } else config.mediaPipeModel
            anchors == model.anchorCount
        }.getOrDefault(false)
    }

    private fun isMediaPipeScores(output: NativeTensorOutput, config: RknnModelConfig): Boolean {
        val classes = config.labels.size + 1
        if (classes <= 1 || output.data.size % classes != 0) return false
        val anchors = output.data.size / classes
        return runCatching {
            val model = if (config.mediaPipeModel == MediaPipeObjectDetectorModel.AUTO) {
                MediaPipeObjectDetectorModel.resolve(config.inputWidth, config.inputHeight, anchors)
            } else config.mediaPipeModel
            anchors == model.anchorCount
        }.getOrDefault(false)
    }
}
