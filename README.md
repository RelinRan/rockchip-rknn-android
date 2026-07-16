# RKNN Android Runtime

**English** | [简体中文](README.zh-CN.md)

An open-source Android inference library for Rockchip NPUs, currently focused on RK3588 devices. It handles RKNN Runtime discovery, model session lifecycles, image preprocessing, JNI inference, output tensor decoding, multi-model result dispatch, and object tracking.

## Requirements

- Android API 24 or later
- RK3588 or another compatible device providing the RKNPU2 Runtime
- `arm64-v8a` or `armeabi-v7a`
- Bundled RKNN Runtime and headers: 2.3.2
- Models converted to `.rknn` with [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Android/librknn_api)

Files such as `.tflite`, `.onnx`, and `.pt` cannot be loaded directly by the Android RKNN Runtime. For YOLO model usage and export guidance, see the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

## Installation

Download the AAR from the repository's [GitHub Releases](../../releases), or include this repository as an Android library module:

```kotlin
dependencies {
    implementation(project(":rknn"))
}
```

The library includes JNI code and `librknnrt.so` for both supported ABIs.

## Model Directory

The default model directory is:

```text
/storage/emulated/0/AiHandHygiene/model/
```

You can provide a different `File` directory during initialization.

## Multi-model API

`ModelApi` is the high-level API for continuous camera-frame workloads such as object detection, action detection, and YOLO Pose.

```kotlin
val actionKey = ModelKey.ACTION
val poseKey = ModelKey("pose")

val config = ModelConfig(
    models = mapOf(
        actionKey to DetectorModel(
            fileName = "action.rknn",
            labelFileName = "action.labels",
            modelType = RknnModelType.OBJECT_DETECTOR,
            decoderType = RknnDecoderType.YOLO_END_TO_END,
            inputWidth = 640,
            inputHeight = 640,
            scoreThreshold = 0.4f,
        ),
        poseKey to DetectorModel(
            fileName = "yolo26n-pose-rk3588.rknn",
            labels = listOf("person"),
            modelType = RknnModelType.POSE_DETECTOR,
            decoderType = RknnDecoderType.YOLO_POSE_LANDMARK,
            inputWidth = 640,
            inputHeight = 640,
            scoreThreshold = 0.25f,
        ),
    ),
)

val modelApi = ModelApi()
val state = modelApi.initialize(context, modelRoot, config)

if (state.readiness(actionKey).ready) {
    val result = modelApi.detect(actionKey, bitmap)
}

modelApi.release()
```

For real-time camera input, use `detectAsync()`. Each `ModelKey` accepts only one in-flight request, preventing camera-frame queues from growing without bounds.

### Synchronous and asynchronous inference

```kotlin
val result = modelApi.detect(actionKey, bitmap)
if (result.success) {
    result.detections.forEach { detection ->
        val category = detection.categories.first()
        val box = detection.boundingBox
    }
}

val accepted = modelApi.detectAsync(actionKey, bitmap)
if (!accepted) {
    // This frame was not accepted because the model is still running.
    bitmap.recycle()
}
```

Different models have independent busy locks, while NPU inference is currently serialized by a runtime-wide lock.

### Observing results

```kotlin
val actionResult: StateFlow<RknnObjectDetectionResult?> = modelApi.result(actionKey)
val poseResult: StateFlow<RknnObjectDetectionResult?> = modelApi.result(poseKey)
```

In Compose, use `modelApi.result(poseKey).collectAsState()`.

## DetectorModel Configuration

| Parameter | Description |
|---|---|
| `enabled` | Whether the model is registered |
| `fileName` | `.rknn` filename under the model directory |
| `labels` | Category names supplied directly |
| `labelFileName` | Label file in the model directory; `labels` takes precedence |
| `scoreThreshold` | Minimum confidence in the `0..1` range |
| `nmsThreshold` | Per-class NMS threshold for raw YOLO and DFL outputs; default `0.5` |
| `maxResults` | Maximum number of detections returned |
| `poseKeyPointCount` | YOLO Pose keypoint count; default is the COCO count of `17` |
| `inputWidth` / `inputHeight` | Input dimensions, which must match the RKNN model |
| `inputType` | Input data type; default `AUTO` |
| `inputLayout` | Input tensor layout; default `AUTO` |
| `normalization` | RGB normalization parameters |
| `decoderType` | Model output decoder; set this explicitly in production |
| `mediaPipeModel` | MediaPipe SSD model specification |
| `modelType` | Business-level model type |

The default normalization is `(pixel - mean) / std`, with `mean = [0, 0, 0]` and `std = [255, 255, 255]`. If the model graph already performs `/255`, use:

```kotlin
normalization = RknnNormalization(
    mean = floatArrayOf(0f, 0f, 0f),
    std = floatArrayOf(1f, 1f, 1f),
)
```

## Supported Decoders

| Decoder | Output |
|---|---|
| `YOLO_END_TO_END` | `[1,N,6]` or `[1,6,N]` end-to-end detections |
| `YOLO_DETECT_RAW` | Raw YOLO tensors with or without objectness |
| `YOLO_DETECT_HEADS` | NCHW/NHWC DFL detection heads |
| `YOLO_POSE_LANDMARK` | Detection boxes and keypoints |
| `YOLO_POSE_RAW` | Raw pose tensor before NMS |
| `YOLO_POSE_HEADS` | DFL box, class, and keypoint branches |
| `YOLO_SEGMENT` | Detections, mask coefficients, and prototypes |
| `YOLO_CLASSIFY` | Classification probabilities or logits |
| `YOLO_OBB` | Oriented bounding boxes |
| `MEDIA_PIPE_SSD` | MediaPipe Model Maker SSD/RetinaNet outputs |
| `AUTO` | Decoder selected from output tensor shapes |

Production configurations should explicitly set `decoderType`; `AUTO` is intended for investigating unknown model outputs.

Multi-label models are enabled with `DetectorModel(multiLabel = true)`. A box is still represented by one `RknnDetection`; all categories meeting the threshold are stored in descending score order in `categories`. Segmentation results are available through `segmentationMask`, and rotated boxes through `orientedBox`, while the regular `boundingBox` remains available.

### YOLO Pose results

Pose keypoints are returned in `detection.keyPoints`. Each `RknnKeypoint` contains its index, name, `x`/`y` coordinates, and score. Coordinates are mapped from the letterboxed model input back into the submitted Bitmap's coordinate system.

## MediaPipe Object Detector

The decoder supports RKNN-converted outputs from these MediaPipe Model Maker variants:

- `MOBILENET_V2`: 256 x 256
- `MOBILENET_V2_I320`: 320 x 320
- `MOBILENET_MULTI_AVG`: 256 x 256
- `MOBILENET_MULTI_AVG_I384`: 384 x 384

```kotlin
DetectorModel(
    fileName = "action.rknn",
    labels = labels,
    inputWidth = 384,
    inputHeight = 384,
    decoderType = RknnDecoderType.MEDIA_PIPE_SSD,
    mediaPipeModel = MediaPipeObjectDetectorModel.MOBILENET_MULTI_AVG_I384,
)
```

## Tracking

`RknnTracker` applies two-stage high/low-confidence association to detections across frames:

```kotlin
val tracker = RknnTracker(RknnTrackerConfig())
val tracked = tracker.update(detectionResult.detections)
tracker.reset()
```

Temporarily unmatched tracks remain in the `LOST` state and are removed after `maxLostFrames`. Tracking is independent from model output decoding and is therefore not a `RknnDecoderType`.

## Low-level RknnRuntime

Use `RknnRuntime` when you need raw tensors, image classification, MediaPipe Pose Landmark, or Hand Landmark output.

```kotlin
val runtime = RknnRuntime()
runtime.initialize(context, RknnOptions(modelRoot = modelRoot, debug = false))

runtime.registerModel(
    RknnModelConfig(
        id = "model-id",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "model.rknn",
        inputWidth = 640,
        inputHeight = 640,
        labels = labels,
        decoderType = RknnDecoderType.YOLO_END_TO_END,
    ),
)

runtime.run(modelId, bitmap)                 // Raw tensors
runtime.detectObjects(modelId, bitmap)       // Object detection and YOLO Pose
runtime.classifyImage(modelId, bitmap)       // Image classification
runtime.detectPoseLandmarks(modelId, bitmap) // Pose Landmark ROI
runtime.detectHandLandmarks(modelId, bitmap) // Hand Landmark ROI
runtime.retryModel(modelId)                  // Explicitly retry a failed session
runtime.unregisterModel(modelId)
runtime.close()
```

A session is created immediately by `registerModel()`. A failed initial creation is not retried on every frame; re-register the model, reset the SDK, or call `retryModel()`.

## Image Classification

Supported classification outputs include EfficientNet-Lite0 and EfficientNet-Lite2 in Float32 and Int8 variants. Results support direct probabilities, logits with Softmax, threshold filtering, and Top-K ordering.

## Pose and Hand Landmark

Supported landmark submodels are Pose Landmark Lite, Full, and Heavy, plus Hand Landmark Full. These decoders target the landmark submodels inside official `.task` packages. They do not include the detector, rotated ROI preparation, or video tracking pipeline; callers must supply a detector-cropped and rotation-corrected ROI Bitmap.

## Debug Logging

Set `debug = true` in `ModelConfig` or `RknnOptions` to log device/runtime versions, model and decoder details, tensor shapes and quantization, memory usage, inference timing, and output summaries. Runtime errors remain logged when debug output is disabled.

Tensor enum logs include both names and numeric values, for example `type=FP16(1)`, `format=NHWC(1)`, and `quantization=AFFINE(2)`.

## Multi-model Scheduling

On RK3588, a practical starting point for concurrent Action and Pose workloads is roughly 80 ms between Action submissions (about 8-12 FPS) and 250 ms between Pose submissions (about 3-4 FPS). Do not submit every camera frame to every model. Schedule each model independently with `detectAsync()` and recycle the Bitmap when it returns `false`.

## Troubleshooting

### `Supported detection output tensors not found`

The actual output shape does not match `decoderType`. Enable debug logging and inspect `model tensor direction=output ... dims=[...]`. For example, an end-to-end YOLO26 Pose output shaped `[1,300,57]` should use `YOLO_POSE_LANDMARK`.

### `openSession failed`

Check that the model targets the device chip (for example `rk3588`), RKNN-Toolkit2 is compatible with the device Runtime, the model file is complete, `/vendor/lib64/librknnrt.so` exists, and the RKNPU driver is available.

### Incorrect detection box positions

The decoder returns coordinates in the submitted Bitmap's coordinate system, not the physical pixels of a `TextureView`. Ensure that inference input dimensions and overlay source-image dimensions agree.

### Unexpectedly low accuracy

Check for duplicate `/255` normalization, RGB/BGR ordering, input dimensions, FP16/INT8 conversion parameters, and whether the label count matches the output class count.

## Build and Test

```bash
./gradlew testDebugUnitTest
./gradlew assembleRelease
```

The release AAR is generated under `build/outputs/aar/`. Pushing a semantic-version tag such as `v1.0.0` runs the GitHub Actions release workflow and attaches the versioned AAR to a GitHub Release.

## Related Projects

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [RKNN-Toolkit2 Android Runtime API](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Android/librknn_api)

## License

Licensed under the [MIT License](LICENSE).
