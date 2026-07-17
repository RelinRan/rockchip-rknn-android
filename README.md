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

### Use the released AAR

Download `rockchip-rknn-android-<version>.aar` from [Releases](../../releases), place it in the application module's `libs/` directory, and add:

```kotlin
dependencies {
    implementation(files("libs/rockchip-rknn-android-1.0.0.aar"))
}
```

The AAR contains Kotlin classes, `librknn_jni.so`, and `librknnrt.so` for `arm64-v8a` and `armeabi-v7a`. Keep the required ABIs enabled in the consuming application.

### Storage and permissions

Prefer an app-owned model directory such as `context.filesDir` or `context.getExternalFilesDir(null)`, which requires no storage permission:

```kotlin
val modelRoot = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }
```

The historical `/storage/emulated/0/AiHandHygiene/model/` path is subject to scoped-storage restrictions. On recent Android versions, use app-owned storage or copy user-selected files from the Storage Access Framework. The library manifest declares no permissions; camera permission and image acquisition remain the application's responsibility.

## Which API Should I Use?

| API | Recommended use |
|---|---|
| `ModelApi` | Multiple object/pose models, camera frames, result flows, and backpressure |
| `RknnRuntime` | Explicit registration, raw tensors, classification, detection, and landmarks |
| Process-wide compatibility objects | Migration from `RknnApi`, `RknnObjectDetector`, `RknnImageClassifier`, and landmark helpers |
| `RknnRuntimeSession` | Direct RKNN contexts, core masks, dynamic shapes, and native memory |

For new code, own a `ModelApi` or `RknnRuntime` instance and close it with the Activity, ViewModel, service, or other component lifecycle.

## Complete Runtime Lifecycle

```kotlin
val runtime = RknnRuntime()
val state = runtime.initialize(
    context,
    RknnOptions(modelRoot = modelRoot, debug = BuildConfig.DEBUG),
)
check(state == RknnState.READY) { runtime.deviceInfo().reasons.joinToString() }

check(runtime.registerModel(
    RknnModelConfig(
        id = "detector",
        type = RknnModelType.OBJECT_DETECTOR,
        fileName = "yolo.rknn",
        inputWidth = 640,
        inputHeight = 640,
        labels = listOf("person", "helmet"),
        decoderType = RknnDecoderType.YOLO_END_TO_END,
        scoreThreshold = 0.25f,
        nmsThreshold = 0.5f,
        maxResults = 100,
    ),
))

val result = runtime.detectObjects("detector", bitmap)
if (result.success) {
    result.detections.forEach { detection ->
        val bestCategory = detection.categories.firstOrNull()
        val sourcePixelBox = detection.boundingBox
    }
} else {
    Log.e("RKNN", result.message.orEmpty())
}

runtime.unregisterModel("detector")
runtime.close()
```

Registration verifies the model file and creates its native session immediately. A failed session is remembered to prevent retries on every frame; correct the cause and call `retryModel(id)`, or unregister and register again. Model IDs must be unique.

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

### Bitmap ownership and concurrency

`detect()` is synchronous and never recycles the Bitmap. `detectAsync()` returns `true` only when the request is accepted. When it returns `false`, the caller still owns the Bitmap and should recycle or reuse it. Do not mutate an accepted Bitmap until inference finishes. Each model has an independent busy gate, while native NPU execution is serialized for runtime safety.

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
| `MEDIA_PIPE_IMAGE_CLASSIFIER` | EfficientNet-Lite probability or logits output |
| `MEDIA_PIPE_POSE_LANDMARK` | Pose landmarks, world landmarks, presence, and optional mask |
| `MEDIA_PIPE_HAND_LANDMARK` | Hand landmarks, world landmarks, presence, and handedness |
| `AUTO` | Decoder selected from output tensor shapes |

Production configurations should explicitly set `decoderType`; `AUTO` is intended for investigating unknown model outputs.

### Decoder selection guide

| Exported output | Decoder |
|---|---|
| `x1,y1,x2,y2,score,classId` rows | `YOLO_END_TO_END` |
| One raw `4+C` or `5+C` candidate tensor | `YOLO_DETECT_RAW` |
| Separate DFL box and class heads | `YOLO_DETECT_HEADS` |
| End-to-end boxes plus `K*3` pose values | `YOLO_POSE_LANDMARK` |
| Raw pose candidates | `YOLO_POSE_RAW` |
| Separate DFL box, class, and keypoint heads | `YOLO_POSE_HEADS` |
| Detections, mask coefficients, and prototype | `YOLO_SEGMENT` |
| Classification vector | `YOLO_CLASSIFY` |
| Center, size, scores, and rotation angle | `YOLO_OBB` |

Tensor dimensions vary by exporter. Enable debug logging and compare actual output names and dimensions before choosing a decoder.

## Inputs, Labels, and Coordinates

- Bitmaps are converted to RGB, letterboxed to `inputWidth x inputHeight`, and normalized per channel.
- `inputType=AUTO` and `inputLayout=AUTO` use tensor metadata. Override them only when the converted model explicitly requires `UINT8`, `INT8`, `FLOAT16`, `FLOAT32`, `NHWC`, or `NCHW`.
- Inline `labels` take precedence over `labelFileName`; label files contain one UTF-8 label per line.
- Boxes and YOLO keypoints are mapped to submitted-Bitmap pixel coordinates.
- MediaPipe landmark coordinates remain normalized; world landmarks are measured in meters.
- Segmentation masks retain their own dimensions and can be resized or converted with `toBinary(threshold)`.

### Reading specialized results

```kotlin
val classification = runtime.classifyImage("classifier", bitmap)
classification.categories.forEach { Log.d("RKNN", "${it.name}: ${it.score}") }

result.detections.forEach { detection ->
    val binaryMask = detection.segmentationMask?.toBinary(0.5f)
    val rotatedCorners = detection.orientedBox?.corners
    val posePoints = detection.keyPoints
}

// Landmark calls require an already cropped and rotation-corrected ROI.
val pose = runtime.detectPoseLandmarks("pose-landmark", personRoiBitmap)
val hand = runtime.detectHandLandmarks("hand-landmark", handRoiBitmap)
```

Classification configurations can set `classifierModel` and `classifierScoreType`. Landmark configurations can set `poseLandmarkModel` or `handLandmarkModel` to enforce the corresponding input size.

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

## Advanced Native Session API

`RknnRuntimeSession` exposes context duplication, NPU core selection, non-blocking execution, dynamic shapes, and native tensor memory. It is intended for callers familiar with the RKNN C API; normal image inference should use `RknnRuntime`.

```kotlin
RknnRuntimeSession.open(File(modelRoot, "model.rknn"), debug = true).use { session ->
    session.setCoreMask(RknnCoreMask.CORE_0_1_2)
    session.setInputShape(0, 1, 640, 640, 3)
    session.createMemory(640L * 640L * 3L).use { memory ->
        memory.buffer?.apply { clear(); put(rgbBytes) }
        memory.synchronize(RknnMemorySyncMode.TO_DEVICE)
        session.setIoMemory(memory, tensorIndex = 0, input = true)
        session.execute()
    }
}
```

Memory belongs to its creating session. Close memory before the session, use a direct buffer for physical memory, and treat non-zero native return codes as RKNN API errors.

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
./gradlew externalNativeBuildRelease
./gradlew assembleRelease
```

JNI outputs are generated below `build/intermediates/cmake/release/obj/<abi>/`; the AAR is generated under `build/outputs/aar/`. Refresh checked-in `output/<abi>/librknn_jni.so` after JNI changes. Pushing a tag matching `VERSION_NAME`, such as `v1.0.0`, runs tests and publishes the versioned AAR plus SHA-256 checksum.

## Related Projects

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [RKNN-Toolkit2 Android Runtime API](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Android/librknn_api)

## License

Licensed under the [MIT License](LICENSE).
