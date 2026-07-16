# RKNN Android Runtime

[English](README.md) | **简体中文**

`rknn` 是面向 Rockchip NPU 的 Android 推理模块，当前目标设备为 RK3588。模块负责 RKNN Runtime 探测、模型 Session 生命周期、图像预处理、JNI 推理、输出张量解码以及多模型结果分发。

## 运行要求

- Android API 24 及以上
- RK3588 或其他提供 RKNPU2 Runtime 的兼容设备
- ABI：`arm64-v8a`、`armeabi-v7a`
- 当前随模块集成的 `librknnrt.so` 和头文件版本：RKNN Runtime 2.3.2
- 模型必须提前通过 RKNN-Toolkit2 转换为 `.rknn`

`.tflite`、`.onnx` 和 `.pt` 文件不能直接交给 Android RKNN Runtime 加载。

## 引入模块

在应用模块中添加依赖：

```kotlin
dependencies {
    implementation(project(":rknn"))
}
```

模块已经包含 JNI 和以下 Runtime：

```text
src/main/jniLibs/arm64-v8a/librknnrt.so
src/main/jniLibs/armeabi-v7a/librknnrt.so
```

## 模型目录

默认模型目录为：

```text
/storage/emulated/0/AiHandHygiene/model/
```

示例：

```text
model/
├── action.rknn
├── action.labels
└── yolo26n-pose-rk3588.rknn
```

也可以在初始化时直接传入其他 `File` 目录。

## 高层多模型 API

`ModelApi` 适用于目标检测、动作检测和 YOLO Pose 等需要持续接收摄像头帧的场景。

### 配置模型

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
            maxResults = 100,
        ),
        poseKey to DetectorModel(
            fileName = "yolo26n-pose-rk3588.rknn",
            labels = listOf("person"),
            modelType = RknnModelType.POSE_DETECTOR,
            decoderType = RknnDecoderType.YOLO_POSE_LANDMARK,
            inputWidth = 640,
            inputHeight = 640,
            scoreThreshold = 0.25f,
            maxResults = 3,
        ),
    ),
    debug = false,
)
```

`ModelKey` 支持任意自定义字符串，不局限于预置的 `TARGET`、`ACTION` 和 `MASK`。

### 初始化

```kotlin
val modelApi = ModelApi()
val modelRoot = File(
    Environment.getExternalStorageDirectory(),
    "AiHandHygiene/model",
)

val state = modelApi.initialize(
    context = context,
    modelRoot = modelRoot,
    config = config,
)
```

只有对应模型的 readiness 为 true 时才能提交推理：

```kotlin
val poseReady = state.readiness(poseKey).ready
```

### 同步检测

```kotlin
val result = modelApi.detect(actionKey, bitmap)

if (result.success) {
    result.detections.forEach { detection ->
        val category = detection.categories.first()
        val box = detection.boundingBox
    }
}
```

### 异步检测

摄像头实时检测推荐使用异步接口：

```kotlin
val accepted = modelApi.detectAsync(actionKey, bitmap)
if (!accepted) {
    // 当前模型仍在推理，本帧未被接收。
    bitmap.recycle()
}
```

同一个 `ModelKey` 同时只执行一项异步推理，不会堆积摄像头帧。不同模型拥有独立 busy 锁，但当前通过运行时全局锁串行执行 NPU 推理。

### 订阅结果

```kotlin
val actionResult: StateFlow<RknnObjectDetectionResult?> =
    modelApi.result(actionKey)

val poseResult: StateFlow<RknnObjectDetectionResult?> =
    modelApi.result(poseKey)
```

Compose 中可以直接使用：

```kotlin
val poseResult by modelApi.result(poseKey).collectAsState()
```

### 释放资源

```kotlin
modelApi.release()
```

页面销毁、Activity 结束或模型不再使用时必须释放。`ModelApi` 也实现了 `Closeable`。

## DetectorModel 参数

| 参数 | 说明 |
|---|---|
| `enabled` | 是否注册该模型 |
| `fileName` | 模型目录下的 `.rknn` 文件名 |
| `labels` | 直接配置类别名称 |
| `labelFileName` | 从模型目录读取标签文件；配置了 `labels` 时优先使用 `labels` |
| `scoreThreshold` | 最低置信度，范围 `0..1` |
| `nmsThreshold` | Raw YOLO 和 DFL 输出的按类别 NMS 阈值，默认 `0.5` |
| `maxResults` | 最多返回的检测数量 |
| `poseKeyPointCount` | YOLO Pose 关键点数量，默认 COCO 的 `17` 点 |
| `inputWidth/inputHeight` | 模型输入尺寸，必须与 RKNN 模型一致 |
| `inputType` | 输入类型，默认 `AUTO` |
| `inputLayout` | 输入布局，默认 `AUTO` |
| `normalization` | RGB 归一化参数 |
| `decoderType` | 模型输出解码格式，建议生产配置显式指定 |
| `mediaPipeModel` | MediaPipe SSD 模型规格 |
| `modelType` | 模型业务类型 |

默认归一化公式：

```text
(pixel - mean) / std
mean = [0, 0, 0]
std  = [255, 255, 255]
```

如果模型图内部已经包含 `/255`，必须改为：

```kotlin
normalization = RknnNormalization(
    mean = floatArrayOf(0f, 0f, 0f),
    std = floatArrayOf(1f, 1f, 1f),
)
```

## 已支持的检测解码器

| 解码器 | 输出格式 |
|---|---|
| `YOLO_END_TO_END` | `[1,N,6]` 或 `[1,6,N]`，每行包含 `x1,y1,x2,y2,score,classId` |
| `YOLO_DETECT_RAW` | `[1,N,4+C]`、`[1,4+C,N]` 及带 objectness 的 `5+C` 输出 |
| `YOLO_DETECT_HEADS` | NCHW/NHWC 的合并或回归/分类分离 DFL 检测头 |
| `YOLO_POSE_LANDMARK` | `[1,N,6+K*3]` 或转置布局，包含检测框和关键点 |
| `YOLO_POSE_RAW` | 未执行 NMS 的 Raw Pose 单张量，支持有无 objectness |
| `YOLO_POSE_HEADS` | 按网格配对的 DFL 框、分类和关键点分支 |
| `YOLO_SEGMENT` | Raw 检测、mask coefficients 和 NCHW/NHWC prototype |
| `YOLO_CLASSIFY` | YOLO 分类概率或 logits 向量 |
| `YOLO_OBB` | 中心点、尺寸、类别分数和弧度角旋转框 |
| `MEDIA_PIPE_SSD` | MediaPipe Model Maker SSD/RetinaNet Box 和 Score 输出 |
| `AUTO` | 根据实际输出张量形状自动选择 |

多标签模型通过 `DetectorModel(multiLabel = true)` 开启。每个检测框仍只返回一个
`RknnDetection`，达到阈值的类别按分数降序保存在 `categories`。实例分割结果位于
`segmentationMask`，旋转框位于 `orientedBox`，同时保留普通 `boundingBox`。

## 目标跟踪

跟踪与模型解码相互独立，连续帧检测结果可交给 `RknnTracker`：

```kotlin
val tracker = RknnTracker(
    RknnTrackerConfig(
        highScoreThreshold = 0.6f,
        lowScoreThreshold = 0.1f,
        matchIouThreshold = 0.3f,
        minConfirmedFrames = 2,
        maxLostFrames = 30,
    ),
)

val tracked = tracker.update(detectionResult.detections)
tracker.reset()
```

`RknnTracker` 使用高低置信度二阶段关联。短暂丢失的轨迹以 `LOST` 状态保留，超过
`maxLostFrames` 后移除；Track 不是模型输出协议，因此不属于 `RknnDecoderType`。

生产环境建议显式设置 `decoderType`。`AUTO` 更适合调试未知模型输出。

### YOLO Pose 结果

YOLO Pose 关键点位于：

```kotlin
detection.keyPoints
```

每个关键点包含：

```kotlin
RknnKeypoint(
    index = 0,
    name = "nose",
    x = 100f,
    y = 120f,
    score = 0.9f,
)
```

坐标已经从模型 Letterbox 输入还原到原始 Bitmap 坐标系。

## MediaPipe Object Detector

支持以下 Model Maker 模型转换后的 RKNN 输出：

- `MOBILENET_V2`：256×256
- `MOBILENET_V2_I320`：320×320
- `MOBILENET_MULTI_AVG`：256×256
- `MOBILENET_MULTI_AVG_I384`：384×384

示例：

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

## 底层 RknnRuntime

需要图像分类、MediaPipe Pose Landmark、Hand Landmark 或原始张量时，使用 `RknnRuntime`。

```kotlin
val runtime = RknnRuntime()
runtime.initialize(
    context,
    RknnOptions(modelRoot = modelRoot, debug = false),
)

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
```

底层接口：

```kotlin
runtime.run(modelId, bitmap)                 // 原始张量
runtime.detectObjects(modelId, bitmap)       // 目标检测和 YOLO Pose
runtime.classifyImage(modelId, bitmap)        // 图像分类
runtime.detectPoseLandmarks(modelId, bitmap) // Pose Landmark ROI
runtime.detectHandLandmarks(modelId, bitmap) // Hand Landmark ROI
runtime.retryModel(modelId)                  // 显式重试失败的 Session
runtime.unregisterModel(modelId)
runtime.close()
```

Session 在 `registerModel()` 时立即创建。首次创建失败后不会逐帧重试，只有重新注册、重置 SDK 或调用 `retryModel()` 才会再次创建。

## 图像分类

支持以下 EfficientNet-Lite 分类输出：

- EfficientNet-Lite0 Float32
- EfficientNet-Lite0 Int8
- EfficientNet-Lite2 Float32
- EfficientNet-Lite2 Int8

分类结果支持概率直出、Logits Softmax、阈值过滤和 Top-K 排序。

## Pose 和 Hand Landmark

支持：

- Pose Landmark Lite、Full、Heavy
- Hand Landmark Full

这些解码器对应官方 `.task` 包中的 Landmark 子模型，不包含 detector、旋转 ROI 和视频跟踪管线。调用前必须提供 detector 裁剪并完成旋转矫正的 ROI Bitmap。

## Debug 日志

```kotlin
ModelConfig(models = models, debug = true)
```

`debug=true` 会输出：

- 芯片和 RKNPU 驱动版本
- RKNN Runtime/API 版本
- 模型文件、业务类型和解码器
- 输入输出张量名称、维度、类型、布局和量化参数
- 模型权重、内部内存、DMA 和 SRAM 信息
- 单次推理耗时与输出张量摘要

张量枚举同时显示名称和编号，例如：

```text
type=FP16(1)
format=NHWC(1)
format=UNDEFINED(3)
quantization=AFFINE(2)
```

`debug=false` 时不打印 JNI INFO 和逐帧调试日志，但真正的初始化、加载和推理错误仍会记录。

## 多模型调度建议

RK3588 同时运行 Action 和 Pose 时建议降低 Pose 频率：

```text
Action：80 ms，约 8～12 FPS
Pose：250 ms，约 3～4 FPS
```

不要将每个摄像头帧无条件提交给全部模型。应根据模型间隔调用 `detectAsync()`，并在返回 false 时回收未接收的 Bitmap。

## 常见问题

### `Supported detection output tensors not found`

实际输出形状与 `decoderType` 不匹配。开启 debug，查看：

```text
model tensor direction=output ... dims=[...]
```

例如 YOLO26 Pose 端到端模型输出 `[1,300,57]`，应配置：

```kotlin
decoderType = RknnDecoderType.YOLO_POSE_LANDMARK
```

### `openSession failed`

检查：

- 模型是否为目标芯片 `rk3588`
- RKNN-Toolkit2 与设备 Runtime 是否兼容
- 模型文件是否完整
- `/vendor/lib64/librknnrt.so` 是否存在
- RKNPU 驱动是否可用

### 检测框位置不正确

确认提交给推理的 Bitmap 尺寸与绘制覆盖层使用的原图尺寸一致。解码器返回的是提交 Bitmap 的坐标系，不是 `TextureView` 的物理像素坐标。

### 识别率明显降低

重点检查：

- 是否重复执行 `/255` 归一化
- RGB/BGR 顺序是否正确
- 输入尺寸是否与转换模型一致
- FP16/INT8 转换参数是否正确
- 标签数量是否与输出类别数量一致

## 构建与测试

```bash
./gradlew testDebugUnitTest
./gradlew assembleRelease
```

Windows：

```powershell
.\gradlew.bat testDebugUnitTest
.\gradlew.bat assembleRelease
```

Release AAR 生成在 `build/outputs/aar/` 目录中。

## 自动发布

项目版本统一配置在 `gradle.properties`，首个版本为 `1.0.0`。推送与项目版本一致的语义化标签（例如 `v1.0.0`）后，GitHub Actions 会自动运行单元测试、构建 Release AAR、生成 SHA-256 校验文件，并将两者上传到 GitHub Release。也可以手动运行工作流进行构建诊断，但手动运行不会创建 Release。

## 关联项目

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [RKNN-Toolkit2 Android Runtime API](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Android/librknn_api)

## 开源协议

本项目基于 [MIT License](LICENSE) 完全开源。
