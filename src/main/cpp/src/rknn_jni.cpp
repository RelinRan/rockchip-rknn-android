#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rknn_api.h"

#define LOG_TAG "rknn_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

// 运行时动态加载的 RKNN API，允许在不支持 RKNN 的设备上安全探测能力。
struct RknnApi {
    void* handle = nullptr;
    int (*init)(rknn_context*, void*, uint32_t, uint32_t, rknn_init_extend*) = nullptr;
    int (*destroy)(rknn_context) = nullptr;
    int (*query)(rknn_context, rknn_query_cmd, void*, uint32_t) = nullptr;
    int (*inputs_set)(rknn_context, uint32_t, rknn_input*) = nullptr;
    int (*run)(rknn_context, rknn_run_extend*) = nullptr;
    int (*outputs_get)(rknn_context, uint32_t, rknn_output*, rknn_output_extend*) = nullptr;
    int (*outputs_release)(rknn_context, uint32_t, rknn_output*) = nullptr;
    int (*dup_context)(rknn_context*, rknn_context*) = nullptr;
    int (*set_batch_core_num)(rknn_context, int) = nullptr;
    int (*set_core_mask)(rknn_context, rknn_core_mask) = nullptr;
    int (*wait)(rknn_context, rknn_run_extend*) = nullptr;
    rknn_tensor_mem* (*create_mem_from_phys)(rknn_context, uint64_t, void*, uint32_t) = nullptr;
    rknn_tensor_mem* (*create_mem_from_fd)(rknn_context, int32_t, void*, uint32_t, int32_t) = nullptr;
    rknn_tensor_mem* (*create_mem_from_mb_blk)(rknn_context, void*, int32_t) = nullptr;
    rknn_tensor_mem* (*create_mem)(rknn_context, uint32_t) = nullptr;
    rknn_tensor_mem* (*create_mem2)(rknn_context, uint64_t, uint64_t) = nullptr;
    int (*destroy_mem)(rknn_context, rknn_tensor_mem*) = nullptr;
    int (*set_weight_mem)(rknn_context, rknn_tensor_mem*) = nullptr;
    int (*set_internal_mem)(rknn_context, rknn_tensor_mem*) = nullptr;
    int (*set_io_mem)(rknn_context, rknn_tensor_mem*, rknn_tensor_attr*) = nullptr;
    int (*set_input_shape)(rknn_context, rknn_tensor_attr*) = nullptr;
    int (*set_input_shapes)(rknn_context, uint32_t, rknn_tensor_attr*) = nullptr;
    int (*mem_sync)(rknn_context, rknn_tensor_mem*, rknn_mem_sync_mode) = nullptr;
    bool loaded = false;
};

// 每个句柄独立持有 Context、张量属性以及跨帧复用的输入输出缓冲区。
struct Session {
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    std::string api_version;
    std::string driver_version;
    std::vector<uint8_t> uint8_input;
    std::vector<int8_t> int8_input;
    std::vector<uint16_t> float16_input;
    std::vector<float> float32_input;
    std::vector<std::vector<float>> output_buffers;
    std::vector<rknn_output> output_requests;
    bool debug = false;
};

RknnApi g_api;
std::mutex g_api_mutex;

std::string tensor_dims(const rknn_tensor_attr& attr) {
    std::string result = "[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i > 0) result += ",";
        result += std::to_string(attr.dims[i]);
    }
    result += "]";
    return result;
}

void log_tensor_attr(const char* direction, const rknn_tensor_attr& attr) {
    const std::string dims = tensor_dims(attr);
    LOGI(
        "model tensor direction=%s index=%u name=%s dims=%s elements=%u size=%u size_with_stride=%u "
        "type=%s(%d) format=%s(%d) quantization=%s(%d) zero_point=%d scale=%g fractional_length=%d "
        "width_stride=%u height_stride=%u pass_through=%u",
        direction, attr.index, attr.name, dims.c_str(), attr.n_elems, attr.size, attr.size_with_stride,
        get_type_string(attr.type), static_cast<int>(attr.type),
        get_format_string(attr.fmt), static_cast<int>(attr.fmt),
        get_qnt_type_string(attr.qnt_type), static_cast<int>(attr.qnt_type),
        attr.zp, attr.scale, attr.fl, attr.w_stride, attr.h_stride, attr.pass_through
    );
}

// 必选符号缺失意味着当前 librknnrt.so 无法提供基础推理能力。
bool load_symbol(void** target, const char* name) {
    dlerror();
    *target = dlsym(g_api.handle, name);
    if (*target == nullptr) {
        const char* error = dlerror();
        LOGE("dlsym failed symbol=%s error=%s", name, error != nullptr ? error : "unknown");
        return false;
    }
    return true;
}

// 可选符号用于扩展 API，旧版 Runtime 缺失时不影响基础推理。
void load_optional_symbol(void** target, const char* name) {
    dlerror();
    *target = dlsym(g_api.handle, name);
}

bool ensure_api_loaded() {
    std::lock_guard<std::mutex> guard(g_api_mutex);
    if (g_api.loaded) return true;
    const char* candidates[] = {
            "librknnrt.so",
            "/vendor/lib64/librknnrt.so",
            "/vendor/lib/librknnrt.so",
            "/system/lib64/librknnrt.so",
            "/system/lib/librknnrt.so",
    };
    for (const char* path : candidates) {
        dlerror();
        g_api.handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (g_api.handle != nullptr) {
            break;
        }
        (void)dlerror();
    }
    if (g_api.handle == nullptr) {
        LOGE("failed to load librknnrt.so");
        return false;
    }
    if (!load_symbol(reinterpret_cast<void**>(&g_api.init), "rknn_init")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.destroy), "rknn_destroy")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.query), "rknn_query")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.inputs_set), "rknn_inputs_set")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.run), "rknn_run")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.outputs_get), "rknn_outputs_get")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.outputs_release), "rknn_outputs_release")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.dup_context), "rknn_dup_context")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_batch_core_num), "rknn_set_batch_core_num")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_core_mask), "rknn_set_core_mask")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.wait), "rknn_wait")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.create_mem_from_phys), "rknn_create_mem_from_phys")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.create_mem_from_fd), "rknn_create_mem_from_fd")) return false;
    load_optional_symbol(reinterpret_cast<void**>(&g_api.create_mem_from_mb_blk), "rknn_create_mem_from_mb_blk");
    if (!load_symbol(reinterpret_cast<void**>(&g_api.create_mem), "rknn_create_mem")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.create_mem2), "rknn_create_mem2")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.destroy_mem), "rknn_destroy_mem")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_weight_mem), "rknn_set_weight_mem")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_internal_mem), "rknn_set_internal_mem")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_io_mem), "rknn_set_io_mem")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_input_shape), "rknn_set_input_shape")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.set_input_shapes), "rknn_set_input_shapes")) return false;
    if (!load_symbol(reinterpret_cast<void**>(&g_api.mem_sync), "rknn_mem_sync")) return false;
    g_api.loaded = true;
    return true;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::vector<uint8_t> data;
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return data;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    if (size <= 0) {
        fclose(fp);
        return data;
    }
    data.resize(static_cast<size_t>(size));
    fread(data.data(), 1, data.size(), fp);
    fclose(fp);
    return data;
}

jobject build_native_tensor_output(JNIEnv* env, int index, const rknn_tensor_attr& attr, const float* data, int count) {
    jclass cls = env->FindClass("androidx/runtime/rknn/internal/NativeTensorOutput");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;[III[F)V");
    jstring name = env->NewStringUTF(attr.name);
    jintArray dims = env->NewIntArray(static_cast<jsize>(attr.n_dims));
    std::vector<jint> dims_vec(attr.n_dims);
    for (uint32_t i = 0; i < attr.n_dims; ++i) dims_vec[i] = static_cast<jint>(attr.dims[i]);
    env->SetIntArrayRegion(dims, 0, static_cast<jsize>(dims_vec.size()), dims_vec.data());
    jfloatArray values = env->NewFloatArray(count);
    env->SetFloatArrayRegion(values, 0, count, data);
    jobject obj = env->NewObject(
            cls,
            ctor,
            index,
            name,
            dims,
            static_cast<jint>(attr.type),
            static_cast<jint>(attr.fmt),
            values
    );
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(dims);
    env->DeleteLocalRef(values);
    return obj;
}

jobject build_native_run_result(
        JNIEnv* env,
        bool success,
        jlong duration_ms,
        const std::string& message,
        const std::string& api_version,
        const std::string& driver_version,
        jobject outputs_list) {
    jclass cls = env->FindClass("androidx/runtime/rknn/internal/NativeRunResult");
    jmethodID ctor = env->GetMethodID(
            cls,
            "<init>",
            "(ZJLjava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/util/List;)V");
    jstring msg = message.empty() ? nullptr : env->NewStringUTF(message.c_str());
    jstring api = api_version.empty() ? nullptr : env->NewStringUTF(api_version.c_str());
    jstring drv = driver_version.empty() ? nullptr : env->NewStringUTF(driver_version.c_str());
    jobject result = env->NewObject(cls, ctor, success, duration_ms, msg, api, drv, outputs_list);
    if (msg) env->DeleteLocalRef(msg);
    if (api) env->DeleteLocalRef(api);
    if (drv) env->DeleteLocalRef(drv);
    return result;
}

jobject new_array_list(JNIEnv* env) {
    jclass cls = env->FindClass("java/util/ArrayList");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "()V");
    return env->NewObject(cls, ctor);
}

void array_list_add(JNIEnv* env, jobject list, jobject item) {
    jclass cls = env->GetObjectClass(list);
    jmethodID add = env->GetMethodID(cls, "add", "(Ljava/lang/Object;)Z");
    env->CallBooleanMethod(list, add, item);
}

uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 ((mantissa + 0x1000u) >> 13));
}

rknn_tensor_type resolve_input_type(int requested, const rknn_tensor_attr& attr) {
    if (requested == 0) return attr.type;
    switch (requested) {
        case 1: return RKNN_TENSOR_UINT8;
        case 2: return RKNN_TENSOR_INT8;
        case 3: return RKNN_TENSOR_FLOAT16;
        case 4: return RKNN_TENSOR_FLOAT32;
        default: return RKNN_TENSOR_TYPE_MAX;
    }
}

rknn_tensor_format resolve_input_format(int requested, const rknn_tensor_attr& attr) {
    if (requested == 0) {
        return attr.fmt == RKNN_TENSOR_NCHW ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
    }
    return requested == 2 ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
}

}  // namespace

extern "C"
JNIEXPORT jboolean JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeIsRuntimeAvailable(JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    return ensure_api_loaded() ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeRuntimeName(JNIEnv* env, jobject thiz) {
    (void)thiz;
    return env->NewStringUTF("rknn");
}

extern "C"
JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeOpenSession(
        JNIEnv* env, jobject thiz, jstring model_path_, jboolean debug) {
    (void)env;
    (void)thiz;
    if (!ensure_api_loaded()) {
        LOGE("open session aborted: runtime API unavailable");
        return 0;
    }
    const char* model_path = env->GetStringUTFChars(model_path_, nullptr);
    if (debug) LOGI("open session start model=%s", model_path);
    std::vector<uint8_t> model = read_file(model_path);
    env->ReleaseStringUTFChars(model_path_, model_path);
    if (model.empty()) {
        LOGE("model file empty");
        return 0;
    }
    if (debug) LOGI("model loaded bytes=%zu", model.size());

    auto* session = new Session();
    session->debug = debug == JNI_TRUE;
    int ret = g_api.init(&session->ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
    if (session->debug) {
        LOGI("rknn_init returned ret=%d context=%llu", ret, static_cast<unsigned long long>(session->ctx));
    }
    if (ret != RKNN_SUCC) {
        LOGE("rknn_init failed ret=%d", ret);
        delete session;
        return 0;
    }

    ret = g_api.query(session->ctx, RKNN_QUERY_IN_OUT_NUM, &session->io_num, sizeof(session->io_num));
    if (ret != RKNN_SUCC) {
        LOGE("query io num failed ret=%d", ret);
        g_api.destroy(session->ctx);
        delete session;
        return 0;
    }

    session->input_attrs.resize(session->io_num.n_input);
    for (uint32_t i = 0; i < session->io_num.n_input; ++i) {
        session->input_attrs[i].index = i;
        ret = g_api.query(session->ctx, RKNN_QUERY_INPUT_ATTR, &session->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOGE("query input attr failed index=%u ret=%d", i, ret);
            g_api.destroy(session->ctx);
            delete session;
            return 0;
        }
    }
    session->output_attrs.resize(session->io_num.n_output);
    session->output_buffers.resize(session->io_num.n_output);
    session->output_requests.resize(session->io_num.n_output);
    for (uint32_t i = 0; i < session->io_num.n_output; ++i) {
        session->output_attrs[i].index = i;
        ret = g_api.query(session->ctx, RKNN_QUERY_OUTPUT_ATTR, &session->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOGE("query output attr failed index=%u ret=%d", i, ret);
            g_api.destroy(session->ctx);
            delete session;
            return 0;
        }
        session->output_buffers[i].resize(session->output_attrs[i].n_elems);
    }

    rknn_sdk_version version{};
    ret = g_api.query(session->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret == RKNN_SUCC) {
        session->api_version = version.api_version;
        session->driver_version = version.drv_version;
    }

    if (session->debug) {
        LOGI(
            "model session ready inputs=%u outputs=%u api_version=%s driver_version=%s",
            session->io_num.n_input, session->io_num.n_output,
            session->api_version.empty() ? "unknown" : session->api_version.c_str(),
            session->driver_version.empty() ? "unknown" : session->driver_version.c_str()
        );
        for (const auto& attr : session->input_attrs) log_tensor_attr("input", attr);
        for (const auto& attr : session->output_attrs) log_tensor_attr("output", attr);
    }

    rknn_mem_size memory{};
    ret = g_api.query(session->ctx, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory));
    if (ret == RKNN_SUCC && session->debug) {
        LOGI(
            "model memory weight=%u internal=%u dma=%llu sram_total=%u sram_free=%u",
            memory.total_weight_size, memory.total_internal_size,
            static_cast<unsigned long long>(memory.total_dma_allocated_size),
            memory.total_sram_size, memory.free_sram_size
        );
    } else if (session->debug) {
        LOGI("model memory unavailable ret=%d", ret);
    }

    return reinterpret_cast<jlong>(session);
}

extern "C"
JNIEXPORT void JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeCloseSession(JNIEnv* env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session) return;
    if (ensure_api_loaded() && session->ctx != 0) {
        g_api.destroy(session->ctx);
    }
    delete session;
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeDuplicateSession(JNIEnv*, jobject, jlong handle) {
    auto* source = reinterpret_cast<Session*>(handle);
    if (!source || !ensure_api_loaded()) return 0;
    auto* duplicate = new Session(*source);
    duplicate->ctx = 0;
    if (g_api.dup_context(&source->ctx, &duplicate->ctx) != RKNN_SUCC) {
        delete duplicate;
        return 0;
    }
    return reinterpret_cast<jlong>(duplicate);
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetBatchCoreNumber(JNIEnv*, jobject, jlong handle, jint count) {
    auto* session = reinterpret_cast<Session*>(handle);
    return session && ensure_api_loaded() ? g_api.set_batch_core_num(session->ctx, count) : RKNN_ERR_CTX_INVALID;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetCoreMask(JNIEnv*, jobject, jlong handle, jint mask) {
    auto* session = reinterpret_cast<Session*>(handle);
    return session && ensure_api_loaded() ? g_api.set_core_mask(session->ctx, static_cast<rknn_core_mask>(mask)) : RKNN_ERR_CTX_INVALID;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeWait(JNIEnv*, jobject, jlong handle, jlong frame_id) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded()) return RKNN_ERR_CTX_INVALID;
    rknn_run_extend extend{};
    extend.frame_id = static_cast<uint64_t>(frame_id);
    return g_api.wait(session->ctx, &extend);
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeExecute(JNIEnv*, jobject, jlong handle, jboolean non_blocking,
                                                          jlong frame_id) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded()) return RKNN_ERR_CTX_INVALID;
    rknn_run_extend extend{};
    extend.non_block = non_blocking ? 1 : 0;
    extend.frame_id = static_cast<uint64_t>(frame_id);
    return g_api.run(session->ctx, &extend);
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeCreateMemory(JNIEnv*, jobject, jlong handle, jlong size, jlong flags) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded() || size <= 0) return 0;
    rknn_tensor_mem* memory = flags == 0
        ? g_api.create_mem(session->ctx, static_cast<uint32_t>(size))
        : g_api.create_mem2(session->ctx, static_cast<uint64_t>(size), static_cast<uint64_t>(flags));
    return reinterpret_cast<jlong>(memory);
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeCreateMemoryFromFd(JNIEnv*, jobject, jlong handle, jint fd, jint size, jint offset) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded() || fd < 0 || size <= 0 || offset < 0) return 0;
    return reinterpret_cast<jlong>(g_api.create_mem_from_fd(session->ctx, fd, nullptr, static_cast<uint32_t>(size), offset));
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeCreateMemoryFromPhysical(JNIEnv* env, jobject, jlong handle,
                                                                           jlong physical, jobject buffer, jint size) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded() || physical == 0 || size <= 0) return 0;
    void* address = buffer ? env->GetDirectBufferAddress(buffer) : nullptr;
    if (buffer && !address) return 0;
    return reinterpret_cast<jlong>(g_api.create_mem_from_phys(session->ctx, static_cast<uint64_t>(physical), address,
                                                              static_cast<uint32_t>(size)));
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeCreateMemoryFromMbBlock(JNIEnv*, jobject, jlong handle, jlong block, jint offset) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded() || !g_api.create_mem_from_mb_blk || block == 0 || offset < 0) return 0;
    return reinterpret_cast<jlong>(g_api.create_mem_from_mb_blk(session->ctx, reinterpret_cast<void*>(block), offset));
}

extern "C" JNIEXPORT jobject JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeGetMemoryBuffer(JNIEnv* env, jobject, jlong memory_handle) {
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return memory && memory->virt_addr ? env->NewDirectByteBuffer(memory->virt_addr, memory->size) : nullptr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeGetMemorySize(JNIEnv*, jobject, jlong memory_handle) {
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return memory ? static_cast<jlong>(memory->size) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeDestroyMemory(JNIEnv*, jobject, jlong handle, jlong memory_handle) {
    auto* session = reinterpret_cast<Session*>(handle);
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return session && memory && ensure_api_loaded() ? g_api.destroy_mem(session->ctx, memory) : RKNN_ERR_PARAM_INVALID;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetWeightMemory(JNIEnv*, jobject, jlong handle, jlong memory_handle) {
    auto* session = reinterpret_cast<Session*>(handle);
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return session && memory && ensure_api_loaded() ? g_api.set_weight_mem(session->ctx, memory) : RKNN_ERR_PARAM_INVALID;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetInternalMemory(JNIEnv*, jobject, jlong handle, jlong memory_handle) {
    auto* session = reinterpret_cast<Session*>(handle);
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return session && memory && ensure_api_loaded() ? g_api.set_internal_mem(session->ctx, memory) : RKNN_ERR_PARAM_INVALID;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetIoMemory(JNIEnv*, jobject, jlong handle, jlong memory_handle,
                                                              jint index, jboolean input) {
    auto* session = reinterpret_cast<Session*>(handle);
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    if (!session || !memory || !ensure_api_loaded() || index < 0) return RKNN_ERR_PARAM_INVALID;
    auto& attrs = input ? session->input_attrs : session->output_attrs;
    if (static_cast<size_t>(index) >= attrs.size()) return RKNN_ERR_PARAM_INVALID;
    return g_api.set_io_mem(session->ctx, memory, &attrs[index]);
}

bool set_dimensions(JNIEnv* env, jintArray dimensions, rknn_tensor_attr* attr) {
    const jsize count = env->GetArrayLength(dimensions);
    if (count <= 0 || count > RKNN_MAX_DIMS) return false;
    std::vector<jint> values(count);
    env->GetIntArrayRegion(dimensions, 0, count, values.data());
    for (jsize i = 0; i < count; ++i) {
        if (values[i] <= 0) return false;
        attr->dims[i] = static_cast<uint32_t>(values[i]);
    }
    attr->n_dims = static_cast<uint32_t>(count);
    return true;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetInputShape(JNIEnv* env, jobject, jlong handle, jint index,
                                                                jintArray dimensions) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded() || index < 0 || static_cast<size_t>(index) >= session->input_attrs.size())
        return RKNN_ERR_PARAM_INVALID;
    rknn_tensor_attr attr = session->input_attrs[index];
    if (!set_dimensions(env, dimensions, &attr)) return RKNN_ERR_PARAM_INVALID;
    const int result = g_api.set_input_shape(session->ctx, &attr);
    if (result == RKNN_SUCC) session->input_attrs[index] = attr;
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSetInputShapes(JNIEnv* env, jobject, jlong handle,
                                                                 jintArray indices, jobjectArray dimensions) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded()) return RKNN_ERR_CTX_INVALID;
    const jsize count = env->GetArrayLength(indices);
    if (count <= 0 || env->GetArrayLength(dimensions) != count) return RKNN_ERR_PARAM_INVALID;
    std::vector<jint> index_values(count);
    env->GetIntArrayRegion(indices, 0, count, index_values.data());
    std::vector<rknn_tensor_attr> attrs(count);
    for (jsize i = 0; i < count; ++i) {
        const int index = index_values[i];
        if (index < 0 || static_cast<size_t>(index) >= session->input_attrs.size()) return RKNN_ERR_PARAM_INVALID;
        attrs[i] = session->input_attrs[index];
        auto* shape = static_cast<jintArray>(env->GetObjectArrayElement(dimensions, i));
        const bool valid = shape && set_dimensions(env, shape, &attrs[i]);
        if (shape) env->DeleteLocalRef(shape);
        if (!valid) return RKNN_ERR_PARAM_INVALID;
    }
    const int result = g_api.set_input_shapes(session->ctx, static_cast<uint32_t>(count), attrs.data());
    if (result == RKNN_SUCC) for (jsize i = 0; i < count; ++i) session->input_attrs[index_values[i]] = attrs[i];
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeSynchronizeMemory(JNIEnv*, jobject, jlong handle,
                                                                    jlong memory_handle, jint mode) {
    auto* session = reinterpret_cast<Session*>(handle);
    auto* memory = reinterpret_cast<rknn_tensor_mem*>(memory_handle);
    return session && memory && ensure_api_loaded()
        ? g_api.mem_sync(session->ctx, memory, static_cast<rknn_mem_sync_mode>(mode)) : RKNN_ERR_PARAM_INVALID;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_androidx_runtime_rknn_internal_RknnJni_nativeRun(JNIEnv* env, jobject thiz, jlong handle, jbyteArray image_bytes_,
                                              jint width, jint height, jint channels, jint requested_type,
                                              jint requested_layout, jfloatArray mean_, jfloatArray std_) {
    (void)thiz;
    auto start = std::chrono::steady_clock::now();
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || !ensure_api_loaded()) {
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "RKNN session unavailable", "", "", outputs);
    }

    jsize length = env->GetArrayLength(image_bytes_);
    if (channels != 3 || width <= 0 || height <= 0 ||
        length != static_cast<jsize>(width * height * channels) ||
        session->io_num.n_input != 1) {
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "Invalid RGB input shape or model input count", session->api_version, session->driver_version, outputs);
    }
    const auto& input_attr = session->input_attrs[0];
    if (input_attr.n_dims >= 4) {
        const bool shape_matches_nhwc = input_attr.dims[1] == static_cast<uint32_t>(height) &&
                                        input_attr.dims[2] == static_cast<uint32_t>(width) &&
                                        input_attr.dims[3] == static_cast<uint32_t>(channels);
        const bool shape_matches_nchw = input_attr.dims[1] == static_cast<uint32_t>(channels) &&
                                        input_attr.dims[2] == static_cast<uint32_t>(height) &&
                                        input_attr.dims[3] == static_cast<uint32_t>(width);
        if (!shape_matches_nhwc && !shape_matches_nchw) {
            jobject outputs = new_array_list(env);
            return build_native_run_result(env, false, 0, "Input dimensions do not match RKNN model", session->api_version, session->driver_version, outputs);
        }
    }
    if (env->GetArrayLength(mean_) != 3 || env->GetArrayLength(std_) != 3) {
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "Normalization requires three channels", session->api_version, session->driver_version, outputs);
    }
    float mean[3];
    float std_values[3];
    env->GetFloatArrayRegion(mean_, 0, 3, mean);
    env->GetFloatArrayRegion(std_, 0, 3, std_values);
    for (float value : std_values) {
        if (!std::isfinite(value) || value == 0.0f) {
            jobject outputs = new_array_list(env);
            return build_native_run_result(env, false, 0, "Invalid normalization divisor", session->api_version, session->driver_version, outputs);
        }
    }
    jbyte* bytes = env->GetByteArrayElements(image_bytes_, nullptr);
    const auto* rgb = reinterpret_cast<const uint8_t*>(bytes);
    const size_t element_count = static_cast<size_t>(width) * height * channels;
    const rknn_tensor_type input_type = resolve_input_type(requested_type, input_attr);
    const rknn_tensor_format input_format = resolve_input_format(requested_layout, input_attr);
    auto& uint8_input = session->uint8_input;
    auto& int8_input = session->int8_input;
    auto& float16_input = session->float16_input;
    auto& float32_input = session->float32_input;
    void* input_data = nullptr;
    uint32_t input_size = 0;
    auto destination_index = [=](int y, int x, int channel) -> size_t {
        if (input_format == RKNN_TENSOR_NCHW) {
            return static_cast<size_t>(channel) * height * width + static_cast<size_t>(y) * width + x;
        }
        return (static_cast<size_t>(y) * width + x) * channels + channel;
    };
    if (input_type == RKNN_TENSOR_UINT8) uint8_input.resize(element_count);
    else if (input_type == RKNN_TENSOR_INT8) int8_input.resize(element_count);
    else if (input_type == RKNN_TENSOR_FLOAT16) float16_input.resize(element_count);
    else if (input_type == RKNN_TENSOR_FLOAT32) float32_input.resize(element_count);
    else {
        env->ReleaseByteArrayElements(image_bytes_, bytes, JNI_ABORT);
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "Unsupported RKNN input tensor type", session->api_version, session->driver_version, outputs);
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int channel = 0; channel < channels; ++channel) {
        const size_t source = (static_cast<size_t>(y) * width + x) * channels + channel;
        const size_t destination = destination_index(y, x, channel);
        const float normalized = (static_cast<float>(rgb[source]) - mean[channel]) / std_values[channel];
        if (input_type == RKNN_TENSOR_UINT8) {
            uint8_input[destination] = rgb[source];
        } else if (input_type == RKNN_TENSOR_INT8) {
            const float scale = input_attr.scale == 0.0f ? 1.0f : input_attr.scale;
            const int quantized = static_cast<int>(std::lround(normalized / scale)) + input_attr.zp;
            int8_input[destination] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        } else if (input_type == RKNN_TENSOR_FLOAT16) {
            float16_input[destination] = float_to_half(normalized);
        } else {
            float32_input[destination] = normalized;
        }
    }
    env->ReleaseByteArrayElements(image_bytes_, bytes, JNI_ABORT);
    if (input_type == RKNN_TENSOR_UINT8) { input_data = uint8_input.data(); input_size = static_cast<uint32_t>(uint8_input.size()); }
    else if (input_type == RKNN_TENSOR_INT8) { input_data = int8_input.data(); input_size = static_cast<uint32_t>(int8_input.size()); }
    else if (input_type == RKNN_TENSOR_FLOAT16) { input_data = float16_input.data(); input_size = static_cast<uint32_t>(float16_input.size() * sizeof(uint16_t)); }
    else { input_data = float32_input.data(); input_size = static_cast<uint32_t>(float32_input.size() * sizeof(float)); }

    rknn_input input{};
    input.index = 0;
    input.buf = input_data;
    input.size = input_size;
    input.pass_through = 0;
    input.type = input_type;
    input.fmt = input_format;

    if (session->debug) {
        LOGI(
            "input prepared type=%s(%d) format=%s(%d) model_type=%s(%d) model_format=%s(%d) bytes=%u",
            get_type_string(input.type), static_cast<int>(input.type),
            get_format_string(input.fmt), static_cast<int>(input.fmt),
            get_type_string(input_attr.type), static_cast<int>(input_attr.type),
            get_format_string(input_attr.fmt), static_cast<int>(input_attr.fmt), input.size
        );
    }

    int ret = g_api.inputs_set(session->ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "rknn_inputs_set failed", session->api_version, session->driver_version, outputs);
    }

    rknn_run_extend run_extend{};
    run_extend.non_block = 0;
    run_extend.timeout_ms = 0;
    ret = g_api.run(session->ctx, &run_extend);
    if (ret != RKNN_SUCC) {
        jobject outputs = new_array_list(env);
        return build_native_run_result(env, false, 0, "rknn_run failed", session->api_version, session->driver_version, outputs);
    }

    auto& outputs = session->output_requests;
    for (uint32_t i = 0; i < session->io_num.n_output; ++i) {
        outputs[i] = {};
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 1;
        outputs[i].buf = session->output_buffers[i].data();
        outputs[i].size = static_cast<uint32_t>(session->output_buffers[i].size() * sizeof(float));
    }

    ret = g_api.outputs_get(session->ctx, session->io_num.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        jobject list = new_array_list(env);
        return build_native_run_result(env, false, 0, "rknn_outputs_get failed", session->api_version, session->driver_version, list);
    }

    jobject list = new_array_list(env);
    for (uint32_t i = 0; i < session->io_num.n_output; ++i) {
        const auto& attr = session->output_attrs[i];
        int count = static_cast<int>(attr.n_elems);
        auto* ptr = reinterpret_cast<float*>(outputs[i].buf);
        jobject tensor = build_native_tensor_output(env, static_cast<int>(i), attr, ptr, count);
        array_list_add(env, list, tensor);
        env->DeleteLocalRef(tensor);
    }
    g_api.outputs_release(session->ctx, session->io_num.n_output, outputs.data());

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return build_native_run_result(env, true, static_cast<jlong>(duration), "", session->api_version, session->driver_version, list);
}
