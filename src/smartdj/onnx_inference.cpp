/**
 * @file onnx_inference.cpp
 * @brief ONNX Runtime C API wrapper for Myna model inference.
 */

#include "onnx_inference.h"

#include <cstdio>
#include <cstring>
#include <onnxruntime_c_api.h>

namespace xune {
namespace smartdj {

// ============================================================================
// PIMPL: ONNX Runtime state
// ============================================================================

struct OnnxInference::Impl {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtSessionOptions* session_options = nullptr;

    // Cached input/output names from model metadata
    const char* input_name = "mel_spectrogram";
    const char* output_name = "embedding";

    bool ready = false;

    Impl() {
        api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }

    ~Impl() {
        if (session) api->ReleaseSession(session);
        if (session_options) api->ReleaseSessionOptions(session_options);
        if (env) api->ReleaseEnv(env);
    }

    bool CheckStatus(OrtStatus* status) {
        if (status != nullptr) {
            const char* msg = api->GetErrorMessage(status);
            fprintf(stderr, "[xune_embedding] ORT error: %s\n", msg);
            api->ReleaseStatus(status);
            return false;
        }
        return true;
    }
};

// ============================================================================
// OnnxInference
// ============================================================================

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}
OnnxInference::~OnnxInference() = default;
OnnxInference::OnnxInference(OnnxInference&&) noexcept = default;
OnnxInference& OnnxInference::operator=(OnnxInference&&) noexcept = default;

bool OnnxInference::LoadModel(const std::string& model_path) {
    auto& api = *impl_->api;

    // Create environment
    if (!impl_->CheckStatus(
            api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xune_smartdj", &impl_->env))) {
        return false;
    }

    // Create session options
    if (!impl_->CheckStatus(api.CreateSessionOptions(&impl_->session_options))) {
        return false;
    }

    // Limit ORT's internal thread pool. Inference is serialized through the C#
    // batch worker (one Run() at a time), so ORT only needs enough threads for
    // intra-op parallelism within a single call. Combined with 4 concurrent
    // FFmpeg decoders + mel computations, unlimited threads cause CPU thrashing.
    if (!impl_->CheckStatus(api.SetIntraOpNumThreads(impl_->session_options, 4))) {
        return false;
    }
    if (!impl_->CheckStatus(api.SetSessionGraphOptimizationLevel(impl_->session_options, ORT_ENABLE_ALL))) {
        return false;
    }

    // Create session (load model)
    // On POSIX, ORTCHAR_T is char. On Windows it's wchar_t.
#ifdef _WIN32
    // Convert UTF-8 to wide string for Windows
    int wlen = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, wpath.data(), wlen);
    if (!impl_->CheckStatus(
            api.CreateSession(impl_->env, wpath.data(),
                              impl_->session_options, &impl_->session))) {
        return false;
    }
#else
    if (!impl_->CheckStatus(
            api.CreateSession(impl_->env, model_path.c_str(),
                              impl_->session_options, &impl_->session))) {
        return false;
    }
#endif

    impl_->ready = true;
    return true;
}

bool OnnxInference::IsReady() const {
    return impl_ && impl_->ready;
}

bool OnnxInference::RunInference(const float* input_data, int batch_size,
                                  int n_mels, int n_frames,
                                  std::vector<float>& output) {
    if (!impl_->ready || !input_data || batch_size <= 0) {
        return false;
    }

    auto& api = *impl_->api;

    // Create memory info for CPU — use OrtDeviceAllocator (malloc/free) instead of
    // OrtArenaAllocator. The arena allocator grows to peak batch size and never
    // returns memory to the OS, causing steady RSS growth across 1500+ tracks
    // with varying batch sizes. The malloc/free overhead is negligible compared
    // to the ~50ms inference time per batch.
    OrtMemoryInfo* memory_info = nullptr;
    if (!impl_->CheckStatus(
            api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info))) {
        return false;
    }

    // Create input tensor
    // Shape: (batch_size, 1, n_mels, n_frames)
    int64_t input_shape[4] = {batch_size, 1, n_mels, n_frames};
    size_t input_size = static_cast<size_t>(batch_size) * 1 * n_mels * n_frames * sizeof(float);

    OrtValue* input_tensor = nullptr;
    bool ok = impl_->CheckStatus(
        api.CreateTensorWithDataAsOrtValue(
            memory_info,
            const_cast<float*>(input_data),  // ORT API takes non-const void*
            input_size,
            input_shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

    api.ReleaseMemoryInfo(memory_info);

    if (!ok) {
        return false;
    }

    // Run inference
    const char* input_names[] = {impl_->input_name};
    const char* output_names[] = {impl_->output_name};
    OrtValue* output_tensor = nullptr;

    ok = impl_->CheckStatus(
        api.Run(impl_->session, nullptr,
                input_names, (const OrtValue* const*)&input_tensor, 1,
                output_names, 1, &output_tensor));

    api.ReleaseValue(input_tensor);

    if (!ok) {
        return false;
    }

    // Read output data
    float* output_data = nullptr;
    ok = impl_->CheckStatus(
        api.GetTensorMutableData(output_tensor, (void**)&output_data));

    if (!ok) {
        api.ReleaseValue(output_tensor);
        return false;
    }

    // Copy output (batch_size * kEmbeddingDim floats)
    size_t output_count = static_cast<size_t>(batch_size) * kEmbeddingDim;
    output.resize(output_count);
    std::memcpy(output.data(), output_data, output_count * sizeof(float));

    api.ReleaseValue(output_tensor);
    return true;
}

}  // namespace smartdj
}  // namespace xune
