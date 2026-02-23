/**
 * @file beat_inference_ort.cpp
 * @brief ONNX Runtime C API backend for Beat This! beat tracking inference.
 *
 * Backend: ONNX Runtime (CPU). Used on Windows and Linux.
 * On macOS, beat_inference_mlx.cpp provides the MLX Metal GPU backend.
 *
 * Input: (batch, time_frames, 128) log-mel spectrogram
 * Output: (batch, time_frames) beat logits + (batch, time_frames) downbeat logits
 */

#include "beat_inference.h"

#include <cstdio>
#include <cstring>
#include <onnxruntime_c_api.h>

namespace xune {
namespace beattracking {

struct BeatInference::Impl {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtSessionOptions* session_options = nullptr;

    const char* input_name = "spectrogram";
    const char* output_names[2] = {"beat_logits", "downbeat_logits"};

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
            fprintf(stderr, "[xune_beat] ORT error: %s\n", msg);
            api->ReleaseStatus(status);
            return false;
        }
        return true;
    }
};

BeatInference::BeatInference() : impl_(std::make_unique<Impl>()) {}
BeatInference::~BeatInference() = default;
BeatInference::BeatInference(BeatInference&&) noexcept = default;
BeatInference& BeatInference::operator=(BeatInference&&) noexcept = default;

bool BeatInference::LoadModel(const std::string& model_path) {
    auto& api = *impl_->api;

    if (!impl_->CheckStatus(
            api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xune_beat", &impl_->env))) {
        return false;
    }

    if (!impl_->CheckStatus(api.CreateSessionOptions(&impl_->session_options))) {
        return false;
    }

    if (!impl_->CheckStatus(api.SetIntraOpNumThreads(impl_->session_options, 4))) {
        return false;
    }
    if (!impl_->CheckStatus(api.SetSessionGraphOptimizationLevel(impl_->session_options, ORT_ENABLE_ALL))) {
        return false;
    }

#ifdef _WIN32
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
    fprintf(stderr, "[xune_beat] ORT model loaded: %s\n", model_path.c_str());
    return true;
}

bool BeatInference::IsReady() const {
    return impl_ && impl_->ready;
}

bool BeatInference::RunInference(const float* mel_data, int batch_size, int n_frames,
                                  std::vector<float>& beat_logits,
                                  std::vector<float>& downbeat_logits) {
    if (!impl_->ready || !mel_data || batch_size <= 0 || n_frames <= 0) {
        return false;
    }

    auto& api = *impl_->api;

    OrtMemoryInfo* memory_info = nullptr;
    if (!impl_->CheckStatus(
            api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info))) {
        return false;
    }

    // Input: (batch_size, n_frames, 128)
    int64_t input_shape[3] = {batch_size, n_frames, 128};
    size_t input_size = static_cast<size_t>(batch_size) * n_frames * 128 * sizeof(float);

    OrtValue* input_tensor = nullptr;
    bool ok = impl_->CheckStatus(
        api.CreateTensorWithDataAsOrtValue(
            memory_info,
            const_cast<float*>(mel_data),
            input_size,
            input_shape, 3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

    api.ReleaseMemoryInfo(memory_info);

    if (!ok) {
        return false;
    }

    // Run inference — two outputs: beat_logits, downbeat_logits
    const char* input_names[] = {impl_->input_name};
    OrtValue* output_tensors[2] = {nullptr, nullptr};

    ok = impl_->CheckStatus(
        api.Run(impl_->session, nullptr,
                input_names, (const OrtValue* const*)&input_tensor, 1,
                impl_->output_names, 2, output_tensors));

    api.ReleaseValue(input_tensor);

    if (!ok) {
        return false;
    }

    // Copy outputs
    size_t output_count = static_cast<size_t>(batch_size) * n_frames;

    float* beat_data = nullptr;
    ok = impl_->CheckStatus(api.GetTensorMutableData(output_tensors[0], (void**)&beat_data));
    if (ok) {
        beat_logits.resize(output_count);
        std::memcpy(beat_logits.data(), beat_data, output_count * sizeof(float));
    }

    float* db_data = nullptr;
    ok = ok && impl_->CheckStatus(api.GetTensorMutableData(output_tensors[1], (void**)&db_data));
    if (ok) {
        downbeat_logits.resize(output_count);
        std::memcpy(downbeat_logits.data(), db_data, output_count * sizeof(float));
    }

    api.ReleaseValue(output_tensors[0]);
    api.ReleaseValue(output_tensors[1]);
    return ok;
}

}  // namespace beattracking
}  // namespace xune
