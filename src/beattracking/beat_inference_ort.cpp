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
#include "../ort_session.h"

#include <cstring>

namespace xune {
namespace beattracking {

struct BeatInference::Impl {
    OrtSession ort{"xune_beat"};

    const char* input_name = "spectrogram";
    const char* output_names[2] = {"beat_logits", "downbeat_logits"};
};

BeatInference::BeatInference() : impl_(std::make_unique<Impl>()) {}
BeatInference::~BeatInference() = default;
BeatInference::BeatInference(BeatInference&&) noexcept = default;
BeatInference& BeatInference::operator=(BeatInference&&) noexcept = default;

bool BeatInference::LoadModel(const std::string& model_path,
                               const std::string& cache_dir) {
    return impl_->ort.LoadModel(model_path, cache_dir, "beat_this_small_opt.onnx");
}

bool BeatInference::IsReady() const {
    return impl_ && impl_->ort.IsReady();
}

const char* BeatInference::GetExecutionProvider() const {
    return impl_ ? impl_->ort.GetExecutionProvider() : "Unknown";
}

bool BeatInference::RunInference(const float* mel_data, int batch_size, int n_frames,
                                  std::vector<float>& beat_logits,
                                  std::vector<float>& downbeat_logits) {
    if (!impl_->ort.IsReady() || !mel_data || batch_size <= 0 || n_frames <= 0) {
        return false;
    }

    auto& api = impl_->ort.ApiRef();

    // Input: (batch_size, n_frames, 128)
    int64_t input_shape[3] = {batch_size, n_frames, 128};
    size_t input_size = static_cast<size_t>(batch_size) * n_frames * 128 * sizeof(float);

    OrtValue* input_tensor = nullptr;
    if (!impl_->ort.CheckStatus(
            api.CreateTensorWithDataAsOrtValue(
                impl_->ort.MemoryInfo(),
                const_cast<float*>(mel_data),
                input_size, input_shape, 3,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &input_tensor))) {
        return false;
    }

    const char* input_names[] = {impl_->input_name};
    OrtValue* output_tensors[2] = {nullptr, nullptr};

    bool ok = impl_->ort.CheckStatus(
        api.Run(impl_->ort.Session(), nullptr,
                input_names, (const OrtValue* const*)&input_tensor, 1,
                impl_->output_names, 2, output_tensors));

    api.ReleaseValue(input_tensor);
    if (!ok) return false;

    size_t output_count = static_cast<size_t>(batch_size) * n_frames;

    float* beat_data = nullptr;
    ok = impl_->ort.CheckStatus(api.GetTensorMutableData(output_tensors[0], (void**)&beat_data));
    if (ok) {
        beat_logits.resize(output_count);
        std::memcpy(beat_logits.data(), beat_data, output_count * sizeof(float));
    }

    float* db_data = nullptr;
    ok = ok && impl_->ort.CheckStatus(api.GetTensorMutableData(output_tensors[1], (void**)&db_data));
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
