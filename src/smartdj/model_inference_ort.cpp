/**
 * @file model_inference_ort.cpp
 * @brief ONNX Runtime C API wrapper for Myna model inference.
 *
 * Backend: ONNX Runtime (CPU). Used on Windows and Linux.
 * On macOS, model_inference_mlx.cpp provides the MLX Metal GPU backend.
 */

#include "model_inference.h"
#include "../ort_session.h"

#include <cstring>

namespace xune {
namespace smartdj {

struct ModelInference::Impl {
    OrtSession ort{"xune_embedding"};

    const char* input_name = "mel_spectrogram";
    const char* output_name = "embedding";
};

ModelInference::ModelInference() : impl_(std::make_unique<Impl>()) {}
ModelInference::~ModelInference() = default;
ModelInference::ModelInference(ModelInference&&) noexcept = default;
ModelInference& ModelInference::operator=(ModelInference&&) noexcept = default;

bool ModelInference::LoadModel(const std::string& model_path,
                               const std::string& cache_dir) {
    return impl_->ort.LoadModel(model_path, cache_dir, "myna_hybrid_opt.onnx");
}

bool ModelInference::IsReady() const {
    return impl_ && impl_->ort.IsReady();
}

const char* ModelInference::GetExecutionProvider() const {
    return impl_ ? impl_->ort.GetExecutionProvider() : "Unknown";
}

bool ModelInference::RunInferenceInto(const float* input_data, int batch_size,
                                      int n_mels, int n_frames,
                                      float* output_buffer, int output_buffer_size) {
    if (!impl_->ort.IsReady() || !input_data || batch_size <= 0 ||
        !output_buffer || output_buffer_size < batch_size * kEmbeddingDim) {
        return false;
    }

    auto& api = impl_->ort.ApiRef();

    // Input: (batch_size, 1, n_mels, n_frames)
    int64_t input_shape[4] = {batch_size, 1, n_mels, n_frames};
    size_t input_size = static_cast<size_t>(batch_size) * 1 * n_mels * n_frames * sizeof(float);

    OrtValue* input_tensor = nullptr;
    if (!impl_->ort.CheckStatus(
            api.CreateTensorWithDataAsOrtValue(
                impl_->ort.MemoryInfo(),
                const_cast<float*>(input_data),
                input_size, input_shape, 4,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &input_tensor))) {
        return false;
    }

    const char* input_names[] = {impl_->input_name};
    const char* output_names[] = {impl_->output_name};
    OrtValue* output_tensor = nullptr;

    bool ok = impl_->ort.CheckStatus(
        api.Run(impl_->ort.Session(), nullptr,
                input_names, (const OrtValue* const*)&input_tensor, 1,
                output_names, 1, &output_tensor));

    api.ReleaseValue(input_tensor);
    if (!ok) return false;

    float* output_data = nullptr;
    ok = impl_->ort.CheckStatus(
        api.GetTensorMutableData(output_tensor, (void**)&output_data));

    if (!ok) {
        api.ReleaseValue(output_tensor);
        return false;
    }

    size_t output_count = static_cast<size_t>(batch_size) * kEmbeddingDim;
    std::memcpy(output_buffer, output_data, output_count * sizeof(float));

    api.ReleaseValue(output_tensor);
    return true;
}

}  // namespace smartdj
}  // namespace xune
