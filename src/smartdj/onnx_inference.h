/**
 * @file onnx_inference.h
 * @brief ONNX Runtime wrapper for Myna model inference.
 *
 * Loads the myna_hybrid.onnx model and runs batched inference on
 * mel spectrogram chunks: (N, 1, 128, 96) -> (N, 768).
 *
 * Thread-safe: the ORT session handles concurrent inference.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xune {
namespace smartdj {

class OnnxInference {
public:
    static constexpr int kEmbeddingDim = 768;

    OnnxInference();
    ~OnnxInference();

    // Non-copyable, movable
    OnnxInference(const OnnxInference&) = delete;
    OnnxInference& operator=(const OnnxInference&) = delete;
    OnnxInference(OnnxInference&&) noexcept;
    OnnxInference& operator=(OnnxInference&&) noexcept;

    /**
     * Load the ONNX model from disk.
     *
     * @param model_path Path to myna_hybrid.onnx
     * @return true on success
     */
    bool LoadModel(const std::string& model_path);

    /**
     * Check if the model is loaded and ready for inference.
     */
    bool IsReady() const;

    /**
     * Run inference on a batch of mel spectrogram chunks.
     *
     * @param input_data Batch of mel chunks, layout: float[batch_size * 1 * n_mels * n_frames]
     * @param batch_size Number of chunks in the batch
     * @param n_mels Number of mel bands (128)
     * @param n_frames Number of time frames per chunk (96)
     * @param output Receives the output embeddings, layout: float[batch_size * kEmbeddingDim]
     * @return true on success
     */
    bool RunInference(const float* input_data, int batch_size,
                      int n_mels, int n_frames,
                      std::vector<float>& output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace smartdj
}  // namespace xune
