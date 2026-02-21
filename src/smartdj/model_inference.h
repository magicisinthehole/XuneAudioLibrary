/**
 * @file model_inference.h
 * @brief Model inference wrapper for Myna embedding model.
 *
 * Loads the Myna hybrid model and runs batched inference on
 * mel spectrogram chunks: (N, 1, 128, 96) -> (N, 768).
 *
 * Backend is selected at compile time:
 *   - macOS: MLX (Metal GPU via Apple Silicon unified memory)
 *   - Windows/Linux: ONNX Runtime (CPU)
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xune {
namespace smartdj {

class ModelInference {
public:
    static constexpr int kEmbeddingDim = 768;

    ModelInference();
    ~ModelInference();

    // Non-copyable, movable
    ModelInference(const ModelInference&) = delete;
    ModelInference& operator=(const ModelInference&) = delete;
    ModelInference(ModelInference&&) noexcept;
    ModelInference& operator=(ModelInference&&) noexcept;

    /**
     * Load the model from disk.
     *
     * @param model_path Path to model file (.safetensors for MLX, .onnx for ORT)
     * @param cache_dir Optional cache directory (used by ORT CoreML EP)
     * @return true on success
     */
    bool LoadModel(const std::string& model_path,
                   const std::string& cache_dir = "");

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
