/**
 * @file beat_inference.h
 * @brief Model inference wrapper for Beat This! beat tracking model.
 *
 * Runs the Beat This! small model on log-mel spectrogram chunks.
 * Input: (batch, time_frames, 128) log-mel spectrogram at 50 fps
 * Output: (batch, time_frames) beat logits + (batch, time_frames) downbeat logits
 *
 * Backend selected at compile time:
 *   - macOS: MLX (Metal GPU via Apple Silicon)
 *   - Windows/Linux: ONNX Runtime (CPU)
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xune {
namespace beattracking {

class BeatInference {
public:
    BeatInference();
    ~BeatInference();

    BeatInference(const BeatInference&) = delete;
    BeatInference& operator=(const BeatInference&) = delete;
    BeatInference(BeatInference&&) noexcept;
    BeatInference& operator=(BeatInference&&) noexcept;

    /**
     * Load the model from disk.
     *
     * @param model_path Path to model file (.safetensors for MLX, .onnx for ORT)
     * @return true on success
     */
    bool LoadModel(const std::string& model_path);

    bool IsReady() const;

    /**
     * Run inference on a mel spectrogram chunk.
     *
     * @param mel_data Input mel data, layout: float[batch_size * n_frames * 128], time-first
     * @param batch_size Number of items in the batch
     * @param n_frames Number of time frames per item
     * @param beat_logits Output beat logits, float[batch_size * n_frames]
     * @param downbeat_logits Output downbeat logits, float[batch_size * n_frames]
     * @return true on success
     */
    bool RunInference(const float* mel_data, int batch_size, int n_frames,
                      std::vector<float>& beat_logits,
                      std::vector<float>& downbeat_logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace beattracking
}  // namespace xune
