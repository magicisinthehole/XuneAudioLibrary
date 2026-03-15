/**
 * @file beat_mel_spectrogram.h
 * @brief Mel spectrogram computation for Beat This! beat tracking pipeline.
 *
 * Computes log-mel spectrograms from mono 22050 Hz float32 PCM audio.
 * Uses vDSP (Accelerate) on macOS, FFTW3 on Linux, for FFT computation.
 *
 * Parameters match Beat This! LogMelSpect (torchaudio MelSpectrogram) exactly:
 *   - sr=22050, n_fft=1024, hop_length=441 (50 fps), n_mels=128
 *   - Hann window, center=true (reflect-pad 512 each side)
 *   - Power=1.0 (magnitude), Slaney mel scale, area-normalized
 *   - fmin=30, fmax=11000
 *   - normalized="frame_length" (÷ n_fft)
 *   - log1p(1000 × mel) transform
 *
 * Output layout: [n_frames x n_mels] row-major (time-first), matching the
 * Beat This! model input shape (B, T, 128).
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <memory>

namespace xune {
namespace beattracking {

class BeatMelSpectrogram {
public:
    static constexpr int kSampleRate = 22050;
    static constexpr int kNFft = 1024;
    static constexpr int kHopLength = 441;  // 22050 / 50 = 441 → 50 fps
    static constexpr int kNMels = 128;
    static constexpr int kFreqBins = kNFft / 2 + 1;  // 513
    static constexpr float kFmin = 30.0f;
    static constexpr float kFmax = 11000.0f;
    static constexpr float kLogMultiplier = 1000.0f;

    /// Reusable scratch memory for Compute(). Vectors only grow, never shrink.
    struct ScratchBuffer {
        std::vector<float> padded;
        std::vector<float> windowed_frame;
        std::vector<float> mag_spectrum;
    };

    BeatMelSpectrogram();
    ~BeatMelSpectrogram();

    // Non-copyable, movable
    BeatMelSpectrogram(const BeatMelSpectrogram&) = delete;
    BeatMelSpectrogram& operator=(const BeatMelSpectrogram&) = delete;
    BeatMelSpectrogram(BeatMelSpectrogram&&) noexcept;
    BeatMelSpectrogram& operator=(BeatMelSpectrogram&&) noexcept;

    /**
     * Compute log-mel spectrogram from mono 22050 Hz float32 PCM.
     *
     * @param pcm Input PCM samples (mono, 22050 Hz, float32)
     * @param num_samples Number of PCM samples
     * @param out_mel Output: [n_frames x n_mels] row-major (time-first)
     * @param out_n_frames Number of output time frames
     * @return true on success
     */
    bool Compute(const float* pcm, int num_samples,
                 std::vector<float>& out_mel, int& out_n_frames) const;

    /**
     * Compute with caller-supplied scratch buffer to avoid heap churn.
     * @param cancel Optional cancellation flag — checked every frame.
     */
    bool Compute(const float* pcm, int num_samples,
                 std::vector<float>& out_mel, int& out_n_frames,
                 ScratchBuffer& scratch,
                 const std::atomic<bool>* cancel = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<float> hann_window_;      // Hann window [kNFft]
    std::vector<float> mel_filterbank_;   // Mel filterbank [kNMels x kFreqBins]

    void InitWindow();
    void InitMelFilterbank();

    static float HzToMel(float hz);
    static float MelToHz(float mel);
};

}  // namespace beattracking
}  // namespace xune
