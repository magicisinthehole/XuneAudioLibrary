/**
 * @file mel_spectrogram.h
 * @brief Mel spectrogram computation for SmartDJ embedding pipeline.
 *
 * Computes power mel spectrograms from mono 16kHz float32 PCM audio.
 * Uses vDSP (Accelerate) on macOS, FFTW3 on Linux, for FFT computation.
 *
 * Parameters match nnAudio's MelSpectrogram(sr=16000, n_mels=128) exactly:
 *   - sr=16000, n_fft=2048, hop_length=512, n_mels=128
 *   - Hann window, center=true (reflect-pad 1024 each side)
 *   - Power=2.0 (magnitude squared), Slaney mel scale, area-normalized
 *   - fmin=0, fmax=8000, NO log transform
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>

namespace xune {
namespace smartdj {

class MelSpectrogram {
public:
    // Parameters matching nnAudio defaults for sr=16000
    static constexpr int kSampleRate = 16000;
    static constexpr int kNFft = 2048;
    static constexpr int kHopLength = 512;
    static constexpr int kNMels = 128;
    static constexpr int kFreqBins = kNFft / 2 + 1;  // 1025
    static constexpr float kFmin = 0.0f;
    static constexpr float kFmax = 8000.0f;

    MelSpectrogram();
    ~MelSpectrogram();

    // Non-copyable, movable
    MelSpectrogram(const MelSpectrogram&) = delete;
    MelSpectrogram& operator=(const MelSpectrogram&) = delete;
    MelSpectrogram(MelSpectrogram&&) noexcept;
    MelSpectrogram& operator=(MelSpectrogram&&) noexcept;

    /**
     * Compute mel spectrogram from mono 16kHz float32 PCM.
     *
     * @param pcm Input PCM samples (mono, 16kHz, float32)
     * @param num_samples Number of PCM samples
     * @param out_mel Output mel spectrogram (n_mels x n_frames, row-major)
     * @param out_n_frames Number of output time frames
     * @return true on success
     */
    bool Compute(const float* pcm, int num_samples,
                 std::vector<float>& out_mel, int& out_n_frames) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Precomputed data
    std::vector<float> hann_window_;      // Hann window [kNFft]
    std::vector<float> mel_filterbank_;   // Mel filterbank [kNMels x kFreqBins]

    void InitWindow();
    void InitMelFilterbank();

    // Helpers
    static float HzToMel(float hz);
    static float MelToHz(float mel);
};

}  // namespace smartdj
}  // namespace xune
