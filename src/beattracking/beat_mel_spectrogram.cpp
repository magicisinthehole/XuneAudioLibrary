/**
 * @file beat_mel_spectrogram.cpp
 * @brief Beat This! mel spectrogram implementation with platform FFT backends.
 *
 * macOS: vDSP (Accelerate framework)
 * Linux: FFTW3
 *
 * Key differences from SmartDJ mel (mel_spectrogram.cpp):
 *   - 22050 Hz sample rate (vs 16000)
 *   - n_fft=1024 (vs 2048), hop=441 (vs 512)
 *   - Power=1 magnitude (vs power=2 squared)
 *   - frame_length normalization (÷ sqrt(n_fft) on STFT output, torch.stft normalized=True)
 *   - log1p(1000 × mel) transform (vs no transform)
 *   - f_min=30, f_max=11000 (vs 0, 8000)
 *   - Output layout: [n_frames x n_mels] time-first (vs [n_mels x n_frames])
 */

#include "beat_mel_spectrogram.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <fftw3.h>
    #include "../fftw_plan_mutex.h"
#endif

namespace xune {
namespace beattracking {

// ============================================================================
// Platform FFT Implementation (PIMPL)
// ============================================================================

struct BeatMelSpectrogram::Impl {
#if defined(__APPLE__)
    FFTSetup fft_setup = nullptr;

    Impl() {
        int log2n = static_cast<int>(std::log2(kNFft));
        fft_setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    }

    ~Impl() {
        if (fft_setup) {
            vDSP_destroy_fftsetup(fft_setup);
        }
    }

    // Compute |FFT(windowed_frame)| / sqrt(n_fft) -> magnitude spectrum [kFreqBins]
    // torchaudio normalized="frame_length" maps to torch.stft(normalized=True),
    // which divides by sqrt(n_fft), not n_fft.
    void ComputeMagSpectrum(const float* windowed_frame, float* mag_spectrum) const {
        int log2n = static_cast<int>(std::log2(kNFft));

        float real_buf[kNFft / 2];
        float imag_buf[kNFft / 2];

        DSPSplitComplex split;
        split.realp = real_buf;
        split.imagp = imag_buf;
        vDSP_ctoz(reinterpret_cast<const DSPComplex*>(windowed_frame), 2,
                   &split, 1, kNFft / 2);

        vDSP_fft_zrip(fft_setup, &split, 1, log2n, FFT_FORWARD);

        // vDSP returns 2*DFT. Magnitude = |vDSP| / 2.
        // With frame_length normalization: magnitude / sqrt(n_fft).
        // Combined: |vDSP| / (2 * sqrt(n_fft))
        static const float kNormFactor = 1.0f / (2.0f * std::sqrt(static_cast<float>(kNFft)));

        // DC component (packed in realp[0])
        mag_spectrum[0] = std::abs(split.realp[0]) * kNormFactor;
        // Nyquist component (packed in imagp[0])
        mag_spectrum[kNFft / 2] = std::abs(split.imagp[0]) * kNormFactor;

        // Remaining bins: |re + j*im| / (2 * sqrt(n_fft))
        for (int i = 1; i < kNFft / 2; i++) {
            float re = split.realp[i];
            float im = split.imagp[i];
            mag_spectrum[i] = std::sqrt(re * re + im * im) * kNormFactor;
        }
    }

#else
    fftwf_plan plan = nullptr;

    Impl() {
        std::lock_guard<std::mutex> lock(FftwPlanMutex());
        std::vector<float> tmp_input(kNFft);
        std::vector<fftwf_complex> tmp_output(kFreqBins);
        plan = fftwf_plan_dft_r2c_1d(kNFft, tmp_input.data(),
                                      tmp_output.data(), FFTW_MEASURE);
    }

    ~Impl() {
        if (plan) {
            std::lock_guard<std::mutex> lock(FftwPlanMutex());
            fftwf_destroy_plan(plan);
        }
    }

    // Compute |FFT(windowed_frame)| / sqrt(n_fft) -> magnitude spectrum [kFreqBins]
    // torchaudio normalized="frame_length" maps to torch.stft(normalized=True),
    // which divides by sqrt(n_fft), not n_fft.
    void ComputeMagSpectrum(const float* windowed_frame, float* mag_spectrum) const {
        float fft_input[kNFft];
        fftwf_complex fft_output[kFreqBins];

        // Direct copy — no float→double conversion needed with FFTW3F
        std::memcpy(fft_input, windowed_frame, kNFft * sizeof(float));

        fftwf_execute_dft_r2c(plan, fft_input, fft_output);

        // FFTW3F returns unnormalized DFT.
        // Magnitude = |DFT| / sqrt(n_fft) (frame_length normalization)
        static const float kNormFactor = 1.0f / std::sqrt(static_cast<float>(kNFft));
        for (int i = 0; i < kFreqBins; i++) {
            float re = fft_output[i][0];
            float im = fft_output[i][1];
            mag_spectrum[i] = std::sqrt(re * re + im * im) * kNormFactor;
        }
    }
#endif

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

// ============================================================================
// BeatMelSpectrogram Implementation
// ============================================================================

BeatMelSpectrogram::BeatMelSpectrogram()
    : impl_(std::make_unique<Impl>()) {
    InitWindow();
    InitMelFilterbank();
}

BeatMelSpectrogram::~BeatMelSpectrogram() = default;

BeatMelSpectrogram::BeatMelSpectrogram(BeatMelSpectrogram&&) noexcept = default;
BeatMelSpectrogram& BeatMelSpectrogram::operator=(BeatMelSpectrogram&&) noexcept = default;

void BeatMelSpectrogram::InitWindow() {
    hann_window_.resize(kNFft);
    for (int i = 0; i < kNFft; i++) {
        constexpr float kPi = 3.14159265358979323846f;
        hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / kNFft));
    }
}

float BeatMelSpectrogram::HzToMel(float hz) {
    // Slaney mel scale: linear below 1000 Hz, logarithmic above
    constexpr float f_sp = 200.0f / 3.0f;
    float mel = hz / f_sp;

    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (hz >= min_log_hz) {
        mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return mel;
}

float BeatMelSpectrogram::MelToHz(float mel) {
    constexpr float f_sp = 200.0f / 3.0f;
    float hz = mel * f_sp;

    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (mel >= min_log_mel) {
        hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return hz;
}

void BeatMelSpectrogram::InitMelFilterbank() {
    const int n_points = kNMels + 2;

    float mel_min = HzToMel(kFmin);
    float mel_max = HzToMel(kFmax);
    std::vector<float> mel_points(n_points);
    for (int i = 0; i < n_points; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_points - 1);
    }

    std::vector<float> hz_points(n_points);
    for (int i = 0; i < n_points; i++) {
        hz_points[i] = MelToHz(mel_points[i]);
    }

    std::vector<float> bin_points(n_points);
    for (int i = 0; i < n_points; i++) {
        bin_points[i] = hz_points[i] * kNFft / static_cast<float>(kSampleRate);
    }

    mel_filterbank_.resize(kNMels * kFreqBins, 0.0f);

    for (int m = 0; m < kNMels; m++) {
        float left = bin_points[m];
        float center = bin_points[m + 1];
        float right = bin_points[m + 2];

        // No area normalization — Beat This! uses mel_scale="slaney" (frequency scale)
        // but NOT norm="slaney" (area normalization). Triangular filters peak at 1.0.
        for (int k = 0; k < kFreqBins; k++) {
            float fk = static_cast<float>(k);
            float weight = 0.0f;

            if (fk >= left && fk <= center && center > left) {
                weight = (fk - left) / (center - left);
            } else if (fk > center && fk <= right && right > center) {
                weight = (right - fk) / (right - center);
            }

            mel_filterbank_[m * kFreqBins + k] = weight;
        }
    }
}

bool BeatMelSpectrogram::Compute(const float* pcm, int num_samples,
                                  std::vector<float>& out_mel, int& out_n_frames) const {
    ScratchBuffer scratch;
    return Compute(pcm, num_samples, out_mel, out_n_frames, scratch);
}

bool BeatMelSpectrogram::Compute(const float* pcm, int num_samples,
                                  std::vector<float>& out_mel, int& out_n_frames,
                                  ScratchBuffer& scratch,
                                  const std::atomic<bool>* cancel) const {
    if (!pcm || num_samples <= 0) {
        return false;
    }

    // Center-pad with reflect padding (pad_length = n_fft / 2 = 512)
    const int pad = kNFft / 2;
    const int padded_length = num_samples + 2 * pad;

    scratch.padded.resize(padded_length);

    // Left reflect pad
    for (int i = 0; i < pad; i++) {
        int src_idx = pad - i;
        if (src_idx >= num_samples) src_idx = num_samples - 1;
        scratch.padded[i] = pcm[src_idx];
    }

    // Copy original signal
    std::memcpy(scratch.padded.data() + pad, pcm, num_samples * sizeof(float));

    // Right reflect pad
    for (int i = 0; i < pad; i++) {
        int src_idx = num_samples - 2 - i;
        if (src_idx < 0) src_idx = 0;
        scratch.padded[pad + num_samples + i] = pcm[src_idx];
    }

    // Compute number of frames
    out_n_frames = (padded_length - kNFft) / kHopLength + 1;
    if (out_n_frames <= 0) {
        return false;
    }

    // Output: [n_frames x n_mels] row-major (time-first for Beat This! model input)
    out_mel.resize(static_cast<size_t>(out_n_frames) * kNMels, 0.0f);

    scratch.windowed_frame.resize(kNFft);
    scratch.mag_spectrum.resize(kFreqBins);

    for (int frame = 0; frame < out_n_frames; frame++) {
        if (cancel && cancel->load(std::memory_order_relaxed)) {
            return false;
        }

        int start = frame * kHopLength;

        // Apply Hann window
        for (int i = 0; i < kNFft; i++) {
            scratch.windowed_frame[i] = scratch.padded[start + i] * hann_window_[i];
        }

        // Compute magnitude spectrum with frame_length normalization
        impl_->ComputeMagSpectrum(scratch.windowed_frame.data(), scratch.mag_spectrum.data());

        // Apply mel filterbank + log1p transform
        // Output layout: out_mel[frame * kNMels + m]
        float* frame_out = out_mel.data() + static_cast<size_t>(frame) * kNMels;

        for (int m = 0; m < kNMels; m++) {
            float sum = 0.0f;
            const float* filter_row = mel_filterbank_.data() + m * kFreqBins;

#if defined(__APPLE__)
            vDSP_dotpr(filter_row, 1, scratch.mag_spectrum.data(), 1, &sum, kFreqBins);
#else
            for (int k = 0; k < kFreqBins; k++) {
                sum += filter_row[k] * scratch.mag_spectrum[k];
            }
#endif

            // log1p(1000 * mel)
            frame_out[m] = std::log1p(kLogMultiplier * sum);
        }
    }

    return true;
}

}  // namespace beattracking
}  // namespace xune
