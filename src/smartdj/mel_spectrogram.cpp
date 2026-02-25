/**
 * @file mel_spectrogram.cpp
 * @brief Mel spectrogram implementation with platform FFT backends.
 *
 * macOS: vDSP (Accelerate framework) — zero external dependencies
 * Linux: FFTW3 — already a dependency via Chromaprint
 * Windows: FFTW3 (or could use MKL in the future)
 */

#include "mel_spectrogram.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

// Platform-specific FFT
#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <fftw3.h>
    #include "../fftw_plan_mutex.h"
#endif

namespace xune {
namespace smartdj {

// ============================================================================
// Platform FFT Implementation (PIMPL)
// ============================================================================

struct MelSpectrogram::Impl {
#if defined(__APPLE__)
    // vDSP FFT setup (read-only after construction — thread-safe)
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

    // Compute |FFT(windowed_frame)|^2 -> power spectrum [kFreqBins]
    // Thread-safe: all FFT buffers are stack-local per call.
    void ComputePowerSpectrum(const float* windowed_frame, float* power_spectrum) const {
        int log2n = static_cast<int>(std::log2(kNFft));

        // Stack-local buffers — each concurrent call uses its own storage
        float real_buf[kNFft / 2];
        float imag_buf[kNFft / 2];

        // Pack real data into split complex format
        DSPSplitComplex split;
        split.realp = real_buf;
        split.imagp = imag_buf;
        vDSP_ctoz(reinterpret_cast<const DSPComplex*>(windowed_frame), 2,
                   &split, 1, kNFft / 2);

        // Forward FFT
        vDSP_fft_zrip(fft_setup, &split, 1, log2n, FFT_FORWARD);

        // vDSP returns 2*DFT, so |vDSP|^2 = 4*|DFT|^2. Divide by 4 to get |DFT|^2.
        // No 1/N normalization — PyTorch/nnAudio use the unnormalized DFT.

        // vDSP packs DC and Nyquist into split.realp[0] and split.imagp[0]
        // DC component
        power_spectrum[0] = (split.realp[0] * split.realp[0]) / 4.0f;
        // Nyquist component
        power_spectrum[kNFft / 2] = (split.imagp[0] * split.imagp[0]) / 4.0f;

        // Remaining bins
        for (int i = 1; i < kNFft / 2; i++) {
            power_spectrum[i] = (split.realp[i] * split.realp[i] +
                                 split.imagp[i] * split.imagp[i]) / 4.0f;
        }
    }

#else
    // FFTW3F single-precision plan (read-only after construction — thread-safe with new-array execute)
    fftwf_plan plan = nullptr;

    Impl() {
        // Create plan with temporary buffers. FFTW only uses these during planning;
        // fftwf_execute_dft_r2c will use caller-supplied arrays at execution time.
        // Planner is NOT thread-safe — must hold process-wide mutex.
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

    // Thread-safe: uses stack-local buffers + fftwf_execute_dft_r2c (new-array variant).
    // Per FFTW docs: "always safe to call in a multi-threaded program as long as
    // different threads use different arrays."
    void ComputePowerSpectrum(const float* windowed_frame, float* power_spectrum) const {
        // Stack-local buffers — each concurrent call uses its own storage
        float fft_input[kNFft];
        fftwf_complex fft_output[kFreqBins];

        // Direct copy — no float→double conversion needed with FFTW3F
        std::memcpy(fft_input, windowed_frame, kNFft * sizeof(float));

        // New-array execute variant: thread-safe with distinct per-call buffers
        fftwf_execute_dft_r2c(plan, fft_input, fft_output);

        // FFTW3F returns the unnormalized DFT — no scaling needed.
        // PyTorch/nnAudio also use unnormalized DFT for power spectrogram.
        for (int i = 0; i < kFreqBins; i++) {
            float re = fft_output[i][0];
            float im = fft_output[i][1];
            power_spectrum[i] = re * re + im * im;
        }
    }
#endif

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

// ============================================================================
// MelSpectrogram Implementation
// ============================================================================

MelSpectrogram::MelSpectrogram()
    : impl_(std::make_unique<Impl>()) {
    InitWindow();
    InitMelFilterbank();
}

MelSpectrogram::~MelSpectrogram() = default;

MelSpectrogram::MelSpectrogram(MelSpectrogram&&) noexcept = default;
MelSpectrogram& MelSpectrogram::operator=(MelSpectrogram&&) noexcept = default;

void MelSpectrogram::InitWindow() {
    hann_window_.resize(kNFft);
    for (int i = 0; i < kNFft; i++) {
        // Periodic Hann window (matching PyTorch/nnAudio convention)
        constexpr float kPi = 3.14159265358979323846f;
        hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / kNFft));
    }
}

float MelSpectrogram::HzToMel(float hz) {
    // Slaney mel scale (NOT HTK)
    // Linear below 1000 Hz, logarithmic above
    constexpr float f_sp = 200.0f / 3.0f;  // ~66.67 Hz
    float mel = hz / f_sp;

    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (hz >= min_log_hz) {
        mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return mel;
}

float MelSpectrogram::MelToHz(float mel) {
    // Inverse of Slaney mel scale
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

void MelSpectrogram::InitMelFilterbank() {
    // Slaney-style area-normalized mel filterbank
    // n_mels+2 points to create n_mels triangular filters
    const int n_points = kNMels + 2;

    // Compute mel points evenly spaced in mel scale
    float mel_min = HzToMel(kFmin);
    float mel_max = HzToMel(kFmax);
    std::vector<float> mel_points(n_points);
    for (int i = 0; i < n_points; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_points - 1);
    }

    // Convert back to Hz
    std::vector<float> hz_points(n_points);
    for (int i = 0; i < n_points; i++) {
        hz_points[i] = MelToHz(mel_points[i]);
    }

    // Convert to FFT bin indices (fractional)
    std::vector<float> bin_points(n_points);
    for (int i = 0; i < n_points; i++) {
        bin_points[i] = hz_points[i] * kNFft / static_cast<float>(kSampleRate);
    }

    // Build filterbank [kNMels x kFreqBins]
    mel_filterbank_.resize(kNMels * kFreqBins, 0.0f);

    for (int m = 0; m < kNMels; m++) {
        float left = bin_points[m];
        float center = bin_points[m + 1];
        float right = bin_points[m + 2];

        // Area normalization (Slaney)
        float enorm = 2.0f / (hz_points[m + 2] - hz_points[m]);

        for (int k = 0; k < kFreqBins; k++) {
            float fk = static_cast<float>(k);
            float weight = 0.0f;

            if (fk >= left && fk <= center && center > left) {
                weight = (fk - left) / (center - left);
            } else if (fk > center && fk <= right && right > center) {
                weight = (right - fk) / (right - center);
            }

            mel_filterbank_[m * kFreqBins + k] = weight * enorm;
        }
    }
}

bool MelSpectrogram::Compute(const float* pcm, int num_samples,
                              std::vector<float>& out_mel, int& out_n_frames) const {
    ScratchBuffer scratch;
    return Compute(pcm, num_samples, out_mel, out_n_frames, scratch);
}

bool MelSpectrogram::Compute(const float* pcm, int num_samples,
                              std::vector<float>& out_mel, int& out_n_frames,
                              ScratchBuffer& scratch) const {
    if (!pcm || num_samples <= 0) {
        return false;
    }

    // Center-pad with reflect padding (pad_length = n_fft / 2 = 1024)
    const int pad = kNFft / 2;
    const int padded_length = num_samples + 2 * pad;

    // Resize scratch buffers (vectors only grow, never shrink — after processing
    // the longest track, no further allocations occur for the rest of the run)
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

    // Allocate output: [kNMels x out_n_frames], row-major (mel-first)
    out_mel.resize(kNMels * out_n_frames, 0.0f);

    // Resize temporary buffers from scratch
    scratch.windowed_frame.resize(kNFft);
    scratch.power_spectrum.resize(kFreqBins);

    // Process each frame
    for (int frame = 0; frame < out_n_frames; frame++) {
        int start = frame * kHopLength;

        // Apply Hann window
        for (int i = 0; i < kNFft; i++) {
            scratch.windowed_frame[i] = scratch.padded[start + i] * hann_window_[i];
        }

        // Compute power spectrum |FFT|^2
        impl_->ComputePowerSpectrum(scratch.windowed_frame.data(), scratch.power_spectrum.data());

        // Apply mel filterbank: mel[m] = sum(filterbank[m,k] * power[k])
        for (int m = 0; m < kNMels; m++) {
            float sum = 0.0f;
            const float* filter_row = mel_filterbank_.data() + m * kFreqBins;

#if defined(__APPLE__)
            // Use vDSP for dot product
            vDSP_dotpr(filter_row, 1, scratch.power_spectrum.data(), 1, &sum, kFreqBins);
#else
            for (int k = 0; k < kFreqBins; k++) {
                sum += filter_row[k] * scratch.power_spectrum[k];
            }
#endif

            out_mel[m * out_n_frames + frame] = sum;
        }
    }

    return true;
}

}  // namespace smartdj
}  // namespace xune
