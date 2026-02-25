/**
 * @file fftw_plan_mutex.h
 * @brief Process-wide mutex for FFTW planner thread safety.
 *
 * FFTW planner routines (fftwf_plan_*, fftwf_destroy_plan) share global
 * wisdom/trig state and are NOT thread-safe. Execution routines
 * (fftwf_execute_*) with distinct per-call buffers ARE safe.
 *
 * This mutex must guard all plan creation and destruction across both
 * SmartDJ (mel_spectrogram.cpp) and BeatTracking (beat_mel_spectrogram.cpp).
 *
 * Only compiled on non-Apple platforms (Apple uses vDSP/Accelerate).
 */

#pragma once

#if !defined(__APPLE__)

#include <mutex>

namespace xune {

inline std::mutex& FftwPlanMutex() {
    static std::mutex instance;
    return instance;
}

}  // namespace xune

#endif
