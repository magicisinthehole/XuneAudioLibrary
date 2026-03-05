/**
 * @file beat_service.cpp
 * @brief C API implementation for beat tracking.
 *
 * Two-phase pipeline matching the embedding API pattern:
 *   Phase 1: compute_mel — thread-safe, concurrent mel computation
 *   Phase 2: analyze_mel — inference + post-processing (caller serializes)
 *
 * One-shot xune_beat_analyze is kept for simple callers (tests, CLI).
 */

#include "xune_audio/xune_beat.h"
#include "beat_mel_spectrogram.h"
#include "beat_inference.h"
#include "beat_postprocess.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#ifdef XUNE_USE_MLX
#include <mlx/memory.h>
namespace mx = mlx::core;
#endif

struct xune_beat_session {
    xune::beattracking::BeatMelSpectrogram mel;
    xune::beattracking::BeatInference model;
    bool available = false;
};

struct xune_beat_mel {
    std::vector<float> data;
    int n_frames = 0;
};

// Helper: copy BeatResult to caller-owned malloc'd arrays
static xune_beat_error_t CopyResultToOutput(
    const xune::beattracking::BeatResult& result,
    float** out_beats, int* out_beat_count,
    float** out_downbeats, int* out_downbeat_count) {

    *out_beat_count = static_cast<int>(result.beats.size());
    if (*out_beat_count > 0) {
        *out_beats = static_cast<float*>(malloc(*out_beat_count * sizeof(float)));
        if (!*out_beats) return XUNE_BEAT_ERROR_ALLOC;
        std::memcpy(*out_beats, result.beats.data(), *out_beat_count * sizeof(float));
    } else {
        *out_beats = nullptr;
    }

    *out_downbeat_count = static_cast<int>(result.downbeats.size());
    if (*out_downbeat_count > 0) {
        *out_downbeats = static_cast<float*>(malloc(*out_downbeat_count * sizeof(float)));
        if (!*out_downbeats) {
            free(*out_beats);
            *out_beats = nullptr;
            return XUNE_BEAT_ERROR_ALLOC;
        }
        std::memcpy(*out_downbeats, result.downbeats.data(), *out_downbeat_count * sizeof(float));
    } else {
        *out_downbeats = nullptr;
    }

    return XUNE_BEAT_OK;
}

extern "C" {

// ============================================================================
// Session Lifecycle
// ============================================================================

xune_beat_error_t xune_beat_session_create(const char* model_path,
                                            const char* cache_dir,
                                            xune_beat_session_t** out_session) {
    if (!model_path || !out_session) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    auto session = std::make_unique<xune_beat_session>();

    std::string cache_str = cache_dir ? cache_dir : "";
    if (!session->model.LoadModel(model_path, cache_str)) {
        fprintf(stderr, "[xune_beat] Failed to load model: %s\n", model_path);
        *out_session = nullptr;
        return XUNE_BEAT_ERROR_MODEL_LOAD;
    }

    session->available = true;
    *out_session = session.release();
    return XUNE_BEAT_OK;
}

void xune_beat_session_destroy(xune_beat_session_t* session) {
    if (!session) return;
    delete session;
#ifdef XUNE_USE_MLX
    mx::clear_cache();
#endif
}

bool xune_beat_is_available(xune_beat_session_t* session) {
    return session && session->available;
}

const char* xune_beat_model_extension() {
#ifdef XUNE_USE_MLX
    return ".safetensors";
#else
    return ".onnx";
#endif
}

const char* xune_beat_execution_provider(xune_beat_session_t* session) {
    if (!session) return "Unknown";
    return session->model.GetExecutionProvider();
}

// ============================================================================
// Phase 1: Mel Spectrogram (thread-safe, concurrent)
// ============================================================================

xune_beat_error_t xune_beat_compute_mel(xune_beat_session_t* session,
                                         const float* pcm_mono_22k, int num_samples,
                                         xune_beat_mel_t** out_mel) {
    if (!session || !pcm_mono_22k || num_samples <= 0 || !out_mel) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_BEAT_ERROR_NOT_AVAILABLE;
    }

    // thread_local scratch buffers — same pattern as embedding mel computation
    thread_local xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;

    auto mel = std::make_unique<xune_beat_mel>();

    if (!session->mel.Compute(pcm_mono_22k, num_samples, mel->data, mel->n_frames, scratch)) {
        *out_mel = nullptr;
        return XUNE_BEAT_ERROR_MEL;
    }

    *out_mel = mel.release();
    return XUNE_BEAT_OK;
}

int xune_beat_mel_n_frames(xune_beat_mel_t* mel) {
    return mel ? mel->n_frames : 0;
}

float* xune_beat_mel_data(xune_beat_mel_t* mel) {
    return mel ? mel->data.data() : nullptr;
}

void xune_beat_free_mel(xune_beat_mel_t* mel) {
    delete mel;
}

// ============================================================================
// Phase 2: Analysis on Pre-computed Mel
// ============================================================================

xune_beat_error_t xune_beat_analyze_mel(xune_beat_session_t* session,
                                         xune_beat_mel_t* mel,
                                         float** out_beats, int* out_beat_count,
                                         float** out_downbeats, int* out_downbeat_count) {
    if (!session || !mel || !out_beats || !out_beat_count ||
        !out_downbeats || !out_downbeat_count) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_BEAT_ERROR_NOT_AVAILABLE;
    }

    if (mel->n_frames <= 0 || mel->data.empty()) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    xune::beattracking::BeatResult result;
    if (!xune::beattracking::BeatPostprocessor::Process(
            mel->data.data(), mel->n_frames, session->model, result)) {
        return XUNE_BEAT_ERROR_INFERENCE;
    }

    return CopyResultToOutput(result, out_beats, out_beat_count,
                              out_downbeats, out_downbeat_count);
}

// ============================================================================
// Batch Inference (caller packs mel, single GPU dispatch)
// ============================================================================

struct xune_beat_batch_result {
    std::vector<xune::beattracking::BeatResult> results;
};

xune_beat_error_t xune_beat_infer_batch(xune_beat_session_t* session,
                                         const float* packed_mel,
                                         const int* frame_offsets,
                                         int total_chunks, int chunk_frames,
                                         xune_beat_batch_result_t** out_result) {
    if (!session || !packed_mel || !frame_offsets ||
        total_chunks <= 0 || chunk_frames <= 0 || !out_result) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_BEAT_ERROR_NOT_AVAILABLE;
    }

    // Single batched inference call
    std::vector<float> beat_logits, db_logits;
    if (!session->model.RunInference(packed_mel, total_chunks, chunk_frames,
                                      beat_logits, db_logits)) {
        return XUNE_BEAT_ERROR_INFERENCE;
    }

    // Per-chunk post-processing (peak pick + snap + time offset)
    auto result = std::make_unique<xune_beat_batch_result>();
    xune::beattracking::BeatPostprocessor::PostprocessChunks(
        beat_logits.data(), db_logits.data(),
        total_chunks, chunk_frames, frame_offsets,
        result->results);

    *out_result = result.release();
    return XUNE_BEAT_OK;
}

int xune_beat_batch_count(xune_beat_batch_result_t* result) {
    return result ? static_cast<int>(result->results.size()) : 0;
}

float* xune_beat_batch_beats(xune_beat_batch_result_t* result, int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->results.size())) return nullptr;
    auto& beats = result->results[index].beats;
    return beats.empty() ? nullptr : beats.data();
}

int xune_beat_batch_beat_count(xune_beat_batch_result_t* result, int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->results.size())) return 0;
    return static_cast<int>(result->results[index].beats.size());
}

float* xune_beat_batch_downbeats(xune_beat_batch_result_t* result, int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->results.size())) return nullptr;
    auto& downbeats = result->results[index].downbeats;
    return downbeats.empty() ? nullptr : downbeats.data();
}

int xune_beat_batch_downbeat_count(xune_beat_batch_result_t* result, int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->results.size())) return 0;
    return static_cast<int>(result->results[index].downbeats.size());
}

void xune_beat_free_batch_result(xune_beat_batch_result_t* result) {
    delete result;
}

// ============================================================================
// One-shot Analysis (convenience)
// ============================================================================

xune_beat_error_t xune_beat_analyze(xune_beat_session_t* session,
                                     const float* pcm_mono_22k, int num_samples,
                                     float** out_beats, int* out_beat_count,
                                     float** out_downbeats, int* out_downbeat_count) {
    if (!session || !pcm_mono_22k || num_samples <= 0 ||
        !out_beats || !out_beat_count || !out_downbeats || !out_downbeat_count) {
        return XUNE_BEAT_ERROR_INVALID_ARGS;
    }

    // Phase 1: Compute mel
    xune_beat_mel_t* mel = nullptr;
    xune_beat_error_t err = xune_beat_compute_mel(session, pcm_mono_22k, num_samples, &mel);
    if (err != XUNE_BEAT_OK) {
        return err;
    }

    // Phase 2: Analyze mel
    err = xune_beat_analyze_mel(session, mel, out_beats, out_beat_count,
                                out_downbeats, out_downbeat_count);

    xune_beat_free_mel(mel);
    return err;
}

void xune_beat_free(float* data) {
    free(data);
}

}  // extern "C"
