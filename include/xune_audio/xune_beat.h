/**
 * @file xune_beat.h
 * @brief Beat tracking API (Beat This! model)
 *
 * Two-phase pipeline matching the embedding API pattern:
 *   Phase 1: xune_beat_compute_mel — thread-safe, concurrent mel computation
 *   Phase 2: xune_beat_analyze_mel — inference + post-processing (serialize via caller)
 *
 * Usage (two-phase, recommended for batching):
 *   1. xune_beat_session_create(model_path, &session)
 *   2. xune_beat_compute_mel(session, pcm, n, &mel)       [concurrent]
 *   3. xune_beat_analyze_mel(session, mel, &beats, ...)    [serialized]
 *   4. xune_beat_free(beats); xune_beat_free(downbeats);
 *   5. xune_beat_free_mel(mel);
 *   6. xune_beat_session_destroy(session);
 *
 * Usage (one-shot, for simple callers):
 *   1. xune_beat_session_create(model_path, &session)
 *   2. xune_beat_analyze(session, pcm, n, &beats, &bc, &downbeats, &dc)
 *   3. xune_beat_free(beats); xune_beat_free(downbeats);
 *   4. xune_beat_session_destroy(session);
 */

#pragma once

#include <stdbool.h>
#include "xune_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error Codes
 * ============================================================================ */

typedef enum {
    XUNE_BEAT_OK = 0,
    XUNE_BEAT_ERROR_INVALID_ARGS  = -1,
    XUNE_BEAT_ERROR_MODEL_LOAD    = -2,
    XUNE_BEAT_ERROR_INFERENCE     = -3,
    XUNE_BEAT_ERROR_MEL           = -4,
    XUNE_BEAT_ERROR_ALLOC         = -5,
    XUNE_BEAT_ERROR_NOT_AVAILABLE = -6,
} xune_beat_error_t;

/* ============================================================================
 * Opaque Types
 * ============================================================================ */

typedef struct xune_beat_session xune_beat_session_t;
typedef struct xune_beat_mel xune_beat_mel_t;
typedef struct xune_beat_batch_result xune_beat_batch_result_t;

/* ============================================================================
 * Session Lifecycle
 * ============================================================================ */

/**
 * @brief Create a beat tracking session.
 *
 * Loads the Beat This! model and precomputes the mel filterbank.
 *
 * @param model_path Path to model file (.safetensors for MLX, .onnx for ORT)
 * @param out_session Receives the created session handle
 * @return XUNE_BEAT_OK on success
 */
XUNE_AUDIO_API xune_beat_error_t xune_beat_session_create(
    const char* model_path,
    xune_beat_session_t** out_session);

/**
 * @brief Destroy a beat tracking session.
 * @param session Session to destroy (NULL is safe)
 */
XUNE_AUDIO_API void xune_beat_session_destroy(
    xune_beat_session_t* session);

/**
 * @brief Check if the beat tracking service is available.
 * @param session Session handle
 * @return true if available
 */
XUNE_AUDIO_API bool xune_beat_is_available(
    xune_beat_session_t* session);

/// Returns the expected model file extension (".safetensors" for MLX, ".onnx" for ORT).
XUNE_AUDIO_API const char* xune_beat_model_extension(void);

/* ============================================================================
 * Phase 1: Mel Spectrogram (thread-safe, concurrent)
 * ============================================================================ */

/**
 * @brief Compute mel spectrogram from PCM audio.
 *
 * Thread-safe — uses thread_local scratch buffers. Multiple threads can
 * call this concurrently with the same session.
 *
 * @param session Active session
 * @param pcm_mono_22k Mono float32 PCM at 22050 Hz
 * @param num_samples Number of PCM samples
 * @param out_mel Receives the mel handle
 * @return XUNE_BEAT_OK on success
 */
XUNE_AUDIO_API xune_beat_error_t xune_beat_compute_mel(
    xune_beat_session_t* session,
    const float* pcm_mono_22k, int num_samples,
    xune_beat_mel_t** out_mel);

/**
 * @brief Get the number of mel frames.
 * @param mel Mel handle from xune_beat_compute_mel
 * @return Number of frames, or 0 if invalid
 */
XUNE_AUDIO_API int xune_beat_mel_n_frames(xune_beat_mel_t* mel);

/**
 * @brief Get pointer to mel data.
 *
 * Layout: [n_frames x 128] row-major (time-first).
 *
 * @param mel Mel handle from xune_beat_compute_mel
 * @return Pointer to float data, or NULL if invalid
 */
XUNE_AUDIO_API float* xune_beat_mel_data(xune_beat_mel_t* mel);

/**
 * @brief Free a mel handle from xune_beat_compute_mel.
 * @param mel Mel handle to free (NULL is safe)
 */
XUNE_AUDIO_API void xune_beat_free_mel(xune_beat_mel_t* mel);

/* ============================================================================
 * Phase 2: Analysis on Pre-computed Mel (serialize via caller)
 * ============================================================================ */

/**
 * @brief Analyze pre-computed mel spectrogram for beat positions.
 *
 * Runs chunked inference and post-processing on the mel data.
 * NOT thread-safe — caller must serialize access (e.g., via Channel worker).
 *
 * Caller must free output arrays with xune_beat_free().
 *
 * @param session Active session
 * @param mel Mel handle from xune_beat_compute_mel
 * @param out_beats Receives allocated array of beat timestamps (seconds)
 * @param out_beat_count Receives number of beats
 * @param out_downbeats Receives allocated array of downbeat timestamps (seconds)
 * @param out_downbeat_count Receives number of downbeats
 * @return XUNE_BEAT_OK on success
 */
XUNE_AUDIO_API xune_beat_error_t xune_beat_analyze_mel(
    xune_beat_session_t* session,
    xune_beat_mel_t* mel,
    float** out_beats, int* out_beat_count,
    float** out_downbeats, int* out_downbeat_count);

/* ============================================================================
 * One-shot Analysis (convenience, not for batching)
 * ============================================================================ */

/**
 * @brief Analyze audio for beat positions (one-shot).
 *
 * Computes mel + inference + post-processing in a single call.
 * For batching scenarios, use the two-phase API instead.
 *
 * Caller must free output arrays with xune_beat_free().
 *
 * @param session Active session
 * @param pcm_mono_22k Mono float32 PCM at 22050 Hz
 * @param num_samples Number of PCM samples
 * @param out_beats Receives allocated array of beat timestamps (seconds)
 * @param out_beat_count Receives number of beats
 * @param out_downbeats Receives allocated array of downbeat timestamps (seconds)
 * @param out_downbeat_count Receives number of downbeats
 * @return XUNE_BEAT_OK on success
 */
XUNE_AUDIO_API xune_beat_error_t xune_beat_analyze(
    xune_beat_session_t* session,
    const float* pcm_mono_22k, int num_samples,
    float** out_beats, int* out_beat_count,
    float** out_downbeats, int* out_downbeat_count);

/* ============================================================================
 * Batch Inference (caller packs mel, single GPU dispatch)
 * ============================================================================ */

/**
 * @brief Batch inference + per-chunk post-processing on pre-packed mel chunks.
 *
 * Caller extracts and packs mel chunks (e.g., edge-only: first/last 30s per
 * track) into a contiguous buffer. This function runs one batched inference
 * call and returns per-chunk beat/downbeat timestamps.
 *
 * Timestamps for chunk i are offset by frame_offsets[i] / 50.0 seconds
 * so they represent absolute positions within the source track.
 *
 * @param session Active session
 * @param packed_mel Contiguous (total_chunks, chunk_frames, 128) mel data
 * @param frame_offsets Array of total_chunks offsets (frame index of each
 *                      chunk's start within its source track)
 * @param total_chunks Number of chunks in the batch
 * @param chunk_frames Number of frames per chunk (all must be equal)
 * @param out_result Receives the batch result handle
 * @return XUNE_BEAT_OK on success
 */
XUNE_AUDIO_API xune_beat_error_t xune_beat_infer_batch(
    xune_beat_session_t* session,
    const float* packed_mel,
    const int* frame_offsets,
    int total_chunks, int chunk_frames,
    xune_beat_batch_result_t** out_result);

/**
 * @brief Get the number of tracks in a batch result.
 */
XUNE_AUDIO_API int xune_beat_batch_count(xune_beat_batch_result_t* result);

/**
 * @brief Get beat timestamps for a track in a batch result.
 * @param result Batch result handle
 * @param index Track index (0-based)
 * @return Pointer to float array (owned by result, do NOT free), or NULL
 */
XUNE_AUDIO_API float* xune_beat_batch_beats(xune_beat_batch_result_t* result, int index);

/**
 * @brief Get beat count for a track in a batch result.
 */
XUNE_AUDIO_API int xune_beat_batch_beat_count(xune_beat_batch_result_t* result, int index);

/**
 * @brief Get downbeat timestamps for a track in a batch result.
 */
XUNE_AUDIO_API float* xune_beat_batch_downbeats(xune_beat_batch_result_t* result, int index);

/**
 * @brief Get downbeat count for a track in a batch result.
 */
XUNE_AUDIO_API int xune_beat_batch_downbeat_count(xune_beat_batch_result_t* result, int index);

/**
 * @brief Free a batch result (frees all per-track data).
 * @param result Batch result to free (NULL is safe)
 */
XUNE_AUDIO_API void xune_beat_free_batch_result(xune_beat_batch_result_t* result);

/* ============================================================================
 * Free Helpers
 * ============================================================================ */

/**
 * @brief Free a beat/downbeat array returned by xune_beat_analyze[_mel].
 * @param data Array to free (NULL is safe)
 */
XUNE_AUDIO_API void xune_beat_free(float* data);

#ifdef __cplusplus
}
#endif
