/**
 * @file xune_embedding.h
 * @brief SmartDJ audio embedding API
 *
 * Two-phase pipeline for computing audio embeddings:
 *   1. xune_embedding_compute_mel() — mel spectrogram + chunking (thread-safe: stack-local FFT buffers)
 *   2. xune_embedding_infer_into() — batched model inference, zero-copy into caller buffer
 *
 * Cross-track batching: compute mel for multiple tracks concurrently, then
 * concatenate their mel data and call xune_embedding_infer_into() once with the
 * combined total_chunks for a single batched inference call.
 *
 * The C# side handles:
 *   - Audio decoding via FFmpeg (to mono 16kHz float32)
 *   - Batch coordination (channel-based micro-batching)
 *   - Marshalling results back to managed memory
 *   - Storing embeddings as float16 in the database
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "xune_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error Codes
 * ============================================================================ */

typedef enum {
    XUNE_EMBEDDING_OK = 0,
    XUNE_EMBEDDING_ERROR_INVALID_ARGS = -1,
    XUNE_EMBEDDING_ERROR_MODEL_LOAD = -2,
    XUNE_EMBEDDING_ERROR_INFERENCE = -3,
    XUNE_EMBEDDING_ERROR_MEL = -4,
    XUNE_EMBEDDING_ERROR_NO_CHUNKS = -5,
    XUNE_EMBEDDING_ERROR_ALLOC = -6,
    XUNE_EMBEDDING_ERROR_NOT_AVAILABLE = -7,
} xune_embedding_error_t;

/* ============================================================================
 * Opaque Types
 * ============================================================================ */

typedef struct xune_embedding_session xune_embedding_session_t;
typedef struct xune_embedding_mel xune_embedding_mel_t;

/* ============================================================================
 * Session Lifecycle
 * ============================================================================ */

/**
 * @brief Create an embedding session.
 *
 * Loads the model and precomputes the mel filterbank.
 * Thread-safe: multiple sessions can exist simultaneously.
 *
 * @param model_path Path to the model file (UTF-8). .safetensors for MLX, .onnx for ORT.
 * @param cache_dir Optional cache directory (NULL to skip). Used by ORT backends.
 * @param out_session Receives the created session handle
 * @return XUNE_EMBEDDING_OK on success, error code on failure
 */
XUNE_AUDIO_API xune_embedding_error_t xune_embedding_create(
    const char* model_path,
    const char* cache_dir,
    xune_embedding_session_t** out_session);

/**
 * @brief Destroy an embedding session and free all resources.
 *
 * @param session Session to destroy (NULL is safe)
 */
XUNE_AUDIO_API void xune_embedding_destroy(
    xune_embedding_session_t* session);

/**
 * @brief Check if the embedding service is available.
 *
 * Returns true if the session was created successfully and inference is ready.
 *
 * @param session Session handle
 * @return true if available, false otherwise
 */
XUNE_AUDIO_API bool xune_embedding_is_available(
    xune_embedding_session_t* session);

/* ============================================================================
 * Phase 1: Mel Spectrogram Computation
 * ============================================================================ */

/**
 * @brief Compute mel spectrogram and chunk it for inference.
 *
 * Thread-safe: all FFT buffers are stack-local per call; the shared FFT
 * plan/setup is read-only after construction. Multiple concurrent calls
 * on the same session are safe.
 *
 * Each chunk is (128 mel bins, 96 time frames) representing ~3.125s of audio.
 * Output data is ready for direct use with xune_embedding_infer_into().
 *
 * @param session Active embedding session
 * @param pcm_mono_16k Mono 16kHz float32 PCM audio samples
 * @param num_samples Number of PCM samples
 * @param out_mel Receives the mel result handle (caller must free with xune_embedding_free_mel)
 * @return XUNE_EMBEDDING_OK on success, error code on failure
 */
XUNE_AUDIO_API xune_embedding_error_t xune_embedding_compute_mel(
    xune_embedding_session_t* session,
    const float* pcm_mono_16k,
    int num_samples,
    xune_embedding_mel_t** out_mel);

/**
 * @brief Get the number of chunks in a mel result.
 *
 * @param mel Mel result handle
 * @return Number of chunks, or 0 if mel is invalid
 */
XUNE_AUDIO_API int xune_embedding_mel_chunk_count(
    const xune_embedding_mel_t* mel);

/**
 * @brief Get pointer to the chunked mel data.
 *
 * Layout: float[chunk_count * 128 * 96], ready for ONNX input.
 * Pointer is valid until xune_embedding_free_mel() is called.
 *
 * @param mel Mel result handle
 * @return Pointer to float array, or NULL if mel is invalid
 */
XUNE_AUDIO_API const float* xune_embedding_mel_data(
    const xune_embedding_mel_t* mel);

/**
 * @brief Free a mel result.
 *
 * @param mel Mel result to free (NULL is safe)
 */
XUNE_AUDIO_API void xune_embedding_free_mel(
    xune_embedding_mel_t* mel);

/* ============================================================================
 * Phase 2: Batched Inference (zero-copy into caller buffer)
 * ============================================================================ */

/**
 * @brief Run batched inference, writing embeddings directly into a caller-owned buffer.
 *
 * Eliminates intermediate allocation — the inference engine copies directly from
 * its output tensor into the caller's buffer.
 *
 * Mel chunks from multiple tracks can be concatenated for cross-track batching.
 * The caller is responsible for tracking per-track chunk counts to split the
 * embeddings back to individual tracks.
 *
 * @param session Active embedding session
 * @param mel_data Concatenated mel chunk data, float[total_chunks * 128 * 96]
 * @param total_chunks Total number of chunks across all tracks
 * @param out_embeddings Caller-owned buffer, must hold total_chunks * 768 floats
 * @param out_buffer_floats Size of out_embeddings in floats (for bounds validation)
 * @param out_dimensions Receives the embedding dimensions (always 768)
 * @return XUNE_EMBEDDING_OK on success, error code on failure
 */
XUNE_AUDIO_API xune_embedding_error_t xune_embedding_infer_into(
    xune_embedding_session_t* session,
    const float* mel_data,
    int total_chunks,
    float* out_embeddings,
    int out_buffer_floats,
    int* out_dimensions);

#ifdef __cplusplus
}
#endif
