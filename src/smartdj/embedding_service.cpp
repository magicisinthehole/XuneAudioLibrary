/**
 * @file embedding_service.cpp
 * @brief C API implementation for SmartDJ audio embedding computation.
 *
 * Two-phase pipeline:
 *   Phase 1 (compute_mel): PCM -> mel spectrogram -> chunk into (128, 96) slices
 *   Phase 2 (infer): batched ONNX inference on mel chunks -> 768-dim embeddings
 *
 * Phase 1 is thread-safe (MelSpectrogram::Compute is const) and can run
 * concurrently for multiple tracks. Phase 2 accepts concatenated mel data
 * from multiple tracks for cross-track batch inference.
 */

#include "xune_audio/xune_embedding.h"
#include "mel_spectrogram.h"
#include "model_inference.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#ifdef XUNE_USE_MLX
#include <mlx/memory.h>
namespace mx = mlx::core;
#endif

// ============================================================================
// Internal Types
// ============================================================================

struct xune_embedding_session {
    xune::smartdj::MelSpectrogram mel;
    xune::smartdj::ModelInference model;
    std::atomic<bool> cancelled{false};
    bool available = false;
};

struct xune_embedding_mel {
    std::vector<float> chunked_data;  // float[chunk_count * kNMels * kNFramesPerChunk]
    int chunk_count = 0;
};

// ============================================================================
// Constants matching the Myna model training configuration
// ============================================================================

// N_SAMPLES = 50000 at 16kHz -> 96 mel frames per chunk (floored to patch_size=16 multiple)
static constexpr int kNFramesPerChunk = 96;

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

xune_embedding_error_t xune_embedding_create(const char* model_path,
                                              const char* cache_dir,
                                              xune_embedding_session_t** out_session) {
    if (!model_path || !out_session) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    try {
        auto session = std::make_unique<xune_embedding_session>();

        // MelSpectrogram initializes filterbank in constructor (always succeeds)
        std::string cache_str = cache_dir ? cache_dir : "";
        if (!session->model.LoadModel(model_path, cache_str)) {
            fprintf(stderr, "[xune_embedding] Failed to load model: %s\n", model_path);
            *out_session = nullptr;
            return XUNE_EMBEDDING_ERROR_MODEL_LOAD;
        }

        session->model.SetCancelFlag(&session->cancelled);
        session->available = true;
        *out_session = session.release();
        return XUNE_EMBEDDING_OK;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xune_embedding] Session creation failed: %s\n", e.what());
        *out_session = nullptr;
        return XUNE_EMBEDDING_ERROR_MODEL_LOAD;
    } catch (...) {
        fprintf(stderr, "[xune_embedding] Session creation failed (unknown exception)\n");
        *out_session = nullptr;
        return XUNE_EMBEDDING_ERROR_MODEL_LOAD;
    }
}

void xune_embedding_cancel(xune_embedding_session_t* session) {
    if (!session) return;
    session->cancelled.store(true, std::memory_order_relaxed);
}

void xune_embedding_destroy(xune_embedding_session_t* session) {
    if (!session) return;
    delete session;
#ifdef XUNE_USE_MLX
    // Release Metal buffer cache — without this, freed weight tensors and
    // activation buffers remain in MLX's pool indefinitely (~300MB).
    mx::clear_cache();
#endif
}

bool xune_embedding_is_available(xune_embedding_session_t* session) {
    return session && session->available;
}

const char* xune_embedding_model_extension() {
#ifdef XUNE_USE_MLX
    return ".safetensors";
#else
    return ".onnx";
#endif
}

const char* xune_embedding_execution_provider(xune_embedding_session_t* session) {
    if (!session) return "Unknown";
    return session->model.GetExecutionProvider();
}

// ============================================================================
// Phase 1: Mel Spectrogram Computation
// ============================================================================

xune_embedding_error_t xune_embedding_compute_mel(xune_embedding_session_t* session,
                                                   const float* pcm_mono_16k,
                                                   int num_samples,
                                                   xune_embedding_mel_t** out_mel) {
    if (!session || !pcm_mono_16k || num_samples <= 0 || !out_mel) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_EMBEDDING_ERROR_NOT_AVAILABLE;
    }

    if (session->cancelled.load(std::memory_order_relaxed)) {
        return XUNE_EMBEDDING_ERROR_MEL;
    }

    // Thread-local scratch buffers survive across tracks on the same thread,
    // eliminating ~15MB alloc/free churn per track that causes native heap
    // fragmentation on macOS (magazine allocator retains freed pages).
    thread_local xune::smartdj::MelSpectrogram::ScratchBuffer scratch;

    // Step 1: Compute mel spectrogram
    std::vector<float> mel_data;
    int n_frames = 0;

    if (!session->mel.Compute(pcm_mono_16k, num_samples, mel_data, n_frames, scratch,
                               &session->cancelled)) {
        return XUNE_EMBEDDING_ERROR_MEL;
    }

    // Step 2: Chunk into slices of kNFramesPerChunk frames
    int num_chunks = n_frames / kNFramesPerChunk;
    if (num_chunks <= 0) {
        return XUNE_EMBEDDING_ERROR_NO_CHUNKS;
    }

    // Step 3: Rearrange into chunked layout for ONNX input
    // Input layout: (num_chunks, 1, n_mels, kNFramesPerChunk)
    // mel_data is currently [n_mels x n_frames], row-major (mel-first)
    const int n_mels = xune::smartdj::MelSpectrogram::kNMels;
    std::vector<float> chunked(
        static_cast<size_t>(num_chunks) * n_mels * kNFramesPerChunk);

    for (int c = 0; c < num_chunks; c++) {
        int frame_start = c * kNFramesPerChunk;
        float* chunk_ptr = chunked.data() +
                           static_cast<size_t>(c) * n_mels * kNFramesPerChunk;

        // Copy mel band by mel band (mel_data is [n_mels x n_frames] row-major)
        for (int m = 0; m < n_mels; m++) {
            const float* mel_row = mel_data.data() + m * n_frames + frame_start;
            float* out_row = chunk_ptr + m * kNFramesPerChunk;
            std::memcpy(out_row, mel_row, kNFramesPerChunk * sizeof(float));
        }
    }

    // Package mel result
    auto result = std::make_unique<xune_embedding_mel>();
    result->chunked_data = std::move(chunked);
    result->chunk_count = num_chunks;

    *out_mel = result.release();
    return XUNE_EMBEDDING_OK;
}

int xune_embedding_mel_chunk_count(const xune_embedding_mel_t* mel) {
    return mel ? mel->chunk_count : 0;
}

const float* xune_embedding_mel_data(const xune_embedding_mel_t* mel) {
    return mel ? mel->chunked_data.data() : nullptr;
}

void xune_embedding_free_mel(xune_embedding_mel_t* mel) {
    delete mel;
}

// ============================================================================
// Phase 2: Batched Inference (zero-copy into caller buffer)
// ============================================================================

xune_embedding_error_t xune_embedding_infer_into(xune_embedding_session_t* session,
                                                  const float* mel_data,
                                                  int total_chunks,
                                                  float* out_embeddings,
                                                  int out_buffer_floats,
                                                  int* out_dimensions) {
    if (!session || !mel_data || total_chunks <= 0 ||
        !out_embeddings || !out_dimensions) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_EMBEDDING_ERROR_NOT_AVAILABLE;
    }

    const int dims = xune::smartdj::ModelInference::kEmbeddingDim;
    int required = total_chunks * dims;
    if (out_buffer_floats < required) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    const int n_mels = xune::smartdj::MelSpectrogram::kNMels;

    if (session->cancelled.load(std::memory_order_relaxed)) {
        return XUNE_EMBEDDING_ERROR_INFERENCE;
    }

    if (!session->model.RunInferenceInto(mel_data, total_chunks,
                                          n_mels, kNFramesPerChunk,
                                          out_embeddings, out_buffer_floats)) {
        return XUNE_EMBEDDING_ERROR_INFERENCE;
    }

    *out_dimensions = dims;
    return XUNE_EMBEDDING_OK;
}

}  // extern "C"
