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
#include "onnx_inference.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// ============================================================================
// Internal Types
// ============================================================================

struct xune_embedding_session {
    xune::smartdj::MelSpectrogram mel;
    xune::smartdj::OnnxInference onnx;
    bool available = false;
};

struct xune_embedding_mel {
    std::vector<float> chunked_data;  // float[chunk_count * kNMels * kNFramesPerChunk]
    int chunk_count = 0;
};

struct xune_embedding_result {
    std::vector<float> data;
    int chunk_count = 0;
    int dimensions = 0;
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
                                              xune_embedding_session_t** out_session) {
    if (!model_path || !out_session) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    auto session = std::make_unique<xune_embedding_session>();

    // MelSpectrogram initializes filterbank in constructor (always succeeds)
    // Load ONNX model
    if (!session->onnx.LoadModel(model_path)) {
        fprintf(stderr, "[xune_embedding] Failed to load ONNX model: %s\n", model_path);
        *out_session = nullptr;
        return XUNE_EMBEDDING_ERROR_MODEL_LOAD;
    }

    session->available = true;
    *out_session = session.release();
    return XUNE_EMBEDDING_OK;
}

void xune_embedding_destroy(xune_embedding_session_t* session) {
    delete session;
}

bool xune_embedding_is_available(xune_embedding_session_t* session) {
    return session && session->available;
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

    // Step 1: Compute mel spectrogram
    std::vector<float> mel_data;
    int n_frames = 0;

    if (!session->mel.Compute(pcm_mono_16k, num_samples, mel_data, n_frames)) {
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
// Phase 2: Batched Inference
// ============================================================================

xune_embedding_error_t xune_embedding_infer(xune_embedding_session_t* session,
                                             const float* mel_data,
                                             int total_chunks,
                                             xune_embedding_result_t** out_result) {
    if (!session || !mel_data || total_chunks <= 0 || !out_result) {
        return XUNE_EMBEDDING_ERROR_INVALID_ARGS;
    }

    if (!session->available) {
        return XUNE_EMBEDDING_ERROR_NOT_AVAILABLE;
    }

    const int n_mels = xune::smartdj::MelSpectrogram::kNMels;

    std::vector<float> embeddings;
    if (!session->onnx.RunInference(mel_data, total_chunks,
                                     n_mels, kNFramesPerChunk, embeddings)) {
        return XUNE_EMBEDDING_ERROR_INFERENCE;
    }

    auto result = std::make_unique<xune_embedding_result>();
    result->data = std::move(embeddings);
    result->chunk_count = total_chunks;
    result->dimensions = xune::smartdj::OnnxInference::kEmbeddingDim;

    *out_result = result.release();
    return XUNE_EMBEDDING_OK;
}

// ============================================================================
// Result Access
// ============================================================================

const float* xune_embedding_result_data(const xune_embedding_result_t* result) {
    return result ? result->data.data() : nullptr;
}

int xune_embedding_result_chunk_count(const xune_embedding_result_t* result) {
    return result ? result->chunk_count : 0;
}

int xune_embedding_result_dimensions(const xune_embedding_result_t* result) {
    return result ? result->dimensions : 0;
}

void xune_embedding_free_result(xune_embedding_result_t* result) {
    delete result;
}

}  // extern "C"
