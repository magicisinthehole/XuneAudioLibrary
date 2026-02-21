/**
 * @file generate_reference_embeddings.cpp
 * @brief Generate reference embeddings using the C++ pipeline.
 *
 * Reads test PCM audio, runs it through the C++ mel spectrogram + ONNX model,
 * and writes the resulting embeddings to a binary file. This ensures that test
 * reference data uses the same mel implementation as the tests themselves,
 * isolating only model-level differences (e.g., FP32 vs INT8 quantization).
 *
 * Usage:
 *   generate_reference_embeddings <model.onnx> <pcm.bin> <output_embeddings.bin>
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "xune_audio/xune_embedding.h"

static std::vector<float> LoadFloatBin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static bool WriteFloatBin(const std::string& path, const float* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return file.good();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.onnx> <pcm.bin> <output_embeddings.bin>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* pcm_path = argv[2];
    const char* output_path = argv[3];

    // Load PCM
    auto pcm = LoadFloatBin(pcm_path);
    if (pcm.empty()) {
        fprintf(stderr, "Error: failed to load PCM from %s\n", pcm_path);
        return 1;
    }
    fprintf(stderr, "Loaded %zu PCM samples from %s\n", pcm.size(), pcm_path);

    // Create session
    xune_embedding_session_t* session = nullptr;
    int err = xune_embedding_create(model_path, nullptr, &session);
    if (err != XUNE_EMBEDDING_OK) {
        fprintf(stderr, "Error: failed to create session (error %d)\n", err);
        return 1;
    }
    fprintf(stderr, "Loaded model: %s\n", model_path);

    // Phase 1: Compute mel spectrogram
    xune_embedding_mel_t* mel = nullptr;
    err = xune_embedding_compute_mel(session, pcm.data(), static_cast<int>(pcm.size()), &mel);
    if (err != XUNE_EMBEDDING_OK) {
        fprintf(stderr, "Error: mel computation failed (error %d)\n", err);
        xune_embedding_destroy(session);
        return 1;
    }

    int mel_chunks = xune_embedding_mel_chunk_count(mel);
    fprintf(stderr, "Computed %d mel chunk(s)\n", mel_chunks);

    // Phase 2: Batched inference
    xune_embedding_result_t* result = nullptr;
    err = xune_embedding_infer(session, xune_embedding_mel_data(mel), mel_chunks, &result);
    if (err != XUNE_EMBEDDING_OK) {
        fprintf(stderr, "Error: inference failed (error %d)\n", err);
        xune_embedding_free_mel(mel);
        xune_embedding_destroy(session);
        return 1;
    }

    int chunks = xune_embedding_result_chunk_count(result);
    int dims = xune_embedding_result_dimensions(result);
    const float* data = xune_embedding_result_data(result);

    fprintf(stderr, "Computed %d chunk(s) x %d dims\n", chunks, dims);

    // Write embeddings
    size_t total = static_cast<size_t>(chunks) * dims;
    if (!WriteFloatBin(output_path, data, total)) {
        fprintf(stderr, "Error: failed to write %s\n", output_path);
        xune_embedding_free_result(result);
        xune_embedding_free_mel(mel);
        xune_embedding_destroy(session);
        return 1;
    }

    fprintf(stderr, "Wrote reference embeddings: %s (%zu floats)\n", output_path, total);

    xune_embedding_free_result(result);
    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
    return 0;
}
