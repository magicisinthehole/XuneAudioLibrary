/**
 * @file smartdj_test.cpp
 * @brief Tests for the SmartDJ embedding pipeline.
 *
 * Validates the C++ mel spectrogram and model inference against Python reference
 * data exported by SmartDJ/onnx_export/export_reference_data.py.
 *
 * Test data paths are set via CMake compile definitions:
 *   REFERENCE_DATA_DIR  — directory containing test_pcm_mono_16k.bin, reference_mel.bin, etc.
 *   MODEL_PATH          — path to model file (.onnx or .safetensors)
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "smartdj/mel_spectrogram.h"
#include "smartdj/model_inference.h"
#include "xune_audio/xune_embedding.h"
#include "test_utils.h"

#ifndef REFERENCE_DATA_DIR
#error "REFERENCE_DATA_DIR must be defined by CMake"
#endif
#ifndef MODEL_PATH
#error "MODEL_PATH must be defined by CMake"
#endif

using xune::test::LoadFloatBin;
using xune::test::CosineSimilarity;

namespace {

const std::string kRefDir = REFERENCE_DATA_DIR;
const std::string kModelPath = MODEL_PATH;

}  // namespace

// ============================================================================
// Mel Spectrogram Tests
// ============================================================================

class MelSpectrogramTest : public ::testing::Test {
protected:
    void SetUp() override {
        pcm_ = LoadFloatBin(kRefDir + "/test_pcm_mono_16k.bin");
        ASSERT_FALSE(pcm_.empty()) << "Failed to load reference PCM";

        ref_mel_ = LoadFloatBin(kRefDir + "/reference_mel.bin");
        ASSERT_FALSE(ref_mel_.empty()) << "Failed to load reference mel";
    }

    std::vector<float> pcm_;
    std::vector<float> ref_mel_;
};

TEST_F(MelSpectrogramTest, OutputShapeMatchesReference) {
    xune::smartdj::MelSpectrogram mel;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames);
    ASSERT_TRUE(ok);

    // Reference mel is 128 x 157 = 20096 floats
    int ref_n_mels = 128;
    int ref_n_frames = static_cast<int>(ref_mel_.size()) / ref_n_mels;

    EXPECT_EQ(n_frames, ref_n_frames)
        << "Frame count mismatch: got " << n_frames << ", expected " << ref_n_frames;
    EXPECT_EQ(out_mel.size(), ref_mel_.size())
        << "Total size mismatch: got " << out_mel.size() << ", expected " << ref_mel_.size();
}

TEST_F(MelSpectrogramTest, CosineSimilarityAboveThreshold) {
    xune::smartdj::MelSpectrogram mel;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

    float sim = CosineSimilarity(out_mel.data(), ref_mel_.data(), out_mel.size());
    EXPECT_GT(sim, 0.999f)
        << "Mel spectrogram cosine similarity " << sim << " below threshold 0.999";
}

TEST_F(MelSpectrogramTest, MagnitudeMatchesReference) {
    xune::smartdj::MelSpectrogram mel;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

    // Compute statistics for both
    float cpp_sum = 0, ref_sum = 0, cpp_max = 0, ref_max = 0;
    float max_abs_diff = 0;
    for (size_t i = 0; i < out_mel.size(); ++i) {
        cpp_sum += out_mel[i];
        ref_sum += ref_mel_[i];
        if (out_mel[i] > cpp_max) cpp_max = out_mel[i];
        if (ref_mel_[i] > ref_max) ref_max = ref_mel_[i];
        float diff = std::fabs(out_mel[i] - ref_mel_[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    float cpp_mean = cpp_sum / out_mel.size();
    float ref_mean = ref_sum / ref_mel_.size();
    float mean_ratio = cpp_mean / ref_mean;
    float max_ratio = cpp_max / ref_max;

    // Log diagnostics
    printf("  C++ mel: mean=%.6f, max=%.6f\n", cpp_mean, cpp_max);
    printf("  Ref mel: mean=%.6f, max=%.6f\n", ref_mean, ref_max);
    printf("  Mean ratio (C++/ref): %.6f\n", mean_ratio);
    printf("  Max ratio (C++/ref): %.6f\n", max_ratio);
    printf("  Max absolute difference: %.6f\n", max_abs_diff);

    // Magnitude should be within 5% of reference
    EXPECT_NEAR(mean_ratio, 1.0f, 0.05f)
        << "Mel mean magnitude ratio " << mean_ratio << " deviates >5% from 1.0";
    EXPECT_NEAR(max_ratio, 1.0f, 0.05f)
        << "Mel max magnitude ratio " << max_ratio << " deviates >5% from 1.0";
}

TEST_F(MelSpectrogramTest, PerFrameCosineSimilarity) {
    xune::smartdj::MelSpectrogram mel;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames);
    ASSERT_TRUE(ok);

    int n_mels = xune::smartdj::MelSpectrogram::kNMels;  // 128
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

    // Check cosine similarity per frame — every frame should be high
    float min_sim = 1.0f;
    int worst_frame = 0;
    for (int f = 0; f < n_frames; ++f) {
        // Mel is row-major (mel-first): mel[m * n_frames + f] for mel band m, frame f
        // Extract column f from both matrices
        std::vector<float> col_out(n_mels), col_ref(n_mels);
        for (int m = 0; m < n_mels; ++m) {
            col_out[m] = out_mel[m * n_frames + f];
            col_ref[m] = ref_mel_[m * n_frames + f];
        }

        float sim = CosineSimilarity(col_out.data(), col_ref.data(), n_mels);
        if (sim < min_sim) {
            min_sim = sim;
            worst_frame = f;
        }
    }

    EXPECT_GT(min_sim, 0.99f)
        << "Worst per-frame cosine similarity " << min_sim
        << " at frame " << worst_frame << " (threshold 0.99)";
}

// ============================================================================
// ONNX Inference Tests
// ============================================================================

class ModelInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Compute mel from PCM using the C++ implementation, matching the
        // reference embeddings (also generated by the C++ pipeline).
        auto pcm = LoadFloatBin(kRefDir + "/test_pcm_mono_16k.bin");
        ASSERT_FALSE(pcm.empty()) << "Failed to load reference PCM";

        xune::smartdj::MelSpectrogram mel;
        bool ok = mel.Compute(pcm.data(), static_cast<int>(pcm.size()),
                              cpp_mel_, cpp_mel_n_frames_);
        ASSERT_TRUE(ok) << "Failed to compute C++ mel spectrogram";

        ref_embeddings_ = LoadFloatBin(kRefDir + "/reference_embeddings.bin");
        ASSERT_FALSE(ref_embeddings_.empty()) << "Failed to load reference embeddings";
    }

    std::vector<float> cpp_mel_;
    int cpp_mel_n_frames_ = 0;
    std::vector<float> ref_embeddings_;
};

TEST_F(ModelInferenceTest, ModelLoads) {
    xune::smartdj::ModelInference onnx;
    bool ok = onnx.LoadModel(kModelPath);
    ASSERT_TRUE(ok) << "Failed to load ONNX model from: " << kModelPath;
    EXPECT_TRUE(onnx.IsReady());
}

TEST_F(ModelInferenceTest, InferenceMatchesReference) {
    xune::smartdj::ModelInference onnx;
    ASSERT_TRUE(onnx.LoadModel(kModelPath));

    // C++ mel is 128 x cpp_mel_n_frames_. Chunk: first 96 frames.
    int n_mels = 128;
    int chunk_frames = 96;
    ASSERT_GE(cpp_mel_n_frames_, chunk_frames);

    // Extract first chunk (128 x 96) from row-major mel (128 x cpp_mel_n_frames_)
    std::vector<float> chunk(n_mels * chunk_frames);
    for (int m = 0; m < n_mels; ++m) {
        for (int f = 0; f < chunk_frames; ++f) {
            chunk[m * chunk_frames + f] = cpp_mel_[m * cpp_mel_n_frames_ + f];
        }
    }

    // Run inference: batch=1
    std::vector<float> output(768);
    bool ok = onnx.RunInferenceInto(chunk.data(), 1, n_mels, chunk_frames,
                                     output.data(), static_cast<int>(output.size()));
    ASSERT_TRUE(ok);
    ASSERT_EQ(ref_embeddings_.size(), 768u);

    // Reference embeddings generated by C++ mel + FP32 ONNX model.
    // INT8 quantization introduces per-element diffs; cosine sim measures
    // whether the embedding direction is preserved (critical for similarity search).
    float sim = CosineSimilarity(output.data(), ref_embeddings_.data(), 768);
    EXPECT_GT(sim, 0.99f)
        << "ONNX inference cosine similarity " << sim << " below threshold 0.99";

    // Also check max absolute difference
    float max_diff = 0.0f;
    for (size_t i = 0; i < 768; ++i) {
        float diff = std::fabs(output[i] - ref_embeddings_[i]);
        if (diff > max_diff) max_diff = diff;
    }
    // INT8 dynamic quantization: max diff typically ~0.04 on random inputs,
    // up to ~0.2 on pathological inputs. 0.5 threshold catches real breakage.
    EXPECT_LT(max_diff, 0.5f)
        << "Max absolute difference " << max_diff << " exceeds 0.5";
}

TEST_F(ModelInferenceTest, BatchedInferenceConsistent) {
    xune::smartdj::ModelInference onnx;
    ASSERT_TRUE(onnx.LoadModel(kModelPath));

    int n_mels = 128;
    int chunk_frames = 96;
    ASSERT_GE(cpp_mel_n_frames_, chunk_frames);

    // Extract first chunk from C++ mel
    std::vector<float> chunk(n_mels * chunk_frames);
    for (int m = 0; m < n_mels; ++m) {
        for (int f = 0; f < chunk_frames; ++f) {
            chunk[m * chunk_frames + f] = cpp_mel_[m * cpp_mel_n_frames_ + f];
        }
    }

    // Run single inference
    std::vector<float> single_output(768);
    ASSERT_TRUE(onnx.RunInferenceInto(chunk.data(), 1, n_mels, chunk_frames,
                                       single_output.data(), 768));

    // Run batch=3 with same chunk repeated
    std::vector<float> batch_input;
    for (int i = 0; i < 3; ++i) {
        batch_input.insert(batch_input.end(), chunk.begin(), chunk.end());
    }

    std::vector<float> batch_output(3 * 768);
    ASSERT_TRUE(onnx.RunInferenceInto(batch_input.data(), 3, n_mels, chunk_frames,
                                       batch_output.data(), 3 * 768));

    // Each batch element should match single output
    for (int b = 0; b < 3; ++b) {
        float sim = CosineSimilarity(
            batch_output.data() + b * 768,
            single_output.data(),
            768);
        EXPECT_GT(sim, 0.9999f)
            << "Batch element " << b << " diverges from single (sim=" << sim << ")";
    }
}

// ============================================================================
// Full Pipeline (C API) Tests — Split Mel + Infer
// ============================================================================

class EmbeddingPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pcm_ = LoadFloatBin(kRefDir + "/test_pcm_mono_16k.bin");
        ASSERT_FALSE(pcm_.empty()) << "Failed to load reference PCM";

        ref_embeddings_ = LoadFloatBin(kRefDir + "/reference_embeddings.bin");
        ASSERT_FALSE(ref_embeddings_.empty()) << "Failed to load reference embeddings";
    }

    std::vector<float> pcm_;
    std::vector<float> ref_embeddings_;
};

TEST_F(EmbeddingPipelineTest, SessionLifecycle) {
    xune_embedding_session_t* session = nullptr;
    int err = xune_embedding_create(kModelPath.c_str(), nullptr, &session);
    ASSERT_EQ(err, XUNE_EMBEDDING_OK);
    ASSERT_NE(session, nullptr);
    EXPECT_TRUE(xune_embedding_is_available(session));

    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, SessionWithCacheDirWritesOptimizedModel) {
    std::string cache_dir = std::string(REFERENCE_DATA_DIR) + "/opt_cache";
    std::filesystem::create_directories(cache_dir);
    std::string opt_path = cache_dir + "/myna_hybrid_opt.onnx";
    std::filesystem::remove(opt_path);

    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), cache_dir.c_str(), &session),
              XUNE_EMBEDDING_OK);
    ASSERT_NE(session, nullptr);

#ifndef XUNE_USE_MLX
    EXPECT_TRUE(std::filesystem::exists(opt_path))
        << "Expected optimized model at: " << opt_path;
#endif

    xune_embedding_destroy(session);

    // Second load reuses the cached optimized model (ORT only)
    xune_embedding_session_t* session2 = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), cache_dir.c_str(), &session2),
              XUNE_EMBEDDING_OK);
    xune_embedding_destroy(session2);

    std::filesystem::remove_all(cache_dir);
}

TEST_F(EmbeddingPipelineTest, SessionCreateWithBadPathFails) {
    xune_embedding_session_t* session = nullptr;
    int err = xune_embedding_create("/nonexistent/model.onnx", nullptr, &session);
    EXPECT_NE(err, XUNE_EMBEDDING_OK);
}

TEST_F(EmbeddingPipelineTest, DestroyNullIsSafe) {
    xune_embedding_destroy(nullptr);  // Should not crash
}

TEST_F(EmbeddingPipelineTest, FreeNullMelIsSafe) {
    xune_embedding_free_mel(nullptr);  // Should not crash
}

TEST_F(EmbeddingPipelineTest, MelProducesCorrectChunkCount) {
    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    xune_embedding_mel_t* mel = nullptr;
    int err = xune_embedding_compute_mel(session, pcm_.data(),
                                          static_cast<int>(pcm_.size()), &mel);
    ASSERT_EQ(err, XUNE_EMBEDDING_OK);
    ASSERT_NE(mel, nullptr);

    // 5 seconds at 16kHz = 80000 samples -> 1 chunk of 96 frames
    EXPECT_EQ(xune_embedding_mel_chunk_count(mel), 1);
    EXPECT_NE(xune_embedding_mel_data(mel), nullptr);

    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, InferProducesCorrectDimensions) {
    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    xune_embedding_mel_t* mel = nullptr;
    ASSERT_EQ(xune_embedding_compute_mel(session, pcm_.data(),
                                          static_cast<int>(pcm_.size()), &mel),
              XUNE_EMBEDDING_OK);

    int chunks = xune_embedding_mel_chunk_count(mel);
    std::vector<float> embeddings(chunks * 768);
    int dims = 0;
    int err = xune_embedding_infer_into(session, xune_embedding_mel_data(mel),
                                         chunks, embeddings.data(),
                                         static_cast<int>(embeddings.size()), &dims);
    ASSERT_EQ(err, XUNE_EMBEDDING_OK);

    EXPECT_EQ(chunks, 1);
    EXPECT_EQ(dims, 768);

    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, EndToEndMatchesReference) {
    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    // Phase 1: mel
    xune_embedding_mel_t* mel = nullptr;
    ASSERT_EQ(xune_embedding_compute_mel(session, pcm_.data(),
                                          static_cast<int>(pcm_.size()), &mel),
              XUNE_EMBEDDING_OK);

    // Phase 2: infer into caller buffer
    int chunks = xune_embedding_mel_chunk_count(mel);
    std::vector<float> embeddings(chunks * 768);
    int dims = 0;
    ASSERT_EQ(xune_embedding_infer_into(session, xune_embedding_mel_data(mel),
                                         chunks, embeddings.data(),
                                         static_cast<int>(embeddings.size()), &dims),
              XUNE_EMBEDDING_OK);

    ASSERT_EQ(chunks, 1);
    ASSERT_EQ(dims, 768);
    ASSERT_EQ(ref_embeddings_.size(), 768u);

    float sim = CosineSimilarity(embeddings.data(), ref_embeddings_.data(), 768);
    EXPECT_GT(sim, 0.99f)
        << "End-to-end cosine similarity " << sim << " below threshold 0.99";

    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, LongerAudioProducesMultipleChunks) {
    // Generate 15 seconds of silence — should produce multiple chunks
    // 15s * 16000 = 240000 samples
    std::vector<float> long_pcm(240000, 0.0f);

    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    xune_embedding_mel_t* mel = nullptr;
    ASSERT_EQ(xune_embedding_compute_mel(session, long_pcm.data(),
                                          static_cast<int>(long_pcm.size()), &mel),
              XUNE_EMBEDDING_OK);

    int mel_chunks = xune_embedding_mel_chunk_count(mel);
    EXPECT_GT(mel_chunks, 1) << "15 seconds of audio should produce multiple chunks";

    std::vector<float> embeddings(mel_chunks * 768);
    int dims = 0;
    ASSERT_EQ(xune_embedding_infer_into(session, xune_embedding_mel_data(mel),
                                         mel_chunks, embeddings.data(),
                                         static_cast<int>(embeddings.size()), &dims),
              XUNE_EMBEDDING_OK);

    EXPECT_EQ(dims, 768);

    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, TooShortAudioReturnsError) {
    // Very short audio — not enough for a single chunk
    std::vector<float> short_pcm(100, 0.0f);

    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    xune_embedding_mel_t* mel = nullptr;
    int err = xune_embedding_compute_mel(session, short_pcm.data(),
                                          static_cast<int>(short_pcm.size()), &mel);
    EXPECT_EQ(err, XUNE_EMBEDDING_ERROR_NO_CHUNKS);

    xune_embedding_free_mel(mel);
    xune_embedding_destroy(session);
}

TEST_F(EmbeddingPipelineTest, CrossTrackBatchInference) {
    xune_embedding_session_t* session = nullptr;
    ASSERT_EQ(xune_embedding_create(kModelPath.c_str(), nullptr, &session), XUNE_EMBEDDING_OK);

    // Track A: original test PCM (5 seconds, 1 chunk)
    xune_embedding_mel_t* mel_a = nullptr;
    ASSERT_EQ(xune_embedding_compute_mel(session, pcm_.data(),
                                          static_cast<int>(pcm_.size()), &mel_a),
              XUNE_EMBEDDING_OK);

    // Track B: 15 seconds of silence (multiple chunks)
    std::vector<float> long_pcm(240000, 0.0f);
    xune_embedding_mel_t* mel_b = nullptr;
    ASSERT_EQ(xune_embedding_compute_mel(session, long_pcm.data(),
                                          static_cast<int>(long_pcm.size()), &mel_b),
              XUNE_EMBEDDING_OK);

    int chunks_a = xune_embedding_mel_chunk_count(mel_a);
    int chunks_b = xune_embedding_mel_chunk_count(mel_b);
    int total_chunks = chunks_a + chunks_b;

    ASSERT_EQ(chunks_a, 1);
    ASSERT_GT(chunks_b, 1);

    // Run inference separately for each track
    std::vector<float> separate_emb_a(chunks_a * 768);
    int dims_a = 0;
    ASSERT_EQ(xune_embedding_infer_into(session, xune_embedding_mel_data(mel_a),
                                         chunks_a, separate_emb_a.data(),
                                         static_cast<int>(separate_emb_a.size()), &dims_a),
              XUNE_EMBEDDING_OK);

    std::vector<float> separate_emb_b(chunks_b * 768);
    int dims_b = 0;
    ASSERT_EQ(xune_embedding_infer_into(session, xune_embedding_mel_data(mel_b),
                                         chunks_b, separate_emb_b.data(),
                                         static_cast<int>(separate_emb_b.size()), &dims_b),
              XUNE_EMBEDDING_OK);

    // Concatenate mel data from both tracks for batched inference
    int chunk_data_size = 128 * 96;
    std::vector<float> combined_mel(static_cast<size_t>(total_chunks) * chunk_data_size);

    const float* data_a = xune_embedding_mel_data(mel_a);
    const float* data_b = xune_embedding_mel_data(mel_b);

    std::memcpy(combined_mel.data(),
                data_a,
                static_cast<size_t>(chunks_a) * chunk_data_size * sizeof(float));
    std::memcpy(combined_mel.data() + static_cast<size_t>(chunks_a) * chunk_data_size,
                data_b,
                static_cast<size_t>(chunks_b) * chunk_data_size * sizeof(float));

    // Run combined batch inference
    std::vector<float> combined_emb(total_chunks * 768);
    int dims_combined = 0;
    ASSERT_EQ(xune_embedding_infer_into(session, combined_mel.data(),
                                         total_chunks, combined_emb.data(),
                                         static_cast<int>(combined_emb.size()), &dims_combined),
              XUNE_EMBEDDING_OK);

    // Track A's embedding from combined batch should match separate inference.
    // INT8 quantization can produce slightly different results at different batch sizes,
    // so we use 0.999 rather than 0.9999.
    float sim_a = CosineSimilarity(combined_emb.data(), separate_emb_a.data(), 768);
    EXPECT_GT(sim_a, 0.999f)
        << "Track A: combined vs separate cosine sim = " << sim_a;

    // Track B's embeddings from combined batch should match separate inference
    for (int c = 0; c < chunks_b; ++c) {
        float sim = CosineSimilarity(
            combined_emb.data() + (chunks_a + c) * 768,
            separate_emb_b.data() + c * 768,
            768);
        EXPECT_GT(sim, 0.999f)
            << "Track B chunk " << c << ": combined vs separate cosine sim = " << sim;
    }

    xune_embedding_free_mel(mel_b);
    xune_embedding_free_mel(mel_a);
    xune_embedding_destroy(session);
}
