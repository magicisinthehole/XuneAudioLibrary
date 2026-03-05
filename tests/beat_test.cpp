/**
 * @file beat_test.cpp
 * @brief Tests for the Beat This! beat tracking pipeline.
 *
 * Validates the C++ mel spectrogram, model inference, and post-processing
 * against Python reference data exported by SmartDJ/BeatTracking/export_reference_data.py.
 *
 * Test data paths are set via CMake compile definitions:
 *   BEAT_REFERENCE_DATA_DIR — directory containing reference .bin files
 *   BEAT_MODEL_PATH         — path to beat model (.safetensors or .onnx)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "beattracking/beat_mel_spectrogram.h"
#include "beattracking/beat_inference.h"
#include "beattracking/beat_postprocess.h"
#include "xune_audio/xune_beat.h"
#include "test_utils.h"

#ifndef BEAT_REFERENCE_DATA_DIR
#error "BEAT_REFERENCE_DATA_DIR must be defined by CMake"
#endif
#ifndef BEAT_MODEL_PATH
#error "BEAT_MODEL_PATH must be defined by CMake"
#endif

using xune::test::LoadFloatBin;
using xune::test::CosineSimilarity;

namespace {

const std::string kBeatRefDir = BEAT_REFERENCE_DATA_DIR;
const std::string kBeatModelPath = BEAT_MODEL_PATH;

}  // namespace

// ============================================================================
// Beat Mel Spectrogram Tests
// ============================================================================

class BeatMelTest : public ::testing::Test {
protected:
    void SetUp() override {
        pcm_ = LoadFloatBin(kBeatRefDir + "/test_pcm_mono_22k.bin");
        ASSERT_FALSE(pcm_.empty()) << "Failed to load reference PCM";

        ref_mel_ = LoadFloatBin(kBeatRefDir + "/reference_beat_mel.bin");
        ASSERT_FALSE(ref_mel_.empty()) << "Failed to load reference mel";
    }

    std::vector<float> pcm_;
    std::vector<float> ref_mel_;
};

TEST_F(BeatMelTest, OutputShapeMatchesReference) {
    xune::beattracking::BeatMelSpectrogram mel;
    xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames, scratch);
    ASSERT_TRUE(ok);

    // Reference mel is [n_frames x 128] (time-first)
    int ref_n_frames = static_cast<int>(ref_mel_.size()) / 128;

    EXPECT_EQ(n_frames, ref_n_frames)
        << "Frame count mismatch: got " << n_frames << ", expected " << ref_n_frames;
    EXPECT_EQ(out_mel.size(), ref_mel_.size())
        << "Total size mismatch: got " << out_mel.size() << ", expected " << ref_mel_.size();
}

TEST_F(BeatMelTest, CosineSimilarityAboveThreshold) {
    xune::beattracking::BeatMelSpectrogram mel;
    xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames, scratch);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

    float sim = CosineSimilarity(out_mel.data(), ref_mel_.data(), out_mel.size());
    EXPECT_GT(sim, 0.999f)
        << "Mel spectrogram cosine similarity " << sim << " below threshold 0.999";
}

TEST_F(BeatMelTest, MagnitudeMatchesReference) {
    xune::beattracking::BeatMelSpectrogram mel;
    xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames, scratch);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

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

    printf("  C++ mel: mean=%.6f, max=%.6f\n", cpp_mean, cpp_max);
    printf("  Ref mel: mean=%.6f, max=%.6f\n", ref_mean, ref_max);
    printf("  Mean ratio (C++/ref): %.6f\n", mean_ratio);
    printf("  Max ratio (C++/ref): %.6f\n", max_ratio);
    printf("  Max absolute difference: %.6f\n", max_abs_diff);

    EXPECT_NEAR(mean_ratio, 1.0f, 0.05f)
        << "Mel mean magnitude ratio " << mean_ratio << " deviates >5% from 1.0";
    EXPECT_NEAR(max_ratio, 1.0f, 0.05f)
        << "Mel max magnitude ratio " << max_ratio << " deviates >5% from 1.0";
}

TEST_F(BeatMelTest, PerFrameCosineSimilarity) {
    xune::beattracking::BeatMelSpectrogram mel;
    xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;
    std::vector<float> out_mel;
    int n_frames = 0;

    bool ok = mel.Compute(pcm_.data(), static_cast<int>(pcm_.size()),
                          out_mel, n_frames, scratch);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out_mel.size(), ref_mel_.size());

    // Beat mel is time-first: [n_frames x 128], row-major
    // Row f = mel[f * 128 ... f * 128 + 127]
    float min_sim = 1.0f;
    int worst_frame = 0;
    for (int f = 0; f < n_frames; ++f) {
        float sim = CosineSimilarity(
            out_mel.data() + f * 128,
            ref_mel_.data() + f * 128,
            128);
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
// Beat Inference Tests
// ============================================================================

class BeatInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        ref_beat_logits_ = LoadFloatBin(kBeatRefDir + "/reference_beat_logits.bin");
        ASSERT_FALSE(ref_beat_logits_.empty()) << "Failed to load reference beat logits";

        ref_db_logits_ = LoadFloatBin(kBeatRefDir + "/reference_db_logits.bin");
        ASSERT_FALSE(ref_db_logits_.empty()) << "Failed to load reference downbeat logits";

        // Compute mel from PCM using C++ (matches how the full pipeline works)
        auto pcm = LoadFloatBin(kBeatRefDir + "/test_pcm_mono_22k.bin");
        ASSERT_FALSE(pcm.empty()) << "Failed to load reference PCM";

        xune::beattracking::BeatMelSpectrogram mel;
        xune::beattracking::BeatMelSpectrogram::ScratchBuffer scratch;
        bool ok = mel.Compute(pcm.data(), static_cast<int>(pcm.size()),
                              cpp_mel_, cpp_n_frames_, scratch);
        ASSERT_TRUE(ok) << "Failed to compute C++ mel spectrogram";
    }

    std::vector<float> cpp_mel_;
    int cpp_n_frames_ = 0;
    std::vector<float> ref_beat_logits_;
    std::vector<float> ref_db_logits_;
};

TEST_F(BeatInferenceTest, ModelLoads) {
    xune::beattracking::BeatInference model;
    bool ok = model.LoadModel(kBeatModelPath);
    ASSERT_TRUE(ok) << "Failed to load beat model from: " << kBeatModelPath;
    EXPECT_TRUE(model.IsReady());
}

TEST_F(BeatInferenceTest, InferenceProducesCorrectShape) {
    xune::beattracking::BeatInference model;
    ASSERT_TRUE(model.LoadModel(kBeatModelPath));

    std::vector<float> beat_logits, db_logits;
    bool ok = model.RunInference(cpp_mel_.data(), 1, cpp_n_frames_,
                                  beat_logits, db_logits);
    ASSERT_TRUE(ok);

    EXPECT_EQ(static_cast<int>(beat_logits.size()), cpp_n_frames_)
        << "Beat logits should have one value per frame";
    EXPECT_EQ(static_cast<int>(db_logits.size()), cpp_n_frames_)
        << "Downbeat logits should have one value per frame";
}

TEST_F(BeatInferenceTest, LogitsMatchReference) {
    xune::beattracking::BeatInference model;
    ASSERT_TRUE(model.LoadModel(kBeatModelPath));

    // Reference logits were generated from the same test signal
    int ref_n_frames = static_cast<int>(ref_beat_logits_.size());

    std::vector<float> beat_logits, db_logits;
    bool ok = model.RunInference(cpp_mel_.data(), 1, cpp_n_frames_,
                                  beat_logits, db_logits);
    ASSERT_TRUE(ok);

    // Frame counts may differ slightly due to mel computation differences
    int compare_frames = std::min(cpp_n_frames_, ref_n_frames);
    ASSERT_GT(compare_frames, 0);

    printf("  C++ frames: %d, Reference frames: %d, Comparing: %d\n",
           cpp_n_frames_, ref_n_frames, compare_frames);

    // Compare beat logits
    float beat_sim = CosineSimilarity(
        beat_logits.data(), ref_beat_logits_.data(), compare_frames);
    printf("  Beat logits cosine similarity: %.6f\n", beat_sim);

    // MLX vs PyTorch will have small numerical differences; 0.95 is reasonable
    EXPECT_GT(beat_sim, 0.95f)
        << "Beat logits cosine similarity " << beat_sim << " below threshold 0.95";

    // Compare downbeat logits
    float db_sim = CosineSimilarity(
        db_logits.data(), ref_db_logits_.data(), compare_frames);
    printf("  Downbeat logits cosine similarity: %.6f\n", db_sim);

    EXPECT_GT(db_sim, 0.95f)
        << "Downbeat logits cosine similarity " << db_sim << " below threshold 0.95";
}

// ============================================================================
// Post-Processing Tests (Unit Tests)
// ============================================================================

TEST(BeatPostprocessTest, ConstantsMatchBeatThis) {
    EXPECT_EQ(xune::beattracking::BeatPostprocessor::kChunkSize, 1500);
    EXPECT_EQ(xune::beattracking::BeatPostprocessor::kBorderSize, 6);
    EXPECT_FLOAT_EQ(xune::beattracking::BeatPostprocessor::kFps, 50.0f);
    EXPECT_EQ(xune::beattracking::BeatPostprocessor::kMaxPoolKernel, 7);
}

// ============================================================================
// Full Pipeline (C API) Tests
// ============================================================================

class BeatPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pcm_ = LoadFloatBin(kBeatRefDir + "/test_pcm_mono_22k.bin");
        ASSERT_FALSE(pcm_.empty()) << "Failed to load reference PCM";

        ref_beats_ = LoadFloatBin(kBeatRefDir + "/reference_beats.bin");
        ref_downbeats_ = LoadFloatBin(kBeatRefDir + "/reference_downbeats.bin");
    }

    std::vector<float> pcm_;
    std::vector<float> ref_beats_;
    std::vector<float> ref_downbeats_;
};

TEST_F(BeatPipelineTest, SessionLifecycle) {
    xune_beat_session_t* session = nullptr;
    auto err = xune_beat_session_create(kBeatModelPath.c_str(), nullptr, &session);
    ASSERT_EQ(err, XUNE_BEAT_OK);
    ASSERT_NE(session, nullptr);
    EXPECT_TRUE(xune_beat_is_available(session));

    xune_beat_session_destroy(session);
}

TEST_F(BeatPipelineTest, SessionWithCacheDirWritesOptimizedModel) {
    std::string cache_dir = std::string(BEAT_REFERENCE_DATA_DIR) + "/opt_cache";
    std::filesystem::create_directories(cache_dir);
    std::string opt_path = cache_dir + "/beat_this_small_opt.onnx";
    std::filesystem::remove(opt_path);

    xune_beat_session_t* session = nullptr;
    ASSERT_EQ(xune_beat_session_create(kBeatModelPath.c_str(), cache_dir.c_str(), &session),
              XUNE_BEAT_OK);
    ASSERT_NE(session, nullptr);

#ifndef XUNE_USE_MLX
    EXPECT_TRUE(std::filesystem::exists(opt_path))
        << "Expected optimized model at: " << opt_path;
#endif

    xune_beat_session_destroy(session);

    // Second load reuses the cached optimized model (ORT only)
    xune_beat_session_t* session2 = nullptr;
    ASSERT_EQ(xune_beat_session_create(kBeatModelPath.c_str(), cache_dir.c_str(), &session2),
              XUNE_BEAT_OK);
    xune_beat_session_destroy(session2);

    std::filesystem::remove_all(cache_dir);
}

TEST_F(BeatPipelineTest, SessionCreateWithBadPathFails) {
    xune_beat_session_t* session = nullptr;
    auto err = xune_beat_session_create("/nonexistent/model.onnx", nullptr, &session);
    EXPECT_NE(err, XUNE_BEAT_OK);
}

TEST_F(BeatPipelineTest, DestroyNullIsSafe) {
    xune_beat_session_destroy(nullptr);
}

TEST_F(BeatPipelineTest, FreeNullIsSafe) {
    xune_beat_free(nullptr);
}

TEST_F(BeatPipelineTest, NullArgsReturnError) {
    xune_beat_session_t* session = nullptr;
    ASSERT_EQ(xune_beat_session_create(kBeatModelPath.c_str(), nullptr, &session), XUNE_BEAT_OK);

    float* beats = nullptr;
    float* downbeats = nullptr;
    int beat_count = 0, db_count = 0;

    // Null PCM
    auto err = xune_beat_analyze(session, nullptr, 1000,
                                  &beats, &beat_count, &downbeats, &db_count);
    EXPECT_EQ(err, XUNE_BEAT_ERROR_INVALID_ARGS);

    // Zero samples
    float dummy = 0.0f;
    err = xune_beat_analyze(session, &dummy, 0,
                             &beats, &beat_count, &downbeats, &db_count);
    EXPECT_EQ(err, XUNE_BEAT_ERROR_INVALID_ARGS);

    xune_beat_session_destroy(session);
}

TEST_F(BeatPipelineTest, EndToEndProducesBeats) {
    xune_beat_session_t* session = nullptr;
    ASSERT_EQ(xune_beat_session_create(kBeatModelPath.c_str(), nullptr, &session), XUNE_BEAT_OK);

    float* beats = nullptr;
    float* downbeats = nullptr;
    int beat_count = 0, db_count = 0;

    auto err = xune_beat_analyze(session, pcm_.data(),
                                  static_cast<int>(pcm_.size()),
                                  &beats, &beat_count, &downbeats, &db_count);
    ASSERT_EQ(err, XUNE_BEAT_OK);

    printf("  C++ beats: %d, C++ downbeats: %d\n", beat_count, db_count);
    printf("  Ref beats: %zu, Ref downbeats: %zu\n",
           ref_beats_.size(), ref_downbeats_.size());

    if (beat_count > 0) {
        printf("  C++ first 5 beats:");
        for (int i = 0; i < std::min(beat_count, 5); i++)
            printf(" %.3f", beats[i]);
        printf("\n");
    }
    if (!ref_beats_.empty()) {
        printf("  Ref first 5 beats:");
        for (size_t i = 0; i < std::min(ref_beats_.size(), size_t(5)); i++)
            printf(" %.3f", ref_beats_[i]);
        printf("\n");
    }

    // Beat timestamps should be monotonically increasing
    for (int i = 1; i < beat_count; i++) {
        EXPECT_GT(beats[i], beats[i - 1])
            << "Beats not monotonically increasing at index " << i;
    }

    // All beat timestamps should be non-negative and within audio duration
    float duration = static_cast<float>(pcm_.size()) / 22050.0f;
    for (int i = 0; i < beat_count; i++) {
        EXPECT_GE(beats[i], 0.0f);
        EXPECT_LE(beats[i], duration + 0.1f);
    }

    // If reference beats are available, check that C++ beats are within tolerance
    // Allow ±40ms (2 frames at 50fps) to account for mel/inference differences
    if (!ref_beats_.empty() && beat_count > 0) {
        int matched = 0;
        for (size_t r = 0; r < ref_beats_.size(); r++) {
            for (int c = 0; c < beat_count; c++) {
                if (std::fabs(beats[c] - ref_beats_[r]) < 0.04f) {
                    matched++;
                    break;
                }
            }
        }
        float match_ratio = static_cast<float>(matched) / ref_beats_.size();
        printf("  Beat match ratio (±40ms): %.1f%% (%d/%zu)\n",
               match_ratio * 100, matched, ref_beats_.size());

        // At least 80% of reference beats should have a C++ match
        EXPECT_GT(match_ratio, 0.8f)
            << "Only " << (match_ratio * 100) << "% of reference beats matched";
    }

    xune_beat_free(beats);
    xune_beat_free(downbeats);
    xune_beat_session_destroy(session);
}

TEST_F(BeatPipelineTest, DownbeatsAreSubsetOfBeats) {
    xune_beat_session_t* session = nullptr;
    ASSERT_EQ(xune_beat_session_create(kBeatModelPath.c_str(), nullptr, &session), XUNE_BEAT_OK);

    float* beats = nullptr;
    float* downbeats = nullptr;
    int beat_count = 0, db_count = 0;

    auto err = xune_beat_analyze(session, pcm_.data(),
                                  static_cast<int>(pcm_.size()),
                                  &beats, &beat_count, &downbeats, &db_count);
    ASSERT_EQ(err, XUNE_BEAT_OK);

    // After snapping, every downbeat should match a beat exactly
    for (int d = 0; d < db_count; d++) {
        bool found = false;
        for (int b = 0; b < beat_count; b++) {
            if (std::fabs(downbeats[d] - beats[b]) < 1e-6f) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found)
            << "Downbeat " << downbeats[d] << "s not found in beats";
    }

    xune_beat_free(beats);
    xune_beat_free(downbeats);
    xune_beat_session_destroy(session);
}
