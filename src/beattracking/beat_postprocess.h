/**
 * @file beat_postprocess.h
 * @brief Beat post-processing: chunking, aggregation, and peak picking.
 *
 * Matches the Beat This! Postprocessor (type="minimal") logic:
 *   1. Chunk mel into 1500-frame windows (30s at 50fps) with 6-frame borders
 *   2. Run inference per chunk, aggregate with keep_first overlap handling
 *   3. Max-pool peak picking on logits (kernel=7) + threshold logit > 0
 *   4. Deduplicate adjacent peaks (collapse to running average)
 *   5. Snap downbeats to nearest beat
 *   6. Convert frame indices to timestamps (frame / 50.0 seconds)
 */

#pragma once

#include <vector>

namespace xune {
namespace beattracking {

class BeatInference;

struct BeatResult {
    std::vector<float> beats;       // Beat timestamps in seconds
    std::vector<float> downbeats;   // Downbeat timestamps in seconds
};

class BeatPostprocessor {
public:
    static constexpr int kChunkSize = 1500;     // 30s at 50fps
    static constexpr int kBorderSize = 6;       // Border frames to discard
    static constexpr float kFps = 50.0f;
    static constexpr int kMaxPoolKernel = 7;    // ±3 frames (±60ms)

    /**
     * Run full beat tracking pipeline on a mel spectrogram.
     *
     * @param mel_data Full mel spectrogram [n_frames x 128], time-first
     * @param n_frames Total number of time frames
     * @param inference BeatInference instance (loaded model)
     * @param result Output beat/downbeat timestamps
     * @return true on success
     */
    static bool Process(const float* mel_data, int n_frames,
                        BeatInference& inference,
                        BeatResult& result);

    /**
     * Post-process batched inference output into per-chunk BeatResults.
     *
     * Takes the raw logit output from a batched RunInference call and
     * runs peak picking + downbeat snapping independently per chunk.
     * Timestamps are offset by frame_offsets[i] / kFps so they are
     * absolute (relative to the start of the source track).
     *
     * @param beat_logits Flat [total_chunks * chunk_frames] beat logits
     * @param db_logits Flat [total_chunks * chunk_frames] downbeat logits
     * @param total_chunks Number of chunks in the batch
     * @param chunk_frames Number of frames per chunk (all equal)
     * @param frame_offsets Per-chunk frame offset within the source track
     * @param results Output: one BeatResult per chunk
     */
    static void PostprocessChunks(const float* beat_logits,
                                  const float* db_logits,
                                  int total_chunks, int chunk_frames,
                                  const int* frame_offsets,
                                  std::vector<BeatResult>& results);

private:
    /// Peak pick on raw logits: max-pool(kernel=7) local maxima, logit > 0,
    /// deduplicate adjacent peaks to their average frame index.
    static void PeakPick(const std::vector<float>& logits, int n_frames,
                         std::vector<float>& timestamps);

    /// Snap each downbeat timestamp to the nearest beat timestamp.
    static void SnapDownbeatsToBeats(const std::vector<float>& beats,
                                     std::vector<float>& downbeats);
};

}  // namespace beattracking
}  // namespace xune
