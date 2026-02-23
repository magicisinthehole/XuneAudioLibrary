/**
 * @file beat_postprocess.cpp
 * @brief Beat post-processing implementation.
 *
 * Follows Beat This! Postprocessor (type="minimal") logic:
 *   - Chunks with border overlap for clean transitions
 *   - keep_first aggregation (first chunk's predictions are kept for overlapping regions)
 *   - Max-pool peak picking on raw logits (kernel=7) + threshold logit > 0
 *   - Deduplicate adjacent peaks (collapse to running average, width=1)
 *   - Snap downbeats to nearest beat, deduplicate
 */

#include "beat_postprocess.h"
#include "beat_inference.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace xune {
namespace beattracking {

bool BeatPostprocessor::Process(const float* mel_data, int n_frames,
                                 BeatInference& inference,
                                 BeatResult& result) {
    if (!mel_data || n_frames <= 0 || !inference.IsReady()) {
        return false;
    }

    // Single chunk — no batching needed
    if (n_frames <= kChunkSize) {
        std::vector<float> beat_logits, db_logits;
        if (!inference.RunInference(mel_data, 1, n_frames, beat_logits, db_logits)) {
            fprintf(stderr, "[xune_beat] Inference failed\n");
            return false;
        }
        PeakPick(beat_logits, n_frames, result.beats);
        PeakPick(db_logits, n_frames, result.downbeats);
        SnapDownbeatsToBeats(result.beats, result.downbeats);
        return true;
    }

    // How many non-border frames each chunk contributes
    int useful_size = kChunkSize - 2 * kBorderSize;  // 1488

    // === Pass 1: compute chunk positions ===
    struct ChunkInfo { int start; int frames; };
    std::vector<ChunkInfo> full_chunks;
    ChunkInfo short_chunk{-1, 0};

    for (int chunk_start = 0; chunk_start < n_frames; ) {
        int chunk_frames = std::min(kChunkSize, n_frames - chunk_start);

        if (chunk_frames == kChunkSize) {
            full_chunks.push_back({chunk_start, chunk_frames});
        } else {
            short_chunk = {chunk_start, chunk_frames};
        }

        if (chunk_start == 0) {
            chunk_start += kChunkSize - kBorderSize;
        } else {
            chunk_start += useful_size;
        }
    }

    // === Pass 2: pack full-size chunks into contiguous (N, kChunkSize, 128) ===
    int batch_size = static_cast<int>(full_chunks.size());
    size_t chunk_floats = static_cast<size_t>(kChunkSize) * 128;
    std::vector<float> packed_mel(batch_size * chunk_floats);

    for (int i = 0; i < batch_size; i++) {
        const float* src = mel_data + static_cast<size_t>(full_chunks[i].start) * 128;
        std::memcpy(packed_mel.data() + i * chunk_floats, src, chunk_floats * sizeof(float));
    }

    // === Pass 3: single batched inference call ===
    std::vector<float> batch_beat_logits, batch_db_logits;
    if (!inference.RunInference(packed_mel.data(), batch_size, kChunkSize,
                                batch_beat_logits, batch_db_logits)) {
        fprintf(stderr, "[xune_beat] Batched inference failed (%d chunks)\n", batch_size);
        return false;
    }

    // === Pass 4: aggregate logits with border handling ===
    std::vector<float> all_beat_logits(n_frames, 0.0f);
    std::vector<float> all_db_logits(n_frames, 0.0f);
    int write_offset = 0;

    for (int i = 0; i < batch_size; i++) {
        const float* chunk_beats = batch_beat_logits.data() + static_cast<size_t>(i) * kChunkSize;
        const float* chunk_dbs = batch_db_logits.data() + static_cast<size_t>(i) * kChunkSize;

        bool is_first = (full_chunks[i].start == 0);
        bool is_last = (i == batch_size - 1 && short_chunk.start < 0);

        int keep_start = is_first ? 0 : kBorderSize;
        int keep_end = is_last ? kChunkSize : (kChunkSize - kBorderSize);
        int keep_count = keep_end - keep_start;

        for (int j = 0; j < keep_count; j++) {
            int dst = write_offset + j;
            if (dst < n_frames) {
                all_beat_logits[dst] = chunk_beats[keep_start + j];
                all_db_logits[dst] = chunk_dbs[keep_start + j];
            }
        }
        write_offset += keep_count;
    }

    // === Handle short last chunk separately (different n_frames) ===
    if (short_chunk.start >= 0) {
        const float* chunk_mel = mel_data + static_cast<size_t>(short_chunk.start) * 128;
        std::vector<float> beat_logits, db_logits;
        if (!inference.RunInference(chunk_mel, 1, short_chunk.frames,
                                    beat_logits, db_logits)) {
            fprintf(stderr, "[xune_beat] Inference failed for short chunk at frame %d\n",
                    short_chunk.start);
            return false;
        }

        int keep_start = kBorderSize;
        int keep_end = short_chunk.frames;  // Last chunk: keep to end
        int keep_count = keep_end - keep_start;

        for (int j = 0; j < keep_count; j++) {
            int dst = write_offset + j;
            if (dst < n_frames) {
                all_beat_logits[dst] = beat_logits[keep_start + j];
                all_db_logits[dst] = db_logits[keep_start + j];
            }
        }
        write_offset += keep_count;
    }

    // Peak pick on raw logits: max-pool → threshold → dedup → timestamps
    PeakPick(all_beat_logits, n_frames, result.beats);
    PeakPick(all_db_logits, n_frames, result.downbeats);

    // Snap each downbeat to its nearest beat
    SnapDownbeatsToBeats(result.beats, result.downbeats);

    return true;
}

void BeatPostprocessor::PeakPick(const std::vector<float>& logits, int n_frames,
                                  std::vector<float>& timestamps) {
    timestamps.clear();
    if (n_frames <= 0) return;

    int half_k = kMaxPoolKernel / 2;  // 3

    // Step 1: Find local maxima via max-pool and threshold logit > 0
    // A frame is a peak if it equals the max in [i-3, i+3] AND logit > 0
    std::vector<int> peak_frames;
    for (int i = 0; i < n_frames; i++) {
        if (logits[i] <= 0.0f) continue;

        bool is_peak = true;
        for (int j = -half_k; j <= half_k; j++) {
            int idx = i + j;
            if (idx < 0 || idx >= n_frames || idx == i) continue;
            if (logits[idx] > logits[i]) {
                is_peak = false;
                break;
            }
        }

        if (is_peak) {
            peak_frames.push_back(i);
        }
    }

    if (peak_frames.empty()) return;

    // Step 2: Deduplicate adjacent peaks (width=1)
    // Groups of peaks separated by ≤1 frame are collapsed to running mean
    std::vector<float> deduped;
    float current = static_cast<float>(peak_frames[0]);
    int count = 1;

    for (size_t i = 1; i < peak_frames.size(); i++) {
        float p2 = static_cast<float>(peak_frames[i]);
        if (p2 - current <= 1.0f) {
            count++;
            current += (p2 - current) / count;
        } else {
            deduped.push_back(current);
            current = p2;
            count = 1;
        }
    }
    deduped.push_back(current);

    // Step 3: Convert frame positions to timestamps (seconds)
    for (float frame : deduped) {
        timestamps.push_back(frame / kFps);
    }
}

void BeatPostprocessor::SnapDownbeatsToBeats(const std::vector<float>& beats,
                                              std::vector<float>& downbeats) {
    if (beats.empty() || downbeats.empty()) return;

    // Snap each downbeat to the nearest beat timestamp
    for (size_t i = 0; i < downbeats.size(); i++) {
        float min_dist = std::fabs(beats[0] - downbeats[i]);
        float nearest = beats[0];
        for (size_t j = 1; j < beats.size(); j++) {
            float dist = std::fabs(beats[j] - downbeats[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = beats[j];
            }
        }
        downbeats[i] = nearest;
    }

    // Deduplicate (multiple downbeats may snap to same beat) and sort
    std::sort(downbeats.begin(), downbeats.end());
    auto last = std::unique(downbeats.begin(), downbeats.end());
    downbeats.erase(last, downbeats.end());
}

// ============================================================================
// Batched post-processing (caller handles mel packing and inference)
// ============================================================================

void BeatPostprocessor::PostprocessChunks(const float* beat_logits,
                                           const float* db_logits,
                                           int total_chunks, int chunk_frames,
                                           const int* frame_offsets,
                                           std::vector<BeatResult>& results) {
    results.resize(total_chunks);

    for (int i = 0; i < total_chunks; i++) {
        const float* chunk_bl = beat_logits + static_cast<size_t>(i) * chunk_frames;
        const float* chunk_dl = db_logits + static_cast<size_t>(i) * chunk_frames;

        std::vector<float> bl_vec(chunk_bl, chunk_bl + chunk_frames);
        std::vector<float> dl_vec(chunk_dl, chunk_dl + chunk_frames);

        PeakPick(bl_vec, chunk_frames, results[i].beats);
        PeakPick(dl_vec, chunk_frames, results[i].downbeats);
        SnapDownbeatsToBeats(results[i].beats, results[i].downbeats);

        // Offset timestamps to absolute position within the source track
        float time_offset = static_cast<float>(frame_offsets[i]) / kFps;
        if (time_offset > 0.0f) {
            for (auto& t : results[i].beats) t += time_offset;
            for (auto& t : results[i].downbeats) t += time_offset;
        }
    }
}

}  // namespace beattracking
}  // namespace xune
