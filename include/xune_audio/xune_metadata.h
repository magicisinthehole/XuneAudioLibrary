/// @file xune_metadata.h
/// @brief Audio file metadata reading and writing via TagLib C++.
///
/// Multi-value tags (ARTISTS, MUSICBRAINZ_ARTISTID, etc.) are exposed as
/// counted arrays, not flattened strings. This preserves individual artist
/// names and MBIDs for proper multi-artist credit support.

#pragma once

#include "xune_export.h"

#if XUNE_AUDIO_API_VERSION < 3
#error "xune_metadata.h requires XUNE_AUDIO_API_VERSION >= 3"
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque Handle ────────────────────────────────────────────────────────────

typedef void* xune_meta_handle_t;

// ── Error Codes ──────────────────────────────────────────────────────────────

typedef enum {
    XUNE_META_OK = 0,
    XUNE_META_ERROR_FILE_NOT_FOUND = -1,
    XUNE_META_ERROR_CORRUPT_FILE = -2,
    XUNE_META_ERROR_UNSUPPORTED_FORMAT = -3,
    XUNE_META_ERROR_WRITE_FAILED = -4,
    XUNE_META_ERROR_INVALID_HANDLE = -5,
    XUNE_META_ERROR_INVALID_ARGS = -6,
} xune_meta_error_t;

// ── Media Type ───────────────────────────────────────────────────────────────

typedef enum {
    XUNE_MEDIA_UNKNOWN = 0,
    XUNE_MEDIA_AUDIO = 1,
    XUNE_MEDIA_VIDEO = 2,
    XUNE_MEDIA_PHOTO = 3,
} xune_media_type_t;

// ── Open / Close ─────────────────────────────────────────────────────────────

/// Open an audio file for reading metadata.
/// @param path UTF-8 encoded file path
/// @param out_handle Receives opaque handle on success
/// @return XUNE_META_OK on success, error code on failure
XUNE_AUDIO_API xune_meta_error_t xune_meta_open(
    const char* path,
    xune_meta_handle_t* out_handle);

/// Close handle and free resources.
XUNE_AUDIO_API void xune_meta_close(xune_meta_handle_t handle);

// ── File Properties ──────────────────────────────────────────────────────────

XUNE_AUDIO_API int xune_meta_duration_ms(xune_meta_handle_t handle);
XUNE_AUDIO_API int xune_meta_bitrate(xune_meta_handle_t handle);
XUNE_AUDIO_API int xune_meta_sample_rate(xune_meta_handle_t handle);
XUNE_AUDIO_API int xune_meta_bits_per_sample(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_codec_description(xune_meta_handle_t handle);
XUNE_AUDIO_API xune_media_type_t xune_meta_media_type(xune_meta_handle_t handle);

// ── Single-Value String Tags ─────────────────────────────────────────────────
// Returns pointer valid until handle is closed. NULL if not present.

XUNE_AUDIO_API const char* xune_meta_title(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_title_sort(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_album(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_album_sort(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_conductor(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_genre(xune_meta_handle_t handle);
XUNE_AUDIO_API uint32_t xune_meta_track_number(xune_meta_handle_t handle);
XUNE_AUDIO_API uint32_t xune_meta_disc_number(xune_meta_handle_t handle);
XUNE_AUDIO_API uint32_t xune_meta_year(xune_meta_handle_t handle);

// ── Multi-Value Tags (proper arrays, not flattened) ──────────────────────────
// Individual artist names from the ARTISTS tag (falls back to ARTIST)
XUNE_AUDIO_API int xune_meta_artist_count(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_artist_at(xune_meta_handle_t handle, int index);
/// Formatted display string from the ARTIST tag (e.g., "A feat. B")
XUNE_AUDIO_API const char* xune_meta_artist_display(xune_meta_handle_t handle);

XUNE_AUDIO_API int xune_meta_album_artist_count(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_album_artist_at(xune_meta_handle_t handle, int index);

XUNE_AUDIO_API int xune_meta_composer_count(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_composer_at(xune_meta_handle_t handle, int index);

// Sort names
XUNE_AUDIO_API const char* xune_meta_artist_sort(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_album_artist_sort(xune_meta_handle_t handle);

// ── MusicBrainz IDs (multi-value where applicable) ───────────────────────────
XUNE_AUDIO_API int xune_meta_mb_artist_id_count(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_mb_artist_id_at(xune_meta_handle_t handle, int index);
XUNE_AUDIO_API int xune_meta_mb_album_artist_id_count(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_mb_album_artist_id_at(xune_meta_handle_t handle, int index);
XUNE_AUDIO_API const char* xune_meta_mb_recording_id(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_mb_release_track_id(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_mb_release_id(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_mb_release_group_id(xune_meta_handle_t handle);
XUNE_AUDIO_API const char* xune_meta_acoustid_fingerprint(xune_meta_handle_t handle);

// ── ReplayGain ───────────────────────────────────────────────────────────────
// Returns NaN if not present (check with isnan())
XUNE_AUDIO_API double xune_meta_replaygain_track_gain(xune_meta_handle_t handle);
XUNE_AUDIO_API double xune_meta_replaygain_track_peak(xune_meta_handle_t handle);
XUNE_AUDIO_API double xune_meta_replaygain_album_gain(xune_meta_handle_t handle);
XUNE_AUDIO_API double xune_meta_replaygain_album_peak(xune_meta_handle_t handle);

// ── Artwork ──────────────────────────────────────────────────────────────────
XUNE_AUDIO_API int xune_meta_has_picture(xune_meta_handle_t handle);
/// @param out_size Receives byte count of picture data
/// @return Pointer to picture bytes (valid until handle closed), NULL if none
XUNE_AUDIO_API const uint8_t* xune_meta_picture_data(xune_meta_handle_t handle, int* out_size);
XUNE_AUDIO_API const char* xune_meta_picture_mime(xune_meta_handle_t handle);

// ── Release Date ─────────────────────────────────────────────────────────────
/// Format-specific: TDRC for ID3v2, RELEASEDATE/DATE for Vorbis, etc.
XUNE_AUDIO_API const char* xune_meta_release_date(xune_meta_handle_t handle);

// ── Writing ──────────────────────────────────────────────────────────────────
// Single-value setters (NULL to clear)
XUNE_AUDIO_API void xune_meta_set_title(xune_meta_handle_t handle, const char* value);
XUNE_AUDIO_API void xune_meta_set_album(xune_meta_handle_t handle, const char* value);
XUNE_AUDIO_API void xune_meta_set_conductor(xune_meta_handle_t handle, const char* value);
XUNE_AUDIO_API void xune_meta_set_genre(xune_meta_handle_t handle, const char* value);
XUNE_AUDIO_API void xune_meta_set_track_number(xune_meta_handle_t handle, uint32_t value);
XUNE_AUDIO_API void xune_meta_set_disc_number(xune_meta_handle_t handle, uint32_t value);
XUNE_AUDIO_API void xune_meta_set_year(xune_meta_handle_t handle, uint32_t value);

/// Multi-value setters (pass array + count; count=0 to clear)
XUNE_AUDIO_API void xune_meta_set_artists(xune_meta_handle_t h, const char** values, int count);
XUNE_AUDIO_API void xune_meta_set_album_artists(xune_meta_handle_t h, const char** values, int count);
XUNE_AUDIO_API void xune_meta_set_composers(xune_meta_handle_t h, const char** values, int count);
XUNE_AUDIO_API void xune_meta_set_mb_artist_ids(xune_meta_handle_t h, const char** values, int count);
XUNE_AUDIO_API void xune_meta_set_mb_album_artist_ids(xune_meta_handle_t h, const char** values, int count);

// Single-value MusicBrainz setters (NULL to clear)
XUNE_AUDIO_API void xune_meta_set_mb_recording_id(xune_meta_handle_t h, const char* value);
XUNE_AUDIO_API void xune_meta_set_mb_release_track_id(xune_meta_handle_t h, const char* value);
XUNE_AUDIO_API void xune_meta_set_mb_release_id(xune_meta_handle_t h, const char* value);
XUNE_AUDIO_API void xune_meta_set_mb_release_group_id(xune_meta_handle_t h, const char* value);
XUNE_AUDIO_API void xune_meta_set_acoustid_fingerprint(xune_meta_handle_t h, const char* value);

// ReplayGain setters (NaN to clear)
XUNE_AUDIO_API void xune_meta_set_replaygain_track_gain(xune_meta_handle_t h, double value);
XUNE_AUDIO_API void xune_meta_set_replaygain_track_peak(xune_meta_handle_t h, double value);
XUNE_AUDIO_API void xune_meta_set_replaygain_album_gain(xune_meta_handle_t h, double value);
XUNE_AUDIO_API void xune_meta_set_replaygain_album_peak(xune_meta_handle_t h, double value);

/// Save all pending changes to disk.
/// @return XUNE_META_OK on success, XUNE_META_ERROR_WRITE_FAILED on failure
XUNE_AUDIO_API xune_meta_error_t xune_meta_save(xune_meta_handle_t handle);

#ifdef __cplusplus
}
#endif
