#include <xune_audio/xune_metadata.h>
#include "MetadataFile.h"

#include <cmath>
#include <cstring>
#include <fstream>

// ── Handle Wrapper ───────────────────────────────────────────────────────────
// Holds the MetadataFile plus cached string/data values for pointer stability.

struct MetadataHandle {
    xune::MetadataFile file;

    std::string title_cache;
    std::string title_sort_cache;
    std::string album_cache;
    std::string album_sort_cache;
    std::string conductor_cache;
    std::string genre_cache;
    std::string artist_display_cache;
    std::string artist_sort_cache;
    std::string album_artist_sort_cache;
    std::string codec_desc_cache;
    std::string release_date_cache;
    std::string picture_mime_cache;

    std::vector<std::string> artists_cache;
    std::vector<std::string> album_artists_cache;
    std::vector<std::string> composers_cache;
    std::vector<std::string> mb_artist_ids_cache;
    std::vector<std::string> mb_album_artist_ids_cache;

    std::string mb_recording_id_cache;
    std::string mb_release_track_id_cache;
    std::string mb_release_id_cache;
    std::string mb_release_group_id_cache;
    std::string acoustid_fp_cache;

    xune::PictureData picture_cache;
    bool picture_loaded = false;

    bool caches_loaded = false;

    explicit MetadataHandle(const std::string& path) : file(path) {}

    void load_caches() {
        if (caches_loaded) return;
        caches_loaded = true;

        title_cache = file.title();
        title_sort_cache = file.title_sort();
        album_cache = file.album();
        album_sort_cache = file.album_sort();
        conductor_cache = file.conductor();
        genre_cache = file.genre();
        artist_display_cache = file.artist_display();
        artist_sort_cache = file.artist_sort();
        album_artist_sort_cache = file.album_artist_sort();
        codec_desc_cache = file.codec_description();
        release_date_cache = file.release_date();

        artists_cache = file.artists();
        album_artists_cache = file.album_artists();
        composers_cache = file.composers();
        mb_artist_ids_cache = file.mb_artist_ids();
        mb_album_artist_ids_cache = file.mb_album_artist_ids();

        mb_recording_id_cache = file.mb_recording_id();
        mb_release_track_id_cache = file.mb_release_track_id();
        mb_release_id_cache = file.mb_release_id();
        mb_release_group_id_cache = file.mb_release_group_id();
        acoustid_fp_cache = file.acoustid_fingerprint();
    }

    void load_picture() {
        if (picture_loaded) return;
        picture_loaded = true;
        if (file.has_picture())
            picture_cache = file.picture();
    }
};

static MetadataHandle* get(xune_meta_handle_t h, bool ensure_caches = true) {
    auto* m = static_cast<MetadataHandle*>(h);
    if (m && ensure_caches) m->load_caches();
    return m;
}

static const char* c_str_or_null(const std::string& s) {
    return s.empty() ? nullptr : s.c_str();
}

// ── Open / Close ─────────────────────────────────────────────────────────────

xune_meta_error_t xune_meta_open(const char* path, xune_meta_handle_t* out_handle) {
    if (!path || !out_handle) return XUNE_META_ERROR_INVALID_ARGS;

    auto* handle = new MetadataHandle(path);
    if (!handle->file.is_valid()) {
        delete handle;
        std::ifstream probe(path);
        return probe.good() ? XUNE_META_ERROR_CORRUPT_FILE : XUNE_META_ERROR_FILE_NOT_FOUND;
    }

    *out_handle = handle;
    return XUNE_META_OK;
}

void xune_meta_close(xune_meta_handle_t handle) {
    delete get(handle);
}

// ── File Properties ──────────────────────────────────────────────────────────

int xune_meta_duration_ms(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.duration_ms() : 0;
}

int xune_meta_bitrate(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.bitrate() : 0;
}

int xune_meta_sample_rate(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.sample_rate() : 0;
}

int xune_meta_bits_per_sample(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.bits_per_sample() : 0;
}

const char* xune_meta_codec_description(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->codec_desc_cache) : nullptr;
}

xune_media_type_t xune_meta_media_type(xune_meta_handle_t h) {
    auto* m = get(h);
    return m ? m->file.media_type() : XUNE_MEDIA_UNKNOWN;
}

// ── Single-Value Tags ────────────────────────────────────────────────────────

const char* xune_meta_title(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->title_cache) : nullptr;
}

const char* xune_meta_title_sort(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->title_sort_cache) : nullptr;
}

const char* xune_meta_album(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->album_cache) : nullptr;
}

const char* xune_meta_album_sort(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->album_sort_cache) : nullptr;
}

const char* xune_meta_conductor(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->conductor_cache) : nullptr;
}

const char* xune_meta_genre(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->genre_cache) : nullptr;
}

uint32_t xune_meta_track_number(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.track_number() : 0;
}

uint32_t xune_meta_disc_number(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.disc_number() : 0;
}

uint32_t xune_meta_year(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.year() : 0;
}

// ── Multi-Value Tags ─────────────────────────────────────────────────────────

int xune_meta_artist_count(xune_meta_handle_t h) {
    auto* m = get(h); return m ? static_cast<int>(m->artists_cache.size()) : 0;
}

const char* xune_meta_artist_at(xune_meta_handle_t h, int index) {
    auto* m = get(h);
    if (!m || index < 0 || index >= static_cast<int>(m->artists_cache.size())) return nullptr;
    return m->artists_cache[index].c_str();
}

const char* xune_meta_artist_display(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->artist_display_cache) : nullptr;
}

int xune_meta_album_artist_count(xune_meta_handle_t h) {
    auto* m = get(h); return m ? static_cast<int>(m->album_artists_cache.size()) : 0;
}

const char* xune_meta_album_artist_at(xune_meta_handle_t h, int index) {
    auto* m = get(h);
    if (!m || index < 0 || index >= static_cast<int>(m->album_artists_cache.size())) return nullptr;
    return m->album_artists_cache[index].c_str();
}

int xune_meta_composer_count(xune_meta_handle_t h) {
    auto* m = get(h); return m ? static_cast<int>(m->composers_cache.size()) : 0;
}

const char* xune_meta_composer_at(xune_meta_handle_t h, int index) {
    auto* m = get(h);
    if (!m || index < 0 || index >= static_cast<int>(m->composers_cache.size())) return nullptr;
    return m->composers_cache[index].c_str();
}

const char* xune_meta_artist_sort(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->artist_sort_cache) : nullptr;
}

const char* xune_meta_album_artist_sort(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->album_artist_sort_cache) : nullptr;
}

// ── MusicBrainz IDs ─────────────────────────────────────────────────────────

int xune_meta_mb_artist_id_count(xune_meta_handle_t h) {
    auto* m = get(h); return m ? static_cast<int>(m->mb_artist_ids_cache.size()) : 0;
}

const char* xune_meta_mb_artist_id_at(xune_meta_handle_t h, int index) {
    auto* m = get(h);
    if (!m || index < 0 || index >= static_cast<int>(m->mb_artist_ids_cache.size())) return nullptr;
    return m->mb_artist_ids_cache[index].c_str();
}

int xune_meta_mb_album_artist_id_count(xune_meta_handle_t h) {
    auto* m = get(h); return m ? static_cast<int>(m->mb_album_artist_ids_cache.size()) : 0;
}

const char* xune_meta_mb_album_artist_id_at(xune_meta_handle_t h, int index) {
    auto* m = get(h);
    if (!m || index < 0 || index >= static_cast<int>(m->mb_album_artist_ids_cache.size())) return nullptr;
    return m->mb_album_artist_ids_cache[index].c_str();
}

const char* xune_meta_mb_recording_id(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->mb_recording_id_cache) : nullptr;
}

const char* xune_meta_mb_release_track_id(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->mb_release_track_id_cache) : nullptr;
}

const char* xune_meta_mb_release_id(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->mb_release_id_cache) : nullptr;
}

const char* xune_meta_mb_release_group_id(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->mb_release_group_id_cache) : nullptr;
}

const char* xune_meta_acoustid_fingerprint(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->acoustid_fp_cache) : nullptr;
}

// ── ReplayGain ───────────────────────────────────────────────────────────────

double xune_meta_replaygain_track_gain(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.replaygain_track_gain() : std::nan("");
}

double xune_meta_replaygain_track_peak(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.replaygain_track_peak() : std::nan("");
}

double xune_meta_replaygain_album_gain(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.replaygain_album_gain() : std::nan("");
}

double xune_meta_replaygain_album_peak(xune_meta_handle_t h) {
    auto* m = get(h); return m ? m->file.replaygain_album_peak() : std::nan("");
}

// ── Artwork ──────────────────────────────────────────────────────────────────

int xune_meta_has_picture(xune_meta_handle_t h) {
    auto* m = get(h);
    if (!m) return 0;
    m->load_picture();
    return m->picture_cache.data.empty() ? 0 : 1;
}

const uint8_t* xune_meta_picture_data(xune_meta_handle_t h, int* out_size) {
    auto* m = get(h);
    if (!m) { if (out_size) *out_size = 0; return nullptr; }
    m->load_picture();
    if (m->picture_cache.data.empty()) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    if (out_size) *out_size = static_cast<int>(m->picture_cache.data.size());
    return m->picture_cache.data.data();
}

const char* xune_meta_picture_mime(xune_meta_handle_t h) {
    auto* m = get(h);
    if (!m) return nullptr;
    m->load_picture();
    if (m->picture_cache.mime_type.empty()) return nullptr;
    return m->picture_cache.mime_type.c_str();
}

// ── Release Date ─────────────────────────────────────────────────────────────

const char* xune_meta_release_date(xune_meta_handle_t h) {
    auto* m = get(h); return m ? c_str_or_null(m->release_date_cache) : nullptr;
}

// ── Writing ──────────────────────────────────────────────────────────────────

static std::vector<std::string> to_vec(const char** values, int count) {
    std::vector<std::string> result;
    for (int i = 0; i < count; i++)
        if (values[i]) result.emplace_back(values[i]);
    return result;
}

void xune_meta_set_title(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_title(v);
    if (m->caches_loaded) m->title_cache = std::move(v);
}

void xune_meta_set_album(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_album(v);
    if (m->caches_loaded) m->album_cache = std::move(v);
}

void xune_meta_set_conductor(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_conductor(v);
    if (m->caches_loaded) m->conductor_cache = std::move(v);
}

void xune_meta_set_genre(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_genre(v);
    if (m->caches_loaded) m->genre_cache = std::move(v);
}

void xune_meta_set_track_number(xune_meta_handle_t h, uint32_t value) {
    auto* m = get(h, false); if (m) m->file.set_track_number(value);
}

void xune_meta_set_disc_number(xune_meta_handle_t h, uint32_t value) {
    auto* m = get(h, false); if (m) m->file.set_disc_number(value);
}

void xune_meta_set_year(xune_meta_handle_t h, uint32_t value) {
    auto* m = get(h, false); if (m) m->file.set_year(value);
}

void xune_meta_set_artists(xune_meta_handle_t h, const char** values, int count) {
    auto* m = get(h, false);
    if (!m) return;
    auto vec = to_vec(values, count);
    m->file.set_artists(vec);
    if (m->caches_loaded) m->artists_cache = std::move(vec);
}

void xune_meta_set_album_artists(xune_meta_handle_t h, const char** values, int count) {
    auto* m = get(h, false);
    if (!m) return;
    auto vec = to_vec(values, count);
    m->file.set_album_artists(vec);
    if (m->caches_loaded) m->album_artists_cache = std::move(vec);
}

void xune_meta_set_composers(xune_meta_handle_t h, const char** values, int count) {
    auto* m = get(h, false);
    if (!m) return;
    auto vec = to_vec(values, count);
    m->file.set_composers(vec);
    if (m->caches_loaded) m->composers_cache = std::move(vec);
}

void xune_meta_set_mb_artist_ids(xune_meta_handle_t h, const char** values, int count) {
    auto* m = get(h, false);
    if (!m) return;
    auto vec = to_vec(values, count);
    m->file.set_mb_artist_ids(vec);
    if (m->caches_loaded) m->mb_artist_ids_cache = std::move(vec);
}

void xune_meta_set_mb_album_artist_ids(xune_meta_handle_t h, const char** values, int count) {
    auto* m = get(h, false);
    if (!m) return;
    auto vec = to_vec(values, count);
    m->file.set_mb_album_artist_ids(vec);
    if (m->caches_loaded) m->mb_album_artist_ids_cache = std::move(vec);
}

void xune_meta_set_mb_recording_id(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_mb_recording_id(v);
    if (m->caches_loaded) m->mb_recording_id_cache = std::move(v);
}

void xune_meta_set_mb_release_track_id(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_mb_release_track_id(v);
    if (m->caches_loaded) m->mb_release_track_id_cache = std::move(v);
}

void xune_meta_set_mb_release_id(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_mb_release_id(v);
    if (m->caches_loaded) m->mb_release_id_cache = std::move(v);
}

void xune_meta_set_mb_release_group_id(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_mb_release_group_id(v);
    if (m->caches_loaded) m->mb_release_group_id_cache = std::move(v);
}

void xune_meta_set_acoustid_fingerprint(xune_meta_handle_t h, const char* value) {
    auto* m = get(h, false);
    if (!m) return;
    std::string v = value ? value : "";
    m->file.set_acoustid_fingerprint(v);
    if (m->caches_loaded) m->acoustid_fp_cache = std::move(v);
}

void xune_meta_set_replaygain_track_gain(xune_meta_handle_t h, double value) {
    auto* m = get(h, false); if (m) m->file.set_replaygain_track_gain(value);
}

void xune_meta_set_replaygain_track_peak(xune_meta_handle_t h, double value) {
    auto* m = get(h, false); if (m) m->file.set_replaygain_track_peak(value);
}

void xune_meta_set_replaygain_album_gain(xune_meta_handle_t h, double value) {
    auto* m = get(h, false); if (m) m->file.set_replaygain_album_gain(value);
}

void xune_meta_set_replaygain_album_peak(xune_meta_handle_t h, double value) {
    auto* m = get(h, false); if (m) m->file.set_replaygain_album_peak(value);
}

xune_meta_error_t xune_meta_save(xune_meta_handle_t h) {
    auto* m = get(h, false);
    if (!m) return XUNE_META_ERROR_INVALID_HANDLE;
    return m->file.save() ? XUNE_META_OK : XUNE_META_ERROR_WRITE_FAILED;
}
