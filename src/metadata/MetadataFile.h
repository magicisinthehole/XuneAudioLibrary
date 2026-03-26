#pragma once

#include <xune_audio/xune_metadata.h>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <tpropertymap.h>

namespace TagLib { class FileRef; }

namespace xune {

struct PictureData {
    std::vector<uint8_t> data;
    std::string mime_type;
};

class MetadataFile {
public:
    explicit MetadataFile(const std::string& path);
    ~MetadataFile();

    MetadataFile(const MetadataFile&) = delete;
    MetadataFile& operator=(const MetadataFile&) = delete;

    bool is_valid() const;

    // ── File Properties ──────────────────────────────────────────────────
    int duration_ms() const;
    int bitrate() const;
    int sample_rate() const;
    int bits_per_sample() const;
    std::string codec_description() const;
    xune_media_type_t media_type() const;

    // ── Single-Value Tags ────────────────────────────────────────────────
    std::string title() const;
    std::string title_sort() const;
    std::string album() const;
    std::string album_sort() const;
    std::string conductor() const;
    std::string genre() const;
    uint32_t track_number() const;
    uint32_t disc_number() const;
    uint32_t year() const;

    // ── Multi-Value Tags (individual entries, not flattened) ─────────────
    // ARTISTS tag (individual artist names) with ARTIST fallback
    std::vector<std::string> artists() const;
    // ARTIST tag (formatted display string, e.g., "A feat. B")
    std::string artist_display() const;
    std::vector<std::string> album_artists() const;
    std::vector<std::string> composers() const;

    // Sort names
    std::string artist_sort() const;
    std::string album_artist_sort() const;

    // ── MusicBrainz IDs (multi-value where applicable) ───────────────────
    std::vector<std::string> mb_artist_ids() const;
    std::vector<std::string> mb_album_artist_ids() const;
    std::string mb_recording_id() const;
    std::string mb_release_track_id() const;
    std::string mb_release_id() const;
    std::string mb_release_group_id() const;
    std::string acoustid_fingerprint() const;

    // ── ReplayGain ───────────────────────────────────────────────────────
    double replaygain_track_gain() const; // NaN if not present
    double replaygain_track_peak() const;
    double replaygain_album_gain() const;
    double replaygain_album_peak() const;

    // ── Artwork ──────────────────────────────────────────────────────────
    bool has_picture() const;
    PictureData picture() const;
    void set_picture(const uint8_t* data, int size, const std::string& mime_type);

    // ── Release Date ─────────────────────────────────────────────────────
    std::string release_date() const;

    // ── Writing ──────────────────────────────────────────────────────────
    void set_title(const std::string& value);
    void set_album(const std::string& value);
    void set_conductor(const std::string& value);
    void set_genre(const std::string& value);
    void set_track_number(uint32_t value);
    void set_disc_number(uint32_t value);
    void set_year(uint32_t value);
    void set_artists(const std::vector<std::string>& values);
    void set_album_artists(const std::vector<std::string>& values);
    void set_composers(const std::vector<std::string>& values);
    void set_mb_artist_ids(const std::vector<std::string>& values);
    void set_mb_album_artist_ids(const std::vector<std::string>& values);
    void set_mb_recording_id(const std::string& value);
    void set_mb_release_track_id(const std::string& value);
    void set_mb_release_id(const std::string& value);
    void set_mb_release_group_id(const std::string& value);
    void set_acoustid_fingerprint(const std::string& value);
    void set_replaygain_track_gain(double value);
    void set_replaygain_track_peak(double value);
    void set_replaygain_album_gain(double value);
    void set_replaygain_album_peak(double value);

    bool save();

private:
    // ReplayGain helpers: handle both Vorbis-style and WM-style ASF names
    double read_replaygain(const char* props_key,
                           const char* asf_vorbis_key,
                           const char* asf_wm_key) const;
    void write_replaygain(const char* props_key,
                          const char* asf_vorbis_key,
                          const char* asf_wm_key,
                          const char* formatted_value);
    void clear_replaygain(const char* props_key,
                          const char* asf_vorbis_key,
                          const char* asf_wm_key);

    // Format-specific multi-value reading helpers
    std::vector<std::string> read_xiph_field(const char* field_name) const;
    std::vector<std::string> read_id3v2_txxx(const char* description) const;
    std::vector<std::string> read_mp4_text(const char* atom_id) const;
    std::vector<std::string> read_asf_attribute(const char* attr_name) const;

    // Format-specific multi-value writing helpers
    void write_xiph_field(const char* field_name, const std::vector<std::string>& values);
    void write_id3v2_txxx(const char* description, const std::vector<std::string>& values);
    void write_mp4_text(const char* atom_id, const std::vector<std::string>& values);
    void write_asf_attribute(const char* attr_name, const std::vector<std::string>& values);

    // Write to all applicable format-specific tags
    void write_custom_field(const char* xiph_name, const char* id3v2_desc,
                           const char* asf_name, const char* mp4_atom,
                           const std::vector<std::string>& values);
    void write_custom_field(const char* xiph_name, const char* id3v2_desc,
                           const char* asf_name, const char* mp4_atom,
                           const std::string& value);

    void flush_properties();

    std::unique_ptr<TagLib::FileRef> file_ref_;
    std::string path_;
    TagLib::PropertyMap props_;
};

} // namespace xune
