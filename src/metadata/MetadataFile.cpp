#include "MetadataFile.h"

#include <fileref.h>
#include <tag.h>

// Format-specific headers
#include <flacfile.h>
#include <mpegfile.h>
#include <mp4file.h>
#include <mp4properties.h>
#include <asffile.h>
#include <oggflacfile.h>
#include <vorbisfile.h>
#include <wavfile.h>

// Tag-type-specific headers
#include <id3v2tag.h>
#include <id3v2frame.h>
#include <textidentificationframe.h>
#include <urllinkframe.h>
#include <xiphcomment.h>
#include <mp4tag.h>
#include <asftag.h>
#include <attachedpictureframe.h>
#include <uniquefileidentifierframe.h>
#include <asfpicture.h>

#include <algorithm>
#include <cmath>

namespace xune {

// ── Construction / Destruction ───────────────────────────────────────────────

MetadataFile::MetadataFile(const std::string& path)
    : path_(path)
{
    file_ref_ = std::make_unique<TagLib::FileRef>(
        TagLib::FileName(path.c_str()));
    if (is_valid())
        props_ = file_ref_->file()->properties();
}

void MetadataFile::flush_properties() {
    if (is_valid())
        file_ref_->file()->setProperties(props_);
}

MetadataFile::~MetadataFile() = default;

bool MetadataFile::is_valid() const {
    return file_ref_ && !file_ref_->isNull() && file_ref_->file() && file_ref_->file()->isValid();
}

// ── File Properties ──────────────────────────────────────────────────────────

int MetadataFile::duration_ms() const {
    if (!is_valid() || !file_ref_->audioProperties()) return 0;
    auto ms = file_ref_->audioProperties()->lengthInMilliseconds();
    return ms > 0 ? ms : 0;
}

int MetadataFile::bitrate() const {
    if (!is_valid() || !file_ref_->audioProperties()) return 0;
    return file_ref_->audioProperties()->bitrate();
}

int MetadataFile::sample_rate() const {
    if (!is_valid() || !file_ref_->audioProperties()) return 0;
    return file_ref_->audioProperties()->sampleRate();
}

int MetadataFile::bits_per_sample() const {
    if (!is_valid()) return 0;
    auto* file = file_ref_->file();

    if (auto* flac = dynamic_cast<TagLib::FLAC::File*>(file))
        return flac->audioProperties() ? flac->audioProperties()->bitsPerSample() : 0;
    if (auto* wav = dynamic_cast<TagLib::RIFF::WAV::File*>(file))
        return wav->audioProperties() ? wav->audioProperties()->bitsPerSample() : 0;

    return 0;
}

std::string MetadataFile::codec_description() const {
    if (!is_valid()) return {};
    auto* file = file_ref_->file();
    if (dynamic_cast<TagLib::FLAC::File*>(file)) return "FLAC";
    if (dynamic_cast<TagLib::MPEG::File*>(file)) return "MPEG Audio";
    if (auto* mp4 = dynamic_cast<TagLib::MP4::File*>(file)) {
        if (mp4->audioProperties())
            return mp4->audioProperties()->codec() == TagLib::MP4::Properties::ALAC ? "ALAC" : "AAC";
        return "AAC";
    }
    if (dynamic_cast<TagLib::Vorbis::File*>(file)) return "Vorbis";
    if (dynamic_cast<TagLib::ASF::File*>(file)) return "WMA";
    if (dynamic_cast<TagLib::RIFF::WAV::File*>(file)) return "WAV";
    return {};
}

xune_media_type_t MetadataFile::media_type() const {
    if (!is_valid()) return XUNE_MEDIA_UNKNOWN;
    if (file_ref_->audioProperties()) return XUNE_MEDIA_AUDIO;
    return XUNE_MEDIA_UNKNOWN;
}

// ── Single-Value Tag Helpers ─────────────────────────────────────────────────

static std::string to_std_string(const TagLib::String& s) {
    if (s.isEmpty()) return {};
    return s.to8Bit(true);
}

static TagLib::String to_taglib_string(const std::string& s) {
    return TagLib::String(s, TagLib::String::UTF8);
}

// ── Single-Value Tags ────────────────────────────────────────────────────────

std::string MetadataFile::title() const {
    if (!is_valid() || !file_ref_->tag()) return {};
    return to_std_string(file_ref_->tag()->title());
}

std::string MetadataFile::title_sort() const {
    if (!is_valid()) return {};
    auto it = props_.find("TITLESORT");
    if (it != props_.end() && !it->second.isEmpty())
        return to_std_string(it->second.front());
    return {};
}

std::string MetadataFile::album() const {
    if (!is_valid() || !file_ref_->tag()) return {};
    return to_std_string(file_ref_->tag()->album());
}

std::string MetadataFile::album_sort() const {
    if (!is_valid()) return {};
    auto it = props_.find("ALBUMSORT");
    if (it != props_.end() && !it->second.isEmpty())
        return to_std_string(it->second.front());
    return {};
}

std::string MetadataFile::conductor() const {
    if (!is_valid()) return {};
    auto it = props_.find("CONDUCTOR");
    if (it != props_.end() && !it->second.isEmpty())
        return to_std_string(it->second.front());
    return {};
}

std::string MetadataFile::genre() const {
    if (!is_valid() || !file_ref_->tag()) return {};
    return to_std_string(file_ref_->tag()->genre());
}

uint32_t MetadataFile::track_number() const {
    if (!is_valid() || !file_ref_->tag()) return 0;
    return file_ref_->tag()->track();
}

uint32_t MetadataFile::disc_number() const {
    if (!is_valid()) return 0;
    auto it = props_.find("DISCNUMBER");
    if (it != props_.end() && !it->second.isEmpty()) {
        auto val = it->second.front().toInt();
        return val > 0 ? static_cast<uint32_t>(val) : 0;
    }
    return 0;
}

uint32_t MetadataFile::year() const {
    if (!is_valid() || !file_ref_->tag()) return 0;
    return file_ref_->tag()->year();
}

// ── ID3v2 Multi-Value Splitting ──────────────────────────────────────────────
// MusicBrainz Picard packs multi-value ID3v2 TXXX fields as "/"-separated
// strings (e.g. "Artist1/Artist2", "mbid1/mbid2"). Split them back out.

static std::vector<std::string> split_id3v2_separator(const std::vector<std::string>& values) {
    std::vector<std::string> result;
    for (const auto& v : values) {
        size_t start = 0;
        size_t pos;
        while ((pos = v.find('/', start)) != std::string::npos) {
            auto token = v.substr(start, pos - start);
            if (!token.empty())
                result.push_back(token);
            start = pos + 1;
        }
        auto tail = v.substr(start);
        if (!tail.empty())
            result.push_back(tail);
    }
    return result;
}

// ── Format-Specific Multi-Value Readers ──────────────────────────────────────

std::vector<std::string> MetadataFile::read_xiph_field(const char* field_name) const {
    auto* file = file_ref_->file();
    TagLib::Ogg::XiphComment* xiph = nullptr;

    if (auto* flac = dynamic_cast<TagLib::FLAC::File*>(file))
        xiph = flac->xiphComment();
    else if (auto* vorbis = dynamic_cast<TagLib::Vorbis::File*>(file))
        xiph = vorbis->tag();
    else if (auto* oggflac = dynamic_cast<TagLib::Ogg::FLAC::File*>(file))
        xiph = oggflac->tag();

    if (!xiph) return {};

    auto map = xiph->fieldListMap();
    auto it = map.find(field_name);
    if (it == map.end()) return {};

    std::vector<std::string> result;
    for (const auto& s : it->second)
        result.push_back(to_std_string(s));
    return result;
}

std::vector<std::string> MetadataFile::read_id3v2_txxx(const char* description) const {
    auto* file = file_ref_->file();
    TagLib::ID3v2::Tag* id3v2 = nullptr;

    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file))
        id3v2 = mpeg->ID3v2Tag();

    if (!id3v2) return {};

    auto target = TagLib::String(description, TagLib::String::UTF8);
    auto target_upper = target.upper();

    auto frames = id3v2->frameList("TXXX");
    for (auto* frame : frames) {
        auto* txxx = dynamic_cast<TagLib::ID3v2::UserTextIdentificationFrame*>(frame);
        if (!txxx) continue;

        if (txxx->description() == target || txxx->description().upper() == target_upper) {
            std::vector<std::string> result;
            auto field_list = txxx->fieldList();
            // First field is the description, skip it
            for (unsigned int i = 1; i < field_list.size(); i++)
                result.push_back(to_std_string(field_list[i]));
            return result;
        }
    }
    return {};
}

std::vector<std::string> MetadataFile::read_mp4_text(const char* atom_id) const {
    auto* file = file_ref_->file();
    auto* mp4 = dynamic_cast<TagLib::MP4::File*>(file);
    if (!mp4 || !mp4->tag()) return {};

    auto& items = mp4->tag()->itemMap();
    auto it = items.find(atom_id);
    if (it == items.end()) return {};

    std::vector<std::string> result;
    for (const auto& s : it->second.toStringList())
        result.push_back(to_std_string(s));
    return result;
}

std::vector<std::string> MetadataFile::read_asf_attribute(const char* attr_name) const {
    auto* file = file_ref_->file();
    auto* asf = dynamic_cast<TagLib::ASF::File*>(file);
    if (!asf || !asf->tag()) return {};

    auto& map = asf->tag()->attributeListMap();

    auto it = map.find(attr_name);
    if (it != map.end()) {
        std::vector<std::string> result;
        for (const auto& attr : it->second)
            result.push_back(to_std_string(attr.toString()));
        return result;
    }

    // Case-insensitive fallback (different taggers use different cases)
    TagLib::String target(attr_name, TagLib::String::UTF8);
    auto target_upper = target.upper();
    for (auto ci = map.begin(); ci != map.end(); ++ci) {
        if (ci->first.upper() == target_upper) {
            std::vector<std::string> result;
            for (const auto& attr : ci->second)
                result.push_back(to_std_string(attr.toString()));
            return result;
        }
    }

    return {};
}

// ── Multi-Value Tags ─────────────────────────────────────────────────────────

std::vector<std::string> MetadataFile::artists() const {
    if (!is_valid()) return {};

    auto result = read_xiph_field("ARTISTS");
    if (!result.empty()) return result;

    result = read_id3v2_txxx("ARTISTS");
    if (!result.empty()) return split_id3v2_separator(result);

    result = read_mp4_text("----:com.apple.iTunes:ARTISTS");
    if (!result.empty()) return result;

    result = read_asf_attribute("WM/ARTISTS");
    if (!result.empty()) return result;

    return {};
}

std::string MetadataFile::artist_display() const {
    if (!is_valid()) return {};

    // Vorbis: return first ARTIST entry, not TagLib's " / " join
    auto xiph = read_xiph_field("ARTIST");
    if (!xiph.empty()) return xiph.front();

    // Other formats: tag()->artist() returns the display string directly
    if (file_ref_->tag())
        return to_std_string(file_ref_->tag()->artist());

    return {};
}

std::vector<std::string> MetadataFile::album_artists() const {
    if (!is_valid()) return {};
    auto it = props_.find("ALBUMARTIST");
    if (it != props_.end() && !it->second.isEmpty()) {
        std::vector<std::string> result;
        for (const auto& s : it->second)
            result.push_back(to_std_string(s));
        return result;
    }
    return {};
}

std::vector<std::string> MetadataFile::composers() const {
    if (!is_valid()) return {};
    auto it = props_.find("COMPOSER");
    if (it != props_.end() && !it->second.isEmpty()) {
        std::vector<std::string> result;
        for (const auto& s : it->second)
            result.push_back(to_std_string(s));
        return result;
    }
    return {};
}

std::string MetadataFile::artist_sort() const {
    if (!is_valid()) return {};
    auto it = props_.find("ARTISTSORT");
    if (it != props_.end() && !it->second.isEmpty())
        return to_std_string(it->second.front());
    return {};
}

std::string MetadataFile::album_artist_sort() const {
    if (!is_valid()) return {};
    auto it = props_.find("ALBUMARTISTSORT");
    if (it != props_.end() && !it->second.isEmpty())
        return to_std_string(it->second.front());
    return {};
}

// ── MusicBrainz IDs ─────────────────────────────────────────────────────────

std::vector<std::string> MetadataFile::mb_artist_ids() const {
    if (!is_valid()) return {};

    auto result = read_xiph_field("MUSICBRAINZ_ARTISTID");
    if (!result.empty()) return result;

    result = read_id3v2_txxx("MusicBrainz Artist Id");
    if (!result.empty()) return split_id3v2_separator(result);

    result = read_mp4_text("----:com.apple.iTunes:MusicBrainz Artist Id");
    if (!result.empty()) return result;

    result = read_asf_attribute("MusicBrainz/Artist Id");
    if (!result.empty()) return result;

    return {};
}

std::vector<std::string> MetadataFile::mb_album_artist_ids() const {
    if (!is_valid()) return {};

    auto result = read_xiph_field("MUSICBRAINZ_ALBUMARTISTID");
    if (!result.empty()) return result;

    result = read_id3v2_txxx("MusicBrainz Album Artist Id");
    if (!result.empty()) return split_id3v2_separator(result);

    result = read_mp4_text("----:com.apple.iTunes:MusicBrainz Album Artist Id");
    if (!result.empty()) return result;

    result = read_asf_attribute("MusicBrainz/Album Artist Id");
    if (!result.empty()) return result;

    return {};
}

#define READ_SINGLE_CUSTOM(xiph_name, id3v2_desc, asf_name, mp4_atom) \
    do { \
        auto result = read_xiph_field(xiph_name); \
        if (!result.empty()) return result.front(); \
        result = read_id3v2_txxx(id3v2_desc); \
        if (!result.empty()) return result.front(); \
        result = read_mp4_text(mp4_atom); \
        if (!result.empty()) return result.front(); \
        result = read_asf_attribute(asf_name); \
        if (!result.empty()) return result.front(); \
        return {}; \
    } while(0)

std::string MetadataFile::mb_recording_id() const {
    if (!is_valid()) return {};

    // Vorbis
    auto result = read_xiph_field("MUSICBRAINZ_TRACKID");
    if (!result.empty()) return result.front();

    // ID3v2: Picard stores recording ID in UFID frame; some taggers use TXXX instead
    auto* file = file_ref_->file();
    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file)) {
        if (auto* id3v2 = mpeg->ID3v2Tag()) {
            auto frames = id3v2->frameList("UFID");
            for (auto* frame : frames) {
                auto* ufid = dynamic_cast<TagLib::ID3v2::UniqueFileIdentifierFrame*>(frame);
                if (ufid && ufid->owner() == "http://musicbrainz.org") {
                    auto id = ufid->identifier();
                    if (!id.isEmpty())
                        return std::string(id.data(), id.size());
                }
            }
        }
    }

    auto txxx_result = read_id3v2_txxx("MusicBrainz Track Id");
    if (!txxx_result.empty()) return txxx_result.front();

    // MP4
    result = read_mp4_text("----:com.apple.iTunes:MusicBrainz Track Id");
    if (!result.empty()) return result.front();

    // ASF
    result = read_asf_attribute("MusicBrainz/Track Id");
    if (!result.empty()) return result.front();

    return {};
}

std::string MetadataFile::mb_release_track_id() const {
    if (!is_valid()) return {};
    READ_SINGLE_CUSTOM("MUSICBRAINZ_RELEASETRACKID", "MusicBrainz Release Track Id",
                       "MusicBrainz/Release Track Id", "----:com.apple.iTunes:MusicBrainz Release Track Id");
}

std::string MetadataFile::mb_release_id() const {
    if (!is_valid()) return {};
    READ_SINGLE_CUSTOM("MUSICBRAINZ_ALBUMID", "MusicBrainz Album Id",
                       "MusicBrainz/Album Id", "----:com.apple.iTunes:MusicBrainz Album Id");
}

std::string MetadataFile::mb_release_group_id() const {
    if (!is_valid()) return {};
    READ_SINGLE_CUSTOM("MUSICBRAINZ_RELEASEGROUPID", "MusicBrainz Release Group Id",
                       "MusicBrainz/Release Group Id", "----:com.apple.iTunes:MusicBrainz Release Group Id");
}

std::string MetadataFile::acoustid_fingerprint() const {
    if (!is_valid()) return {};

    // Vorbis
    auto result = read_xiph_field("ACOUSTID_FINGERPRINT");
    if (!result.empty()) return result.front();

    // ID3v2: try Picard convention ("Acoustid Fingerprint") and underscore convention ("ACOUSTID_FINGERPRINT")
    result = read_id3v2_txxx("Acoustid Fingerprint");
    if (!result.empty()) return result.front();
    result = read_id3v2_txxx("ACOUSTID_FINGERPRINT");
    if (!result.empty()) return result.front();

    // MP4
    result = read_mp4_text("----:com.apple.iTunes:Acoustid Fingerprint");
    if (!result.empty()) return result.front();

    // ASF
    result = read_asf_attribute("Acoustid/Fingerprint");
    if (!result.empty()) return result.front();

    return {};
}

#undef READ_SINGLE_CUSTOM

// ── ReplayGain ───────────────────────────────────────────────────────────────

static double parse_replaygain_value(const std::string& s) {
    if (s.empty()) return std::nan("");
    try {
        // Strip " dB" suffix if present
        auto val = s;
        auto pos = val.find(" dB");
        if (pos != std::string::npos) val = val.substr(0, pos);
        return std::stod(val);
    } catch (...) {
        return std::nan("");
    }
}

double MetadataFile::read_replaygain(const char* props_key,
                                      const char* asf_vorbis_key,
                                      const char* asf_wm_key) const {
    auto it = props_.find(props_key);
    if (it != props_.end() && !it->second.isEmpty())
        return parse_replaygain_value(to_std_string(it->second.front()));

    // ASF: try Vorbis-style name first, then WM-style
    auto result = read_asf_attribute(asf_vorbis_key);
    if (!result.empty())
        return parse_replaygain_value(result.front());

    if (asf_wm_key) {
        result = read_asf_attribute(asf_wm_key);
        if (!result.empty())
            return parse_replaygain_value(result.front());
    }

    return std::nan("");
}

double MetadataFile::replaygain_track_gain() const {
    if (!is_valid()) return std::nan("");
    return read_replaygain("REPLAYGAIN_TRACK_GAIN",
                           "REPLAYGAIN_TRACK_GAIN", "ReplayGain/Track");
}

double MetadataFile::replaygain_track_peak() const {
    if (!is_valid()) return std::nan("");
    return read_replaygain("REPLAYGAIN_TRACK_PEAK",
                           "REPLAYGAIN_TRACK_PEAK", "ReplayGain/Track Peak");
}

double MetadataFile::replaygain_album_gain() const {
    if (!is_valid()) return std::nan("");
    return read_replaygain("REPLAYGAIN_ALBUM_GAIN",
                           "REPLAYGAIN_ALBUM_GAIN", "ReplayGain/Album");
}

double MetadataFile::replaygain_album_peak() const {
    if (!is_valid()) return std::nan("");
    return read_replaygain("REPLAYGAIN_ALBUM_PEAK",
                           "REPLAYGAIN_ALBUM_PEAK", "ReplayGain/Album Peak");
}

// ── Artwork ──────────────────────────────────────────────────────────────────

bool MetadataFile::has_picture() const {
    if (!is_valid()) return false;
    auto* file = file_ref_->file();

    if (auto* flac = dynamic_cast<TagLib::FLAC::File*>(file))
        return !flac->pictureList().isEmpty();

    if (auto* ogg = dynamic_cast<TagLib::Ogg::Vorbis::File*>(file)) {
        if (auto* xiph = ogg->tag())
            return !xiph->pictureList().isEmpty();
    }

    if (auto* oggflac = dynamic_cast<TagLib::Ogg::FLAC::File*>(file)) {
        if (auto* xiph = oggflac->tag())
            return !xiph->pictureList().isEmpty();
    }

    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file)) {
        if (auto* id3v2 = mpeg->ID3v2Tag())
            return !id3v2->frameList("APIC").isEmpty();
    }

    if (auto* mp4 = dynamic_cast<TagLib::MP4::File*>(file)) {
        if (mp4->tag()) {
            auto it = mp4->tag()->itemMap().find("covr");
            return it != mp4->tag()->itemMap().end();
        }
    }

    if (auto* asf = dynamic_cast<TagLib::ASF::File*>(file)) {
        if (asf->tag()) {
            auto it = asf->tag()->attributeListMap().find("WM/Picture");
            return it != asf->tag()->attributeListMap().end() && !it->second.isEmpty();
        }
    }

    return false;
}

static PictureData extract_flac_picture(const TagLib::List<TagLib::FLAC::Picture*>& pics) {
    if (pics.isEmpty()) return {};
    auto* pic = pics.front();
    auto data = pic->data();
    return {
        std::vector<uint8_t>(data.begin(), data.end()),
        to_std_string(pic->mimeType())
    };
}

PictureData MetadataFile::picture() const {
    if (!is_valid()) return {};
    auto* file = file_ref_->file();

    if (auto* flac = dynamic_cast<TagLib::FLAC::File*>(file))
        return extract_flac_picture(flac->pictureList());

    if (auto* ogg = dynamic_cast<TagLib::Ogg::Vorbis::File*>(file)) {
        if (auto* xiph = ogg->tag())
            return extract_flac_picture(xiph->pictureList());
    }

    if (auto* oggflac = dynamic_cast<TagLib::Ogg::FLAC::File*>(file)) {
        if (auto* xiph = oggflac->tag())
            return extract_flac_picture(xiph->pictureList());
    }

    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file)) {
        if (auto* id3v2 = mpeg->ID3v2Tag()) {
            auto frames = id3v2->frameList("APIC");
            if (!frames.isEmpty()) {
                auto* pic = dynamic_cast<TagLib::ID3v2::AttachedPictureFrame*>(frames.front());
                if (pic) {
                    auto data = pic->picture();
                    return {
                        std::vector<uint8_t>(data.begin(), data.end()),
                        to_std_string(pic->mimeType())
                    };
                }
            }
        }
    }

    if (auto* mp4 = dynamic_cast<TagLib::MP4::File*>(file)) {
        if (mp4->tag()) {
            auto it = mp4->tag()->itemMap().find("covr");
            if (it != mp4->tag()->itemMap().end()) {
                auto covers = it->second.toCoverArtList();
                if (!covers.isEmpty()) {
                    auto data = covers.front().data();
                    auto fmt = covers.front().format();
                    std::string mime = "image/jpeg";
                    if (fmt == TagLib::MP4::CoverArt::PNG) mime = "image/png";
                    return {
                        std::vector<uint8_t>(data.begin(), data.end()),
                        mime
                    };
                }
            }
        }
    }

    if (auto* asf = dynamic_cast<TagLib::ASF::File*>(file)) {
        if (asf->tag()) {
            auto it = asf->tag()->attributeListMap().find("WM/Picture");
            if (it != asf->tag()->attributeListMap().end() && !it->second.isEmpty()) {
                auto pic = it->second.front().toPicture();
                auto data = pic.picture();
                if (!data.isEmpty()) {
                    return {
                        std::vector<uint8_t>(data.begin(), data.end()),
                        to_std_string(pic.mimeType())
                    };
                }
            }
        }
    }

    return {};
}

// ── Release Date ─────────────────────────────────────────────────────────────

std::string MetadataFile::release_date() const {
    if (!is_valid()) return {};

    // Try format-specific date fields
    auto result = read_xiph_field("RELEASEDATE");
    if (!result.empty()) return result.front();

    result = read_xiph_field("DATE");
    if (!result.empty()) return result.front();

    // ID3v2: TDRC frame
    auto* file = file_ref_->file();
    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file)) {
        if (auto* id3v2 = mpeg->ID3v2Tag()) {
            auto frames = id3v2->frameList("TDRC");
            if (!frames.isEmpty())
                return to_std_string(frames.front()->toString());
        }
    }

    // ASF
    result = read_asf_attribute("WM/OriginalReleaseTime");
    if (!result.empty()) return result.front();
    result = read_asf_attribute("WM/Year");
    if (!result.empty()) return result.front();

    return {};
}

// ── Writing ──────────────────────────────────────────────────────────────────

void MetadataFile::set_title(const std::string& value) {
    if (!is_valid()) return;
    if (value.empty()) {
        props_.erase("TITLE");
    } else {
        props_["TITLE"] = TagLib::StringList(to_taglib_string(value));
    }
    flush_properties();
}

void MetadataFile::set_album(const std::string& value) {
    if (!is_valid()) return;
    if (value.empty()) {
        props_.erase("ALBUM");
    } else {
        props_["ALBUM"] = TagLib::StringList(to_taglib_string(value));
    }
    flush_properties();
}

void MetadataFile::set_conductor(const std::string& value) {
    if (!is_valid()) return;
    if (value.empty()) {
        props_.erase("CONDUCTOR");
    } else {
        props_["CONDUCTOR"] = TagLib::StringList(to_taglib_string(value));
    }
    flush_properties();
}

void MetadataFile::set_genre(const std::string& value) {
    if (!is_valid()) return;
    if (value.empty()) {
        props_.erase("GENRE");
    } else {
        props_["GENRE"] = TagLib::StringList(to_taglib_string(value));
    }
    flush_properties();
}

void MetadataFile::set_track_number(uint32_t value) {
    if (!is_valid()) return;
    if (value == 0) {
        props_.erase("TRACKNUMBER");
    } else {
        props_["TRACKNUMBER"] = TagLib::StringList(TagLib::String::number(value));
    }
    flush_properties();
}

void MetadataFile::set_disc_number(uint32_t value) {
    if (!is_valid()) return;
    if (value == 0) {
        props_.erase("DISCNUMBER");
    } else {
        props_["DISCNUMBER"] = TagLib::StringList(TagLib::String::number(value));
    }
    flush_properties();
}

void MetadataFile::set_year(uint32_t value) {
    if (!is_valid()) return;
    if (value == 0) {
        props_.erase("DATE");
    } else {
        props_["DATE"] = TagLib::StringList(TagLib::String::number(value));
    }
    flush_properties();
}

// ── Format-Specific Multi-Value Writers ──────────────────────────────────────

void MetadataFile::write_xiph_field(const char* field_name, const std::vector<std::string>& values) {
    auto* file = file_ref_->file();
    TagLib::Ogg::XiphComment* xiph = nullptr;

    if (auto* flac = dynamic_cast<TagLib::FLAC::File*>(file))
        xiph = flac->xiphComment(true);
    else if (auto* vorbis = dynamic_cast<TagLib::Vorbis::File*>(file))
        xiph = vorbis->tag();
    else if (auto* oggflac = dynamic_cast<TagLib::Ogg::FLAC::File*>(file))
        xiph = oggflac->tag();

    if (!xiph) return;

    xiph->removeFields(field_name);
    for (const auto& v : values)
        xiph->addField(field_name, to_taglib_string(v), false);
}

void MetadataFile::write_id3v2_txxx(const char* description, const std::vector<std::string>& values) {
    auto* file = file_ref_->file();
    TagLib::ID3v2::Tag* id3v2 = nullptr;

    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file))
        id3v2 = mpeg->ID3v2Tag(true);

    if (!id3v2) return;

    TagLib::ID3v2::Frame* existing = nullptr;
    for (auto* frame : id3v2->frameList("TXXX")) {
        auto* txxx = dynamic_cast<TagLib::ID3v2::UserTextIdentificationFrame*>(frame);
        if (txxx && txxx->description() == TagLib::String(description, TagLib::String::UTF8)) {
            existing = txxx;
            break;
        }
    }
    if (existing) id3v2->removeFrame(existing);

    if (values.empty()) return;

    auto* frame = new TagLib::ID3v2::UserTextIdentificationFrame(TagLib::String::UTF8);
    frame->setDescription(TagLib::String(description, TagLib::String::UTF8));
    TagLib::StringList sl;
    for (const auto& v : values)
        sl.append(to_taglib_string(v));
    frame->setText(sl);
    id3v2->addFrame(frame);
}

void MetadataFile::write_mp4_text(const char* atom_id, const std::vector<std::string>& values) {
    auto* file = file_ref_->file();
    auto* mp4 = dynamic_cast<TagLib::MP4::File*>(file);
    if (!mp4 || !mp4->tag()) return;

    if (values.empty()) {
        mp4->tag()->removeItem(atom_id);
        return;
    }

    TagLib::StringList sl;
    for (const auto& v : values)
        sl.append(to_taglib_string(v));
    mp4->tag()->setItem(atom_id, TagLib::MP4::Item(sl));
}

void MetadataFile::write_asf_attribute(const char* attr_name, const std::vector<std::string>& values) {
    auto* file = file_ref_->file();
    auto* asf = dynamic_cast<TagLib::ASF::File*>(file);
    if (!asf || !asf->tag()) return;

    asf->tag()->removeItem(attr_name);
    for (const auto& v : values)
        asf->tag()->addAttribute(attr_name, TagLib::ASF::Attribute(to_taglib_string(v)));
}

void MetadataFile::write_custom_field(const char* xiph_name, const char* id3v2_desc,
                                      const char* asf_name, const char* mp4_atom,
                                      const std::vector<std::string>& values) {
    write_xiph_field(xiph_name, values);
    write_id3v2_txxx(id3v2_desc, values);
    write_mp4_text(mp4_atom, values);
    write_asf_attribute(asf_name, values);
}

void MetadataFile::write_custom_field(const char* xiph_name, const char* id3v2_desc,
                                      const char* asf_name, const char* mp4_atom,
                                      const std::string& value) {
    std::vector<std::string> values;
    if (!value.empty()) values.push_back(value);
    write_custom_field(xiph_name, id3v2_desc, asf_name, mp4_atom, values);
}

void MetadataFile::set_artists(const std::vector<std::string>& values) {
    if (!is_valid()) return;

    // Vorbis: write individual ARTIST entries (the standard multi-value mechanism)
    write_xiph_field("ARTIST", values);

    // ID3v2/MP4/ASF: ARTIST property is the display string, ARTISTS is the multi-value field
    if (!values.empty()) {
        std::string display;
        for (size_t i = 0; i < values.size(); i++) {
            if (i > 0) display += "; ";
            display += values[i];
        }
        props_["ARTIST"] = TagLib::StringList(to_taglib_string(display));
    } else {
        props_.erase("ARTIST");
    }
    flush_properties();

    write_id3v2_txxx("ARTISTS", values);
    write_mp4_text("----:com.apple.iTunes:ARTISTS", values);
    write_asf_attribute("WM/ARTISTS", values);
}

void MetadataFile::set_album_artists(const std::vector<std::string>& values) {
    if (!is_valid()) return;
    TagLib::StringList sl;
    for (const auto& v : values)
        sl.append(to_taglib_string(v));
    props_["ALBUMARTIST"] = sl;
    flush_properties();
}

void MetadataFile::set_composers(const std::vector<std::string>& values) {
    if (!is_valid()) return;
    TagLib::StringList sl;
    for (const auto& v : values)
        sl.append(to_taglib_string(v));
    props_["COMPOSER"] = sl;
    flush_properties();
}

void MetadataFile::set_mb_artist_ids(const std::vector<std::string>& values) {
    write_custom_field("MUSICBRAINZ_ARTISTID", "MusicBrainz Artist Id",
                       "MusicBrainz/Artist Id", "----:com.apple.iTunes:MusicBrainz Artist Id", values);
}

void MetadataFile::set_mb_album_artist_ids(const std::vector<std::string>& values) {
    write_custom_field("MUSICBRAINZ_ALBUMARTISTID", "MusicBrainz Album Artist Id",
                       "MusicBrainz/Album Artist Id", "----:com.apple.iTunes:MusicBrainz Album Artist Id", values);
}

void MetadataFile::set_mb_recording_id(const std::string& value) {
    write_custom_field("MUSICBRAINZ_TRACKID", "MusicBrainz Track Id",
                       "MusicBrainz/Track Id", "----:com.apple.iTunes:MusicBrainz Track Id", value);

    // ID3v2: also write/update UFID frame (Picard's primary storage for recording ID)
    auto* file = file_ref_->file();
    if (auto* mpeg = dynamic_cast<TagLib::MPEG::File*>(file)) {
        auto* id3v2 = mpeg->ID3v2Tag(true);
        if (!id3v2) return;

        TagLib::ID3v2::Frame* existing_ufid = nullptr;
        for (auto* frame : id3v2->frameList("UFID")) {
            auto* ufid = dynamic_cast<TagLib::ID3v2::UniqueFileIdentifierFrame*>(frame);
            if (ufid && ufid->owner() == "http://musicbrainz.org") {
                existing_ufid = ufid;
                break;
            }
        }
        if (existing_ufid) id3v2->removeFrame(existing_ufid);

        if (!value.empty()) {
            auto* ufid = new TagLib::ID3v2::UniqueFileIdentifierFrame(
                "http://musicbrainz.org",
                TagLib::ByteVector(value.c_str(), value.size()));
            id3v2->addFrame(ufid);
        }
    }
}

void MetadataFile::set_mb_release_track_id(const std::string& value) {
    write_custom_field("MUSICBRAINZ_RELEASETRACKID", "MusicBrainz Release Track Id",
                       "MusicBrainz/Release Track Id", "----:com.apple.iTunes:MusicBrainz Release Track Id", value);
}

void MetadataFile::set_mb_release_id(const std::string& value) {
    write_custom_field("MUSICBRAINZ_ALBUMID", "MusicBrainz Album Id",
                       "MusicBrainz/Album Id", "----:com.apple.iTunes:MusicBrainz Album Id", value);
}

void MetadataFile::set_mb_release_group_id(const std::string& value) {
    write_custom_field("MUSICBRAINZ_RELEASEGROUPID", "MusicBrainz Release Group Id",
                       "MusicBrainz/Release Group Id", "----:com.apple.iTunes:MusicBrainz Release Group Id", value);
}

void MetadataFile::set_acoustid_fingerprint(const std::string& value) {
    write_custom_field("ACOUSTID_FINGERPRINT", "Acoustid Fingerprint",
                       "Acoustid/Fingerprint", "----:com.apple.iTunes:Acoustid Fingerprint", value);
}

void MetadataFile::write_replaygain(const char* props_key,
                                     const char* asf_vorbis_key,
                                     const char* asf_wm_key,
                                     const char* formatted_value) {
    props_[props_key] = TagLib::StringList(TagLib::String(formatted_value, TagLib::String::UTF8));
    flush_properties();

    // ASF: write both naming conventions for interoperability
    write_asf_attribute(asf_vorbis_key, {formatted_value});
    if (asf_wm_key)
        write_asf_attribute(asf_wm_key, {formatted_value});
}

void MetadataFile::clear_replaygain(const char* props_key,
                                     const char* asf_vorbis_key,
                                     const char* asf_wm_key) {
    props_.erase(props_key);
    flush_properties();
    write_asf_attribute(asf_vorbis_key, {});
    if (asf_wm_key)
        write_asf_attribute(asf_wm_key, {});
}

void MetadataFile::set_replaygain_track_gain(double value) {
    if (!is_valid()) return;
    if (std::isnan(value)) {
        clear_replaygain("REPLAYGAIN_TRACK_GAIN",
                         "REPLAYGAIN_TRACK_GAIN", "ReplayGain/Track");
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f dB", value);
    write_replaygain("REPLAYGAIN_TRACK_GAIN",
                     "REPLAYGAIN_TRACK_GAIN", "ReplayGain/Track", buf);
}

void MetadataFile::set_replaygain_track_peak(double value) {
    if (!is_valid()) return;
    if (std::isnan(value)) {
        clear_replaygain("REPLAYGAIN_TRACK_PEAK",
                         "REPLAYGAIN_TRACK_PEAK", "ReplayGain/Track Peak");
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6f", value);
    write_replaygain("REPLAYGAIN_TRACK_PEAK",
                     "REPLAYGAIN_TRACK_PEAK", "ReplayGain/Track Peak", buf);
}

void MetadataFile::set_replaygain_album_gain(double value) {
    if (!is_valid()) return;
    if (std::isnan(value)) {
        clear_replaygain("REPLAYGAIN_ALBUM_GAIN",
                         "REPLAYGAIN_ALBUM_GAIN", "ReplayGain/Album");
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f dB", value);
    write_replaygain("REPLAYGAIN_ALBUM_GAIN",
                     "REPLAYGAIN_ALBUM_GAIN", "ReplayGain/Album", buf);
}

void MetadataFile::set_replaygain_album_peak(double value) {
    if (!is_valid()) return;
    if (std::isnan(value)) {
        clear_replaygain("REPLAYGAIN_ALBUM_PEAK",
                         "REPLAYGAIN_ALBUM_PEAK", "ReplayGain/Album Peak");
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6f", value);
    write_replaygain("REPLAYGAIN_ALBUM_PEAK",
                     "REPLAYGAIN_ALBUM_PEAK", "ReplayGain/Album Peak", buf);
}

bool MetadataFile::save() {
    if (!is_valid()) return false;
    return file_ref_->save();
}

} // namespace xune
