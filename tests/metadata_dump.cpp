#include <xune_audio/xune_metadata.h>
#include <cstdio>
#include <cmath>
#include <string>

static std::string js(const char* s) {
    if (!s) return "";
    std::string out;
    for (const char* p = s; *p; p++) {
        switch (*p) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += *p;
        }
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filepath>\n", argv[0]);
        return 1;
    }

    xune_meta_handle_t handle = nullptr;
    auto err = xune_meta_open(argv[1], &handle);
    if (err != XUNE_META_OK) {
        printf("{\"error\": %d}\n", err);
        return 1;
    }

    printf("{\n");
    printf("  \"duration_ms\": %d,\n", xune_meta_duration_ms(handle));
    printf("  \"bitrate\": %d,\n", xune_meta_bitrate(handle));
    printf("  \"sample_rate\": %d,\n", xune_meta_sample_rate(handle));
    printf("  \"bits_per_sample\": %d,\n", xune_meta_bits_per_sample(handle));
    printf("  \"title\": \"%s\",\n", js(xune_meta_title(handle)).c_str());
    printf("  \"title_sort\": \"%s\",\n", js(xune_meta_title_sort(handle)).c_str());
    printf("  \"album\": \"%s\",\n", js(xune_meta_album(handle)).c_str());
    printf("  \"album_sort\": \"%s\",\n", js(xune_meta_album_sort(handle)).c_str());
    printf("  \"genre\": \"%s\",\n", js(xune_meta_genre(handle)).c_str());
    printf("  \"conductor\": \"%s\",\n", js(xune_meta_conductor(handle)).c_str());
    printf("  \"track_number\": %u,\n", xune_meta_track_number(handle));
    printf("  \"disc_number\": %u,\n", xune_meta_disc_number(handle));
    printf("  \"year\": %u,\n", xune_meta_year(handle));

    // Artists
    printf("  \"artist_display\": \"%s\",\n", js(xune_meta_artist_display(handle)).c_str());
    printf("  \"artist_sort\": \"%s\",\n", js(xune_meta_artist_sort(handle)).c_str());
    printf("  \"album_artist_sort\": \"%s\",\n", js(xune_meta_album_artist_sort(handle)).c_str());

    printf("  \"artists\": [");
    int ac = xune_meta_artist_count(handle);
    for (int i = 0; i < ac; i++)
        printf("%s\"%s\"", i ? ", " : "", js(xune_meta_artist_at(handle, i)).c_str());
    printf("],\n");

    printf("  \"album_artists\": [");
    int aac = xune_meta_album_artist_count(handle);
    for (int i = 0; i < aac; i++)
        printf("%s\"%s\"", i ? ", " : "", js(xune_meta_album_artist_at(handle, i)).c_str());
    printf("],\n");

    printf("  \"composers\": [");
    int cc = xune_meta_composer_count(handle);
    for (int i = 0; i < cc; i++)
        printf("%s\"%s\"", i ? ", " : "", js(xune_meta_composer_at(handle, i)).c_str());
    printf("],\n");

    // MusicBrainz IDs
    printf("  \"mb_recording_id\": \"%s\",\n", js(xune_meta_mb_recording_id(handle)).c_str());
    printf("  \"mb_release_track_id\": \"%s\",\n", js(xune_meta_mb_release_track_id(handle)).c_str());
    printf("  \"mb_release_id\": \"%s\",\n", js(xune_meta_mb_release_id(handle)).c_str());
    printf("  \"mb_release_group_id\": \"%s\",\n", js(xune_meta_mb_release_group_id(handle)).c_str());
    printf("  \"acoustid_fingerprint\": \"%s\",\n", js(xune_meta_acoustid_fingerprint(handle)).c_str());

    printf("  \"mb_artist_ids\": [");
    int maic = xune_meta_mb_artist_id_count(handle);
    for (int i = 0; i < maic; i++)
        printf("%s\"%s\"", i ? ", " : "", js(xune_meta_mb_artist_id_at(handle, i)).c_str());
    printf("],\n");

    printf("  \"mb_album_artist_ids\": [");
    int maaic = xune_meta_mb_album_artist_id_count(handle);
    for (int i = 0; i < maaic; i++)
        printf("%s\"%s\"", i ? ", " : "", js(xune_meta_mb_album_artist_id_at(handle, i)).c_str());
    printf("],\n");

    // ReplayGain
    double tg = xune_meta_replaygain_track_gain(handle);
    double tp = xune_meta_replaygain_track_peak(handle);
    double ag = xune_meta_replaygain_album_gain(handle);
    double ap = xune_meta_replaygain_album_peak(handle);
    printf("  \"replaygain_track_gain\": %s,\n", std::isnan(tg) ? "null" : ([&]{ static char b[32]; snprintf(b, 32, "%.6f", tg); return b; })());
    printf("  \"replaygain_track_peak\": %s,\n", std::isnan(tp) ? "null" : ([&]{ static char b[32]; snprintf(b, 32, "%.6f", tp); return b; })());
    printf("  \"replaygain_album_gain\": %s,\n", std::isnan(ag) ? "null" : ([&]{ static char b[32]; snprintf(b, 32, "%.6f", ag); return b; })());
    printf("  \"replaygain_album_peak\": %s,\n", std::isnan(ap) ? "null" : ([&]{ static char b[32]; snprintf(b, 32, "%.6f", ap); return b; })());

    // Artwork
    int pic_size = 0;
    xune_meta_picture_data(handle, &pic_size);
    printf("  \"has_picture\": %s,\n", xune_meta_has_picture(handle) ? "true" : "false");
    printf("  \"picture_size\": %d,\n", pic_size);
    printf("  \"picture_mime\": \"%s\",\n", js(xune_meta_picture_mime(handle)).c_str());

    // Release date
    printf("  \"release_date\": \"%s\"\n", js(xune_meta_release_date(handle)).c_str());

    printf("}\n");

    xune_meta_close(handle);
    return 0;
}
