#include <xune_audio/xune_metadata.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef TEST_FILE_PATH
#error "TEST_FILE_PATH must be defined"
#endif

#define S(x) ((x) ? (x) : "(null)")

static void print_double(const char* label, double val, const char* unit) {
    if (std::isnan(val))
        printf("%s: NaN\n", label);
    else
        printf("%s: %.2f%s\n", label, val, unit);
}

static void print_peak(const char* label, double val) {
    if (std::isnan(val))
        printf("%s: NaN\n", label);
    else
        printf("%s: %.6f\n", label, val);
}

int main() {
    xune_meta_handle_t handle = nullptr;
    auto err = xune_meta_open(TEST_FILE_PATH, &handle);
    if (err != XUNE_META_OK) {
        printf("[FAIL] xune_meta_open returned %d\n", err);
        return 1;
    }

    printf("=== File Properties ===\n");
    printf("Duration:    %d ms\n", xune_meta_duration_ms(handle));
    printf("Bitrate:     %d kbps\n", xune_meta_bitrate(handle));
    printf("Sample rate: %d Hz\n", xune_meta_sample_rate(handle));
    printf("Bit depth:   %d\n", xune_meta_bits_per_sample(handle));
    printf("Codec:       %s\n", S(xune_meta_codec_description(handle)));

    printf("\n=== Single-Value Tags ===\n");
    printf("Title:       %s\n", S(xune_meta_title(handle)));
    printf("Title sort:  %s\n", S(xune_meta_title_sort(handle)));
    printf("Album:       %s\n", S(xune_meta_album(handle)));
    printf("Album sort:  %s\n", S(xune_meta_album_sort(handle)));
    printf("Genre:       %s\n", S(xune_meta_genre(handle)));
    printf("Track:       %u\n", xune_meta_track_number(handle));
    printf("Disc:        %u\n", xune_meta_disc_number(handle));
    printf("Year:        %u\n", xune_meta_year(handle));

    printf("\n=== Multi-Value Artists ===\n");
    int artist_count = xune_meta_artist_count(handle);
    printf("Artist count: %d\n", artist_count);
    for (int i = 0; i < artist_count; i++)
        printf("  Artist[%d]: '%s'\n", i, xune_meta_artist_at(handle, i));
    printf("Artist display: '%s'\n", S(xune_meta_artist_display(handle)));
    printf("Artist sort: '%s'\n", S(xune_meta_artist_sort(handle)));

    int aa_count = xune_meta_album_artist_count(handle);
    printf("Album artist count: %d\n", aa_count);
    for (int i = 0; i < aa_count; i++)
        printf("  AlbumArtist[%d]: '%s'\n", i, xune_meta_album_artist_at(handle, i));
    printf("Album artist sort: '%s'\n", S(xune_meta_album_artist_sort(handle)));

    printf("\n=== MusicBrainz IDs ===\n");
    int mbaid_count = xune_meta_mb_artist_id_count(handle);
    printf("MB Artist ID count: %d\n", mbaid_count);
    for (int i = 0; i < mbaid_count; i++)
        printf("  MB ArtistID[%d]: '%s'\n", i, xune_meta_mb_artist_id_at(handle, i));

    int mbaaid_count = xune_meta_mb_album_artist_id_count(handle);
    printf("MB Album Artist ID count: %d\n", mbaaid_count);
    for (int i = 0; i < mbaaid_count; i++)
        printf("  MB AlbumArtistID[%d]: '%s'\n", i, xune_meta_mb_album_artist_id_at(handle, i));

    printf("MB Recording ID:     %s\n", S(xune_meta_mb_recording_id(handle)));
    printf("MB Release Track ID: %s\n", S(xune_meta_mb_release_track_id(handle)));
    printf("MB Release ID:       %s\n", S(xune_meta_mb_release_id(handle)));
    printf("MB Release Group ID: %s\n", S(xune_meta_mb_release_group_id(handle)));
    printf("AcoustID FP:         %s\n", xune_meta_acoustid_fingerprint(handle) ? "present" : "(null)");

    printf("\n=== ReplayGain ===\n");
    double tg = xune_meta_replaygain_track_gain(handle);
    double tp = xune_meta_replaygain_track_peak(handle);
    double ag = xune_meta_replaygain_album_gain(handle);
    double ap = xune_meta_replaygain_album_peak(handle);
    print_double("Track gain", tg, " dB");
    print_peak("Track peak", tp);
    print_double("Album gain", ag, " dB");
    print_peak("Album peak", ap);

    printf("\n=== Artwork ===\n");
    printf("Has picture: %d\n", xune_meta_has_picture(handle));
    int pic_size = 0;
    auto* pic_data = xune_meta_picture_data(handle, &pic_size);
    printf("Picture size: %d bytes\n", pic_size);
    printf("Picture MIME: %s\n", S(xune_meta_picture_mime(handle)));

    printf("\n=== Release Date ===\n");
    printf("Release date: %s\n", S(xune_meta_release_date(handle)));

    // Validation
    printf("\n=== VALIDATION ===\n");
    int pass = 0, fail = 0;

    #define CHECK(cond, msg) do { \
        if (cond) { printf("[OK] %s\n", msg); pass++; } \
        else { printf("[FAIL] %s\n", msg); fail++; } \
    } while(0)

    CHECK(artist_count == 2, "Artist count should be 2 (Polo & Pan, Metronomy)");
    CHECK(aa_count == 1, "Album artist count should be 1 (Polo & Pan)");
    CHECK(mbaid_count == 2, "MB Artist ID count should be 2");
    CHECK(mbaaid_count == 1, "MB Album Artist ID count should be 1");
    CHECK(xune_meta_mb_recording_id(handle) != nullptr, "MB Recording ID present");
    CHECK(xune_meta_mb_release_track_id(handle) != nullptr, "MB Release Track ID present");
    CHECK(xune_meta_mb_release_id(handle) != nullptr, "MB Release ID present");
    CHECK(xune_meta_mb_release_group_id(handle) != nullptr, "MB Release Group ID present");
    CHECK(xune_meta_acoustid_fingerprint(handle) != nullptr, "AcoustID fingerprint present");
    CHECK(!std::isnan(tg), "Track gain present");
    CHECK(!std::isnan(ag), "Album gain present");
    CHECK(xune_meta_has_picture(handle), "Has artwork");

    auto* artist0 = xune_meta_artist_at(handle, 0);
    auto* artist1 = xune_meta_artist_at(handle, 1);
    CHECK(artist0 && strcmp(artist0, "Polo & Pan") == 0, "Artist[0] is 'Polo & Pan'");
    CHECK(artist1 && strcmp(artist1, "Metronomy") == 0, "Artist[1] is 'Metronomy'");

    auto* display = xune_meta_artist_display(handle);
    CHECK(display && strstr(display, "Polo & Pan") && strstr(display, "Metronomy"),
          "Artist display contains both names");

    printf("\n%d passed, %d failed\n", pass, fail);

    xune_meta_close(handle);
    return fail > 0 ? 1 : 0;
}
