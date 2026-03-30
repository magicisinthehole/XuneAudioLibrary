#include <xune_audio/xune_metadata.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined"
#endif

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    if (cond) { printf("  [OK] " fmt "\n", ##__VA_ARGS__); g_pass++; } \
    else { printf("  [FAIL] " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while(0)

#define S(x) ((x) ? (x) : "(null)")

void test_format(const char* filename) {
    std::string path = std::string(TEST_DATA_DIR) + "/" + filename;
    printf("\n=== %s ===\n", filename);

    xune_meta_handle_t handle = nullptr;
    auto err = xune_meta_open(path.c_str(), &handle);
    if (err != XUNE_META_OK) {
        printf("  [FAIL] Failed to open: error %d\n", err);
        g_fail++;
        return;
    }

    // Single-value tags
    auto* title = xune_meta_title(handle);
    CHECK(title && strcmp(title, "Test Track") == 0, "Title = 'Test Track' (got: '%s')", S(title));

    auto* album = xune_meta_album(handle);
    CHECK(album && strcmp(album, "Test Album") == 0, "Album = 'Test Album' (got: '%s')", S(album));

    auto* genre = xune_meta_genre(handle);
    CHECK(genre && strcmp(genre, "Electronic") == 0, "Genre = 'Electronic' (got: '%s')", S(genre));

    CHECK(xune_meta_track_number(handle) == 3, "Track number = 3 (got: %u)", xune_meta_track_number(handle));
    CHECK(xune_meta_disc_number(handle) == 1, "Disc number = 1 (got: %u)", xune_meta_disc_number(handle));

    // Multi-value artists (CRITICAL)
    int ac = xune_meta_artist_count(handle);
    CHECK(ac == 2, "Artist count = 2 (got: %d)", ac);
    if (ac >= 2) {
        auto* a0 = xune_meta_artist_at(handle, 0);
        auto* a1 = xune_meta_artist_at(handle, 1);
        CHECK(a0 && strcmp(a0, "Polo & Pan") == 0, "Artist[0] = 'Polo & Pan' (got: '%s')", S(a0));
        CHECK(a1 && strcmp(a1, "Metronomy") == 0, "Artist[1] = 'Metronomy' (got: '%s')", S(a1));
    }

    auto* display = xune_meta_artist_display(handle);
    CHECK(display && strstr(display, "Polo") && strstr(display, "Metronomy"),
          "Artist display contains both (got: '%s')", S(display));

    int aac = xune_meta_album_artist_count(handle);
    CHECK(aac >= 1, "Album artist count >= 1 (got: %d)", aac);
    if (aac >= 1) {
        auto* aa0 = xune_meta_album_artist_at(handle, 0);
        CHECK(aa0 && strcmp(aa0, "Polo & Pan") == 0, "AlbumArtist[0] = 'Polo & Pan' (got: '%s')", S(aa0));
    }

    // Multi-value MBIDs
    int mbac = xune_meta_mb_artist_id_count(handle);
    CHECK(mbac == 2, "MB Artist ID count = 2 (got: %d)", mbac);
    if (mbac >= 2) {
        auto* mb0 = xune_meta_mb_artist_id_at(handle, 0);
        auto* mb1 = xune_meta_mb_artist_id_at(handle, 1);
        CHECK(mb0 && strstr(mb0, "1d9ec7ea"), "MB ArtistID[0] starts with 1d9ec7ea (got: '%s')", S(mb0));
        CHECK(mb1 && strstr(mb1, "93eb7110"), "MB ArtistID[1] starts with 93eb7110 (got: '%s')", S(mb1));
    }

    int mbaac = xune_meta_mb_album_artist_id_count(handle);
    CHECK(mbaac >= 1, "MB Album Artist ID count >= 1 (got: %d)", mbaac);

    CHECK(xune_meta_mb_recording_id(handle) != nullptr, "MB Recording ID present");
    CHECK(xune_meta_mb_release_track_id(handle) != nullptr, "MB Release Track ID present");
    CHECK(xune_meta_mb_release_id(handle) != nullptr, "MB Release ID present");
    CHECK(xune_meta_mb_release_group_id(handle) != nullptr, "MB Release Group ID present");
    CHECK(xune_meta_acoustid_fingerprint(handle) != nullptr, "AcoustID fingerprint present");

    // ReplayGain
    double tg = xune_meta_replaygain_track_gain(handle);
    double ag = xune_meta_replaygain_album_gain(handle);
    CHECK(!std::isnan(tg), "Track gain present (got: %f)", tg);
    CHECK(!std::isnan(ag), "Album gain present (got: %f)", ag);

    xune_meta_close(handle);
}

int main() {
    const char* formats[] = {
        "test.flac",
        "test.mp3",
        "test.ogg",
        "test.m4a",
        "test.wma",
    };

    for (auto* f : formats)
        test_format(f);

    printf("\n========================================\n");
    printf("TOTAL: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
