#!/usr/bin/env python3
"""
Cross-library metadata comparison: xune_audio (native C++) vs mutagen (Python).

Reads all test fixtures with both libraries, compares field-by-field.
Also tests write round-trips: writes tags via native, reads back with mutagen.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mutagen
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.asf import ASF
from mutagen.id3 import TXXX, UFID

SCRIPT_DIR = Path(__file__).resolve().parent
# XuneAudioLibrary/tests/ -> Xune/tests/
FIXTURE_DIR = SCRIPT_DIR.parent.parent / "tests" / "Xune.Tests.Infrastructure.Ingestion" / "TestData" / "Fixtures"

NATIVE_DUMP = None  # Set after build


def find_native_dump():
    """Find or build the metadata_dump binary."""
    build_dir = Path(__file__).parent.parent / "build"
    # Check if it exists in build dir
    candidates = [
        build_dir / "metadata_dump",
        build_dir / "tests" / "metadata_dump",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def build_native_dump():
    """Build metadata_dump from the existing build tree."""
    build_dir = Path(__file__).parent.parent / "build"
    src = Path(__file__).parent / "metadata_dump.cpp"

    if not build_dir.exists():
        print("ERROR: Build directory not found. Run ./build.sh first.", file=sys.stderr)
        sys.exit(1)

    # Compile against the existing build tree
    include_dir = Path(__file__).parent.parent / "include"
    lib_dir = build_dir

    cmd = [
        "c++", "-std=c++17", "-O2",
        f"-I{include_dir}",
        f"-L{lib_dir}",
        "-lxune_audio",
        f"-Wl,-rpath,{lib_dir}",
        str(src),
        "-o", str(build_dir / "metadata_dump")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return str(build_dir / "metadata_dump")


def read_native(filepath):
    """Read metadata using our native library via metadata_dump."""
    result = subprocess.run(
        [NATIVE_DUMP, str(filepath)],
        capture_output=True, text=True,
        env={**os.environ, "DYLD_LIBRARY_PATH": str(Path(NATIVE_DUMP).parent)}
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}\nOutput: {result.stdout[:200]}"}


def read_mutagen(filepath):
    """Read metadata using mutagen."""
    ext = filepath.suffix.lower()
    try:
        f = mutagen.File(str(filepath))
    except Exception as e:
        return {"error": str(e)}

    if f is None:
        return {"error": "mutagen returned None"}

    data = {
        "duration_ms": int((f.info.length or 0) * 1000),
        "bitrate": getattr(f.info, "bitrate", 0) // 1000 if hasattr(f.info, "bitrate") and f.info.bitrate else 0,
        "sample_rate": getattr(f.info, "sample_rate", 0),
        "bits_per_sample": getattr(f.info, "bits_per_sample", 0),
    }

    if isinstance(f, (FLAC, OggVorbis)):
        data.update(_read_vorbis(f))
    elif isinstance(f, MP3):
        data.update(_read_id3(f))
    elif isinstance(f, MP4):
        data.update(_read_mp4(f))
    elif isinstance(f, ASF):
        data.update(_read_asf(f))
    else:
        data["error"] = f"Unsupported format: {type(f).__name__}"

    return data


def _first(tags, key, default=""):
    v = tags.get(key)
    if v is None:
        return default
    if isinstance(v, list):
        return str(v[0]) if v else default
    return str(v)


def _all(tags, key):
    v = tags.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _read_vorbis(f):
    """Read Vorbis Comment tags (FLAC, OGG)."""
    tags = f.tags or {}
    # Vorbis stores multi-value as repeated keys
    def multi(key):
        return tags.get(key, [])

    artists_raw = multi("ARTISTS")
    album_artists = multi("ALBUMARTIST")

    data = {
        "title": _first(tags, "TITLE"),
        "title_sort": _first(tags, "TITLESORT"),
        "album": _first(tags, "ALBUM"),
        "album_sort": _first(tags, "ALBUMSORT"),
        "genre": _first(tags, "GENRE"),
        "conductor": _first(tags, "CONDUCTOR"),
        "track_number": int(_first(tags, "TRACKNUMBER", "0").split("/")[0] or 0),
        "disc_number": int(_first(tags, "DISCNUMBER", "0").split("/")[0] or 0),
        "year": int(_first(tags, "DATE", "0")[:4] or 0) if _first(tags, "DATE") else 0,
        "artist_display": _first(tags, "ARTIST"),
        "artist_sort": _first(tags, "ARTISTSORT"),
        "album_artist_sort": _first(tags, "ALBUMARTISTSORT"),
        "artists": [str(a) for a in artists_raw],
        "album_artists": [str(a) for a in album_artists],
        "composers": [str(c) for c in multi("COMPOSER")],
        "mb_recording_id": _first(tags, "MUSICBRAINZ_TRACKID"),
        "mb_release_track_id": _first(tags, "MUSICBRAINZ_RELEASETRACKID"),
        "mb_release_id": _first(tags, "MUSICBRAINZ_ALBUMID"),
        "mb_release_group_id": _first(tags, "MUSICBRAINZ_RELEASEGROUPID"),
        "acoustid_fingerprint": _first(tags, "ACOUSTID_FINGERPRINT"),
        "mb_artist_ids": [str(x) for x in multi("MUSICBRAINZ_ARTISTID")],
        "mb_album_artist_ids": [str(x) for x in multi("MUSICBRAINZ_ALBUMARTISTID")],
        "has_picture": (len(f.pictures) > 0 if hasattr(f, "pictures") else False)
            or "metadata_block_picture" in tags or "METADATA_BLOCK_PICTURE" in tags,
    }

    # ReplayGain
    rg_tg = _first(tags, "REPLAYGAIN_TRACK_GAIN")
    rg_tp = _first(tags, "REPLAYGAIN_TRACK_PEAK")
    rg_ag = _first(tags, "REPLAYGAIN_ALBUM_GAIN")
    rg_ap = _first(tags, "REPLAYGAIN_ALBUM_PEAK")
    data["replaygain_track_gain"] = _parse_rg(rg_tg)
    data["replaygain_track_peak"] = _parse_rg_peak(rg_tp)
    data["replaygain_album_gain"] = _parse_rg(rg_ag)
    data["replaygain_album_peak"] = _parse_rg_peak(rg_ap)

    return data


def _read_id3(f):
    """Read ID3v2 tags (MP3)."""
    tags = f.tags or {}

    def txxx(desc):
        key = f"TXXX:{desc}"
        v = tags.get(key)
        if v:
            return list(v.text) if hasattr(v, "text") else [str(v)]
        return []

    def txxx_first(desc, default=""):
        vals = txxx(desc)
        return vals[0] if vals else default

    def tag_text(key, default=""):
        v = tags.get(key)
        if v and hasattr(v, "text") and v.text:
            return str(v.text[0])
        return default

    def tag_texts(key):
        v = tags.get(key)
        if v and hasattr(v, "text"):
            return [str(t) for t in v.text]
        return []

    # UFID for recording ID, TXXX fallback
    ufid = tags.get("UFID:http://musicbrainz.org")
    mb_recording = ufid.data.decode("ascii", errors="replace") if ufid else ""
    if not mb_recording:
        mb_recording = txxx_first("MusicBrainz Track Id")

    artists_raw = txxx("ARTISTS")

    data = {
        "title": tag_text("TIT2"),
        "title_sort": tag_text("TSOT"),
        "album": tag_text("TALB"),
        "album_sort": tag_text("TSOA"),
        "genre": tag_text("TCON"),
        "conductor": tag_text("TPE3"),
        "track_number": int(tag_text("TRCK", "0").split("/")[0] or 0),
        "disc_number": int(tag_text("TPOS", "0").split("/")[0] or 0),
        "year": int(tag_text("TDRC", "0")[:4] or 0) if tag_text("TDRC") else 0,
        "artist_display": tag_text("TPE1"),
        "artist_sort": tag_text("TSOP"),
        "album_artist_sort": txxx_first("ALBUMARTISTSORT") or tag_text("TSO2"),
        "artists": artists_raw,
        "album_artists": tag_texts("TPE2"),
        "composers": tag_texts("TCOM"),
        "mb_recording_id": mb_recording,
        "mb_release_track_id": txxx_first("MusicBrainz Release Track Id"),
        "mb_release_id": txxx_first("MusicBrainz Album Id"),
        "mb_release_group_id": txxx_first("MusicBrainz Release Group Id"),
        "acoustid_fingerprint": txxx_first("Acoustid Fingerprint") or txxx_first("ACOUSTID_FINGERPRINT"),
        "mb_artist_ids": txxx("MusicBrainz Artist Id"),
        "mb_album_artist_ids": txxx("MusicBrainz Album Artist Id"),
        "has_picture": any(k.startswith("APIC") for k in tags.keys()),
    }

    # ReplayGain
    data["replaygain_track_gain"] = _parse_rg(txxx_first("REPLAYGAIN_TRACK_GAIN") or txxx_first("replaygain_track_gain"))
    data["replaygain_track_peak"] = _parse_rg_peak(txxx_first("REPLAYGAIN_TRACK_PEAK") or txxx_first("replaygain_track_peak"))
    data["replaygain_album_gain"] = _parse_rg(txxx_first("REPLAYGAIN_ALBUM_GAIN") or txxx_first("replaygain_album_gain"))
    data["replaygain_album_peak"] = _parse_rg_peak(txxx_first("REPLAYGAIN_ALBUM_PEAK") or txxx_first("replaygain_album_peak"))

    return data


def _read_mp4(f):
    """Read MP4/M4A tags."""
    tags = f.tags or {}

    def itunes(key):
        v = tags.get(f"----:com.apple.iTunes:{key}")
        if v:
            return [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in v]
        return []

    def itunes_first(key, default=""):
        vals = itunes(key)
        return vals[0] if vals else default

    def mp4_text(key, default=""):
        v = tags.get(key)
        if v:
            return str(v[0]) if isinstance(v, list) else str(v)
        return default

    def mp4_int(key, default=0):
        v = tags.get(key)
        if v:
            if isinstance(v[0], tuple):
                return v[0][0]
            return int(v[0])
        return default

    artists_raw = itunes("ARTISTS")

    data = {
        "title": mp4_text("\xa9nam"),
        "title_sort": mp4_text("sonm"),
        "album": mp4_text("\xa9alb"),
        "album_sort": mp4_text("soal"),
        "genre": mp4_text("\xa9gen"),
        "conductor": "",  # MP4 has no standard conductor field
        "track_number": mp4_int("trkn"),
        "disc_number": mp4_int("disk"),
        "year": int(mp4_text("\xa9day", "0")[:4] or 0) if mp4_text("\xa9day") else 0,
        "artist_display": mp4_text("\xa9ART"),
        "artist_sort": mp4_text("soar"),
        "album_artist_sort": mp4_text("soaa"),
        "artists": artists_raw,
        "album_artists": [mp4_text("aART")] if mp4_text("aART") else [],
        "composers": [mp4_text("\xa9wrt")] if mp4_text("\xa9wrt") else [],
        "mb_recording_id": itunes_first("MusicBrainz Track Id"),
        "mb_release_track_id": itunes_first("MusicBrainz Release Track Id"),
        "mb_release_id": itunes_first("MusicBrainz Album Id"),
        "mb_release_group_id": itunes_first("MusicBrainz Release Group Id"),
        "acoustid_fingerprint": itunes_first("Acoustid Fingerprint"),
        "mb_artist_ids": itunes("MusicBrainz Artist Id"),
        "mb_album_artist_ids": itunes("MusicBrainz Album Artist Id"),
        "has_picture": "covr" in tags,
    }

    data["replaygain_track_gain"] = _parse_rg(itunes_first("REPLAYGAIN_TRACK_GAIN") or itunes_first("replaygain_track_gain"))
    data["replaygain_track_peak"] = _parse_rg_peak(itunes_first("REPLAYGAIN_TRACK_PEAK") or itunes_first("replaygain_track_peak"))
    data["replaygain_album_gain"] = _parse_rg(itunes_first("REPLAYGAIN_ALBUM_GAIN") or itunes_first("replaygain_album_gain"))
    data["replaygain_album_peak"] = _parse_rg_peak(itunes_first("REPLAYGAIN_ALBUM_PEAK") or itunes_first("replaygain_album_peak"))

    return data


def _read_asf(f):
    """Read ASF/WMA tags."""
    tags = f.tags or {}

    def asf_text(key, default=""):
        v = tags.get(key)
        if v:
            return str(v[0])
        return default

    def asf_all(key):
        v = tags.get(key)
        if v:
            return [str(x) for x in v]
        return []

    artists_raw = asf_all("WM/ARTISTS")

    data = {
        "title": asf_text("Title"),
        "title_sort": asf_text("TITLESORT"),
        "album": asf_text("WM/AlbumTitle"),
        "album_sort": asf_text("ALBUMSORT"),
        "genre": asf_text("WM/Genre"),
        "conductor": asf_text("WM/Conductor"),
        "track_number": int(asf_text("WM/TrackNumber", "0") or 0),
        "disc_number": int(asf_text("WM/PartOfSet", "0").split("/")[0] or 0),
        "year": _parse_year(asf_text("WM/Year")),
        "artist_display": asf_text("Author"),
        "artist_sort": asf_text("ARTISTSORT") or asf_text("WM/ArtistSortOrder"),
        "album_artist_sort": asf_text("ALBUMARTISTSORT") or asf_text("WM/AlbumArtistSortOrder"),
        "artists": artists_raw,
        "album_artists": asf_all("WM/AlbumArtist"),
        "composers": asf_all("WM/Composer"),
        "mb_recording_id": asf_text("MusicBrainz/Track Id"),
        "mb_release_track_id": asf_text("MusicBrainz/Release Track Id"),
        "mb_release_id": asf_text("MusicBrainz/Album Id"),
        "mb_release_group_id": asf_text("MusicBrainz/Release Group Id"),
        "acoustid_fingerprint": asf_text("Acoustid/Fingerprint"),
        "mb_artist_ids": asf_all("MusicBrainz/Artist Id"),
        "mb_album_artist_ids": asf_all("MusicBrainz/Album Artist Id"),
        "has_picture": "WM/Picture" in tags,
    }

    data["replaygain_track_gain"] = _parse_rg(asf_text("REPLAYGAIN_TRACK_GAIN") or asf_text("ReplayGain/Track"))
    data["replaygain_track_peak"] = _parse_rg_peak(asf_text("REPLAYGAIN_TRACK_PEAK") or asf_text("ReplayGain/Track Peak"))
    data["replaygain_album_gain"] = _parse_rg(asf_text("REPLAYGAIN_ALBUM_GAIN") or asf_text("ReplayGain/Album"))
    data["replaygain_album_peak"] = _parse_rg_peak(asf_text("REPLAYGAIN_ALBUM_PEAK") or asf_text("ReplayGain/Album Peak"))

    return data


def _parse_year(val):
    if not val:
        return 0
    m = re.match(r"(\d{4})", val)
    if m:
        return int(m.group(1))
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _parse_rg(val):
    if not val:
        return None
    m = re.search(r"[-+]?\d+\.?\d*", val)
    return float(m.group()) if m else None


def _parse_rg_peak(val):
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── pytaglib reader (same TagLib C library as our native code) ───────────────

def read_pytaglib(filepath):
    """Read metadata using pytaglib (TagLib C library Python bindings)."""
    import taglib
    try:
        f = taglib.File(str(filepath))
    except Exception as e:
        return {"error": str(e)}

    tags = f.tags or {}

    def first(key, default=""):
        v = tags.get(key)
        if v:
            return str(v[0])
        return default

    def multi(key):
        return [str(x) for x in tags.get(key, [])]

    data = {
        "duration_ms": int(f.length * 1000) if f.length else 0,
        "bitrate": f.bitrate or 0,
        "sample_rate": f.sampleRate or 0,
        "title": first("TITLE"),
        "title_sort": first("TITLESORT"),
        "album": first("ALBUM"),
        "album_sort": first("ALBUMSORT"),
        "genre": first("GENRE"),
        "conductor": first("CONDUCTOR"),
        "track_number": int(first("TRACKNUMBER", "0").split("/")[0] or 0),
        "disc_number": int(first("DISCNUMBER", "0").split("/")[0] or 0),
        "year": _parse_year(first("DATE")),
        "artist_display": first("ARTIST"),
        "artist_sort": first("ARTISTSORT"),
        "album_artist_sort": first("ALBUMARTISTSORT"),
        "artists": multi("ARTISTS"),
        "album_artists": multi("ALBUMARTIST"),
        "composers": multi("COMPOSER"),
        "mb_recording_id": first("MUSICBRAINZ_TRACKID"),
        "mb_release_track_id": first("MUSICBRAINZ_RELEASETRACKID"),
        "mb_release_id": first("MUSICBRAINZ_ALBUMID"),
        "mb_release_group_id": first("MUSICBRAINZ_RELEASEGROUPID"),
        "acoustid_fingerprint": first("ACOUSTID_FINGERPRINT"),
        "mb_artist_ids": multi("MUSICBRAINZ_ARTISTID"),
        "mb_album_artist_ids": multi("MUSICBRAINZ_ALBUMARTISTID"),
        "replaygain_track_gain": _parse_rg(first("REPLAYGAIN_TRACK_GAIN")),
        "replaygain_track_peak": _parse_rg_peak(first("REPLAYGAIN_TRACK_PEAK")),
        "replaygain_album_gain": _parse_rg(first("REPLAYGAIN_ALBUM_GAIN")),
        "replaygain_album_peak": _parse_rg_peak(first("REPLAYGAIN_ALBUM_PEAK")),
        "has_picture": False,  # pytaglib doesn't expose artwork through properties()
    }
    f.close()
    return data


# ── Write round-trip test ────────────────────────────────────────────────────

NATIVE_WRITE_DUMP = None  # path to metadata_dump binary (reused)

def write_roundtrip_test(filepath):
    """Write tags via native API, read back with mutagen, verify."""
    ext = filepath.suffix.lower()
    if ext in (".wav", ".aac"):
        return None  # skip formats with limited write support

    with tempfile.TemporaryDirectory() as tmpdir:
        copy = Path(tmpdir) / filepath.name
        shutil.copy2(filepath, copy)

        # Write via native using the existing test infrastructure
        # We'll use a small C program to set known values
        write_bin = _build_write_test()
        if not write_bin:
            return {"skipped": "write test binary not built"}

        result = subprocess.run(
            [write_bin, str(copy)],
            capture_output=True, text=True,
            env={**os.environ, "DYLD_LIBRARY_PATH": str(Path(write_bin).parent)}
        )
        if result.returncode != 0:
            return {"error": f"write failed: {result.stderr}"}

        # Read back with mutagen
        mg = read_mutagen(copy)
        if "error" in mg:
            return mg

        # Read back with native
        native = read_native(copy)
        if "error" in native:
            return native

        errors = []
        expected = {
            "title": "Roundtrip Title",
            "album": "Roundtrip Album",
            "genre": "Electronic",
            "track_number": 7,
            "disc_number": 2,
            "year": 2025,
        }
        for key, exp in expected.items():
            mg_val = mg.get(key, "")
            nat_val = native.get(key, "")
            if mg_val != exp:
                errors.append(f"  mutagen {key}: expected {exp!r}, got {mg_val!r}")
            if nat_val != exp:
                errors.append(f"  native  {key}: expected {exp!r}, got {nat_val!r}")

        return {"errors": errors} if errors else {"ok": True}


_write_test_bin = None

def _build_write_test():
    global _write_test_bin
    if _write_test_bin:
        return _write_test_bin

    build_dir = Path(__file__).parent.parent / "build"
    src = build_dir / "_write_test.cpp"

    src.write_text('''
#include <xune_audio/xune_metadata.h>
#include <cstdio>
int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    xune_meta_handle_t h = nullptr;
    if (xune_meta_open(argv[1], &h) != XUNE_META_OK) return 1;
    xune_meta_set_title(h, "Roundtrip Title");
    xune_meta_set_album(h, "Roundtrip Album");
    xune_meta_set_genre(h, "Electronic");
    xune_meta_set_track_number(h, 7);
    xune_meta_set_disc_number(h, 2);
    xune_meta_set_year(h, 2025);
    const char* artists[] = {"Artist A", "Artist B"};
    xune_meta_set_artists(h, artists, 2);
    const char* aa[] = {"Album Artist X"};
    xune_meta_set_album_artists(h, aa, 1);
    auto err = xune_meta_save(h);
    xune_meta_close(h);
    return err == XUNE_META_OK ? 0 : 1;
}
''')

    include_dir = Path(__file__).parent.parent / "include"
    cmd = [
        "c++", "-std=c++17", "-O2",
        f"-I{include_dir}", f"-L{build_dir}",
        "-lxune_audio", f"-Wl,-rpath,{build_dir}",
        str(src), "-o", str(build_dir / "metadata_write_test")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Write test build failed: {result.stderr}", file=sys.stderr)
        return None

    _write_test_bin = str(build_dir / "metadata_write_test")
    return _write_test_bin


# ── Comparison Logic ─────────────────────────────────────────────────────────

# Fields where we tolerate differences
KNOWN_DEVIATIONS = {
    # WAV has very limited tag support
    ".wav": {"all": "WAV has limited tag support in most implementations"},
    # AAC (raw ADTS) has no tag container
    ".aac": {"all": "Raw AAC has no tag container"},
}

# Fields to skip comparison for
SKIP_FIELDS = {"picture_size", "picture_mime", "release_date", "bits_per_sample"}

# Float comparison tolerance
FLOAT_TOL = 0.01
DURATION_TOL_MS = 50


def compare(native, ref_data, ext, ref_label="ref", skip_fields=None):
    """Compare native vs reference results. Returns list of mismatches."""
    mismatches = []
    skip = (skip_fields or set()) | SKIP_FIELDS

    if ext in KNOWN_DEVIATIONS and "all" in KNOWN_DEVIATIONS[ext]:
        return []

    compare_fields = [
        "title", "album", "genre", "conductor",
        "track_number", "disc_number", "year",
        "artist_display",
        "mb_recording_id", "mb_release_track_id",
        "mb_release_id", "mb_release_group_id",
        "acoustid_fingerprint",
        "has_picture",
    ]

    for field in compare_fields:
        if field in skip:
            continue
        n = native.get(field, "")
        m = ref_data.get(field, "")
        if n is None:
            n = ""
        if m is None:
            m = ""
        if n != m:
            mismatches.append(f"  {field}: native={n!r}  {ref_label}={m!r}")

    nd = native.get("duration_ms", 0)
    md = ref_data.get("duration_ms", 0)
    if "duration_ms" not in skip and abs(nd - md) > DURATION_TOL_MS:
        mismatches.append(f"  duration_ms: native={nd}  {ref_label}={md}  (diff={abs(nd-md)}ms)")

    for field in ["artists", "album_artists", "composers", "mb_artist_ids", "mb_album_artist_ids"]:
        if field in skip:
            continue
        n = native.get(field, [])
        m = ref_data.get(field, [])
        if n != m:
            mismatches.append(f"  {field}: native={n!r}  {ref_label}={m!r}")

    for field in ["replaygain_track_gain", "replaygain_track_peak",
                   "replaygain_album_gain", "replaygain_album_peak"]:
        if field in skip:
            continue
        n = native.get(field)
        m = ref_data.get(field)
        if n is None and m is None:
            continue
        if n is None or m is None:
            mismatches.append(f"  {field}: native={n}  {ref_label}={m}")
        elif abs(float(n) - float(m)) > FLOAT_TOL:
            mismatches.append(f"  {field}: native={n}  {ref_label}={m}")

    for field in ["title_sort", "album_sort", "artist_sort", "album_artist_sort"]:
        if field in skip:
            continue
        n = native.get(field, "")
        m = ref_data.get(field, "")
        if n != m:
            mismatches.append(f"  {field}: native={n!r}  {ref_label}={m!r}")

    return mismatches


# ── Comparison pass runner ────────────────────────────────────────────────────

def run_pass(files, reader, label, base_dir=None, skip_fields=None, max_diffs=20):
    """Run a comparison pass of native vs a reference reader on a list of files."""
    passed = 0
    failed = 0
    errs = 0
    mismatches = {}

    for filepath in files:
        ext = filepath.suffix.lower()
        name = filepath.relative_to(base_dir) if base_dir else filepath.name

        if ext in (".wav", ".aac"):
            continue

        native = read_native(filepath)
        ref = reader(filepath)

        if "error" in native:
            errs += 1
            if errs <= 5:
                print(f"  ERR   {name} native: {native['error']}")
            continue
        if "error" in ref:
            errs += 1
            if errs <= 5:
                print(f"  ERR   {name} {label}: {ref['error']}")
            continue

        diffs = compare(native, ref, ext, label, skip_fields)
        if diffs:
            failed += 1
            mismatches[str(name)] = diffs
            if failed <= max_diffs:
                print(f"  DIFF  {name}")
                for d in diffs:
                    print(f"        {d}")
        else:
            passed += 1

    total = passed + failed + errs
    print(f"\n  {passed} passed, {failed} failed, {errs} errors / {total} total")

    if failed > max_diffs:
        print(f"  (showing first {max_diffs} of {failed} failures)")

    if mismatches:
        field_counts = {}
        for fname, diffs in mismatches.items():
            for d in diffs:
                field = d.strip().split(":")[0]
                field_counts[field] = field_counts.get(field, 0) + 1
        print("\n  Mismatch field frequency:")
        for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
            print(f"    {field}: {count} files")

    return failed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global NATIVE_DUMP

    NATIVE_DUMP = find_native_dump()
    if not NATIVE_DUMP:
        print("Building metadata_dump...")
        NATIVE_DUMP = build_native_dump()
    print(f"Using native binary: {NATIVE_DUMP}")

    if not FIXTURE_DIR.exists():
        print(f"Fixture directory not found: {FIXTURE_DIR}", file=sys.stderr)
        sys.exit(1)

    fixtures = sorted(FIXTURE_DIR.glob("*"))
    fixtures = [f for f in fixtures if f.suffix.lower() in
                (".mp3", ".flac", ".ogg", ".oga", ".m4a", ".alac", ".wma", ".wav", ".aac")]

    REAL_MUSIC_DIR = Path("/Users/andymoe/Music/Music Testing/Music")
    real_files = []
    if REAL_MUSIC_DIR.exists():
        for ext in ("*.mp3", "*.flac", "*.m4a", "*.ogg", "*.wma", "*.oga"):
            real_files.extend(REAL_MUSIC_DIR.rglob(ext))
        real_files.sort()

    total_fail = 0

    # pytaglib doesn't expose has_picture or bits_per_sample
    pytaglib_skip = {"has_picture", "bits_per_sample"}

    # ── pytaglib (same TagLib C library — authoritative comparison) ──
    print(f"\n{'=' * 72}")
    print(f"pytaglib vs native — Fixtures ({len(fixtures)} files)")
    print("=" * 72)
    total_fail += run_pass(fixtures, read_pytaglib, "pytaglib", skip_fields=pytaglib_skip)

    if real_files:
        print(f"\n{'=' * 72}")
        print(f"pytaglib vs native — Real library ({len(real_files)} files)")
        print("=" * 72)
        total_fail += run_pass(real_files, read_pytaglib, "pytaglib",
                               base_dir=REAL_MUSIC_DIR, skip_fields=pytaglib_skip)

    # ── mutagen (independent implementation — cross-validation) ──
    print(f"\n{'=' * 72}")
    print(f"mutagen vs native — Fixtures ({len(fixtures)} files)")
    print("=" * 72)
    total_fail += run_pass(fixtures, read_mutagen, "mutagen")

    if real_files:
        print(f"\n{'=' * 72}")
        print(f"mutagen vs native — Real library ({len(real_files)} files)")
        print("=" * 72)
        total_fail += run_pass(real_files, read_mutagen, "mutagen",
                               base_dir=REAL_MUSIC_DIR)

    # ── Write round-trip tests ──
    print("\n" + "=" * 72)
    print("Write round-trip tests (native write -> mutagen + native read)")
    print("=" * 72)

    write_formats = [f for f in fixtures if f.suffix.lower() in (".mp3", ".flac", ".ogg", ".m4a", ".wma")]
    write_pass = 0
    write_fail = 0
    write_skip = 0

    for filepath in write_formats:
        name = filepath.name
        result = write_roundtrip_test(filepath)
        if result is None:
            print(f"  SKIP  {name}")
            write_skip += 1
        elif "skipped" in result:
            print(f"  SKIP  {name} ({result['skipped']})")
            write_skip += 1
        elif "error" in result:
            print(f"  ERR   {name}: {result['error']}")
            write_fail += 1
        elif result.get("ok"):
            print(f"  OK    {name}")
            write_pass += 1
        else:
            print(f"  DIFF  {name}")
            for e in result.get("errors", []):
                print(f"        {e}")
            write_fail += 1

    print("=" * 72)
    print(f"\nWrite round-trip: {write_pass} passed, {write_fail} failed, {write_skip} skipped")

    total_ok = total_fail == 0 and write_fail == 0
    print(f"\n{'ALL PASSED' if total_ok else 'FAILURES DETECTED'}")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
