#!/usr/bin/env python3
"""Generate test audio files in all supported formats with multi-artist metadata.

Uses FFmpeg to create short silent audio files, then mutagen to tag them
with consistent multi-artist data matching MusicBrainz Picard's tagging style.

Formats: FLAC, MP3 (ID3v2.4), OGG Vorbis, M4A (AAC), WMA (ASF)
"""

import os
import subprocess
import sys

# Check dependencies
try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.id3 import (ID3, TIT2, TALB, TPE1, TPE2, TRCK, TPOS, TDRC,
                              TCON, TXXX, TCOM, TPE3)
    from mutagen.oggvorbis import OggVorbis
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.asf import ASF, ASFUnicodeAttribute
except ImportError:
    print("ERROR: mutagen not installed. Run: pip install mutagen")
    sys.exit(1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "testdata")
os.makedirs(OUT_DIR, exist_ok=True)

# 1 second of silence at 44100 Hz
DURATION = "1"
SAMPLE_RATE = "44100"

# ── Test metadata (mirrors Disco Nap by Polo & Pan feat. Metronomy) ──────────

TITLE = "Test Track"
ALBUM = "Test Album"
GENRE = "Electronic"
TRACK_NUMBER = 3
DISC_NUMBER = 1
YEAR = "2025"
RELEASE_DATE = "2025-03-28"

# Multi-artist: track has two artists, album has one
TRACK_ARTISTS = ["Polo & Pan", "Metronomy"]
TRACK_ARTIST_DISPLAY = "Polo & Pan feat. Metronomy"
ALBUM_ARTISTS = ["Polo & Pan"]
COMPOSER = "Test Composer"
CONDUCTOR = "Test Conductor"

# MusicBrainz IDs
MB_ARTIST_IDS = [
    "1d9ec7ea-0fa4-41d9-917b-723c735ebbfe",
    "93eb7110-0bc9-4d3f-816b-4b52ef982ec8",
]
MB_ALBUM_ARTIST_IDS = ["1d9ec7ea-0fa4-41d9-917b-723c735ebbfe"]
MB_RECORDING_ID = "4302ab95-3a56-4ad6-9d7c-64ff302c4d65"
MB_RELEASE_TRACK_ID = "2eaa2045-d40b-482e-b7dd-d4c905bd41f1"
MB_RELEASE_ID = "38df5f8a-7506-4e03-9111-25267963d14f"
MB_RELEASE_GROUP_ID = "c5895329-b865-479c-955f-567df2aaf959"
ACOUSTID_FP = "AQAATestFingerprint"

# ReplayGain
RG_TRACK_GAIN = "-9.29 dB"
RG_TRACK_PEAK = "1.031652"
RG_ALBUM_GAIN = "-9.07 dB"
RG_ALBUM_PEAK = "1.109558"


def run_ffmpeg(output_path, extra_args=None):
    """Generate a short silent audio file."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", DURATION,
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg failed for {output_path}: {result.stderr}")
        return False
    return True


def tag_flac(path):
    """Tag FLAC with Vorbis comments (Picard style: repeated keys for multi-value)."""
    f = FLAC(path)
    f["TITLE"] = TITLE
    f["ALBUM"] = ALBUM
    f["ARTIST"] = TRACK_ARTIST_DISPLAY
    f["ARTISTS"] = TRACK_ARTISTS  # Multi-value: separate entries
    f["ALBUMARTIST"] = ALBUM_ARTISTS[0]
    f["GENRE"] = GENRE
    f["TRACKNUMBER"] = str(TRACK_NUMBER)
    f["DISCNUMBER"] = str(DISC_NUMBER)
    f["DATE"] = RELEASE_DATE
    f["COMPOSER"] = COMPOSER
    f["CONDUCTOR"] = CONDUCTOR

    # Multi-value MBIDs (separate Vorbis comment entries)
    f["MUSICBRAINZ_ARTISTID"] = MB_ARTIST_IDS
    f["MUSICBRAINZ_ALBUMARTISTID"] = MB_ALBUM_ARTIST_IDS
    f["MUSICBRAINZ_TRACKID"] = MB_RECORDING_ID
    f["MUSICBRAINZ_RELEASETRACKID"] = MB_RELEASE_TRACK_ID
    f["MUSICBRAINZ_ALBUMID"] = MB_RELEASE_ID
    f["MUSICBRAINZ_RELEASEGROUPID"] = MB_RELEASE_GROUP_ID
    f["ACOUSTID_FINGERPRINT"] = ACOUSTID_FP

    f["REPLAYGAIN_TRACK_GAIN"] = RG_TRACK_GAIN
    f["REPLAYGAIN_TRACK_PEAK"] = RG_TRACK_PEAK
    f["REPLAYGAIN_ALBUM_GAIN"] = RG_ALBUM_GAIN
    f["REPLAYGAIN_ALBUM_PEAK"] = RG_ALBUM_PEAK

    f["ARTISTSORT"] = TRACK_ARTIST_DISPLAY
    f["ALBUMARTISTSORT"] = ALBUM_ARTISTS[0]

    f.save()
    print(f"  Tagged: {path}")


def tag_ogg(path):
    """Tag OGG Vorbis (same Vorbis comment format as FLAC)."""
    f = OggVorbis(path)
    f["TITLE"] = TITLE
    f["ALBUM"] = ALBUM
    f["ARTIST"] = TRACK_ARTIST_DISPLAY
    f["ARTISTS"] = TRACK_ARTISTS
    f["ALBUMARTIST"] = ALBUM_ARTISTS[0]
    f["GENRE"] = GENRE
    f["TRACKNUMBER"] = str(TRACK_NUMBER)
    f["DISCNUMBER"] = str(DISC_NUMBER)
    f["DATE"] = RELEASE_DATE
    f["COMPOSER"] = COMPOSER
    f["CONDUCTOR"] = CONDUCTOR

    f["MUSICBRAINZ_ARTISTID"] = MB_ARTIST_IDS
    f["MUSICBRAINZ_ALBUMARTISTID"] = MB_ALBUM_ARTIST_IDS
    f["MUSICBRAINZ_TRACKID"] = MB_RECORDING_ID
    f["MUSICBRAINZ_RELEASETRACKID"] = MB_RELEASE_TRACK_ID
    f["MUSICBRAINZ_ALBUMID"] = MB_RELEASE_ID
    f["MUSICBRAINZ_RELEASEGROUPID"] = MB_RELEASE_GROUP_ID
    f["ACOUSTID_FINGERPRINT"] = ACOUSTID_FP

    f["REPLAYGAIN_TRACK_GAIN"] = RG_TRACK_GAIN
    f["REPLAYGAIN_TRACK_PEAK"] = RG_TRACK_PEAK
    f["REPLAYGAIN_ALBUM_GAIN"] = RG_ALBUM_GAIN
    f["REPLAYGAIN_ALBUM_PEAK"] = RG_ALBUM_PEAK

    f["ARTISTSORT"] = TRACK_ARTIST_DISPLAY
    f["ALBUMARTISTSORT"] = ALBUM_ARTISTS[0]

    f.save()
    print(f"  Tagged: {path}")


def tag_mp3(path):
    """Tag MP3 with ID3v2.4 (Picard style: TXXX for MBIDs, multi-value via null-byte)."""
    f = MP3(path)
    f.tags = ID3()

    f.tags.add(TIT2(encoding=3, text=TITLE))
    f.tags.add(TALB(encoding=3, text=ALBUM))
    # TPE1: track artist (single formatted string for display)
    f.tags.add(TPE1(encoding=3, text=TRACK_ARTIST_DISPLAY))
    # TPE2: album artist
    f.tags.add(TPE2(encoding=3, text=ALBUM_ARTISTS[0]))
    f.tags.add(TCON(encoding=3, text=GENRE))
    f.tags.add(TRCK(encoding=3, text=str(TRACK_NUMBER)))
    f.tags.add(TPOS(encoding=3, text=str(DISC_NUMBER)))
    f.tags.add(TDRC(encoding=3, text=RELEASE_DATE))
    f.tags.add(TCOM(encoding=3, text=COMPOSER))
    f.tags.add(TPE3(encoding=3, text=CONDUCTOR))

    # TXXX frames for MusicBrainz IDs (multi-value via ID3v2.4 list)
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Artist Id", text=MB_ARTIST_IDS))
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Album Artist Id", text=MB_ALBUM_ARTIST_IDS))
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Track Id", text=[MB_RECORDING_ID]))
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Release Track Id", text=[MB_RELEASE_TRACK_ID]))
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Album Id", text=[MB_RELEASE_ID]))
    f.tags.add(TXXX(encoding=3, desc="MusicBrainz Release Group Id", text=[MB_RELEASE_GROUP_ID]))
    f.tags.add(TXXX(encoding=3, desc="Acoustid Fingerprint", text=[ACOUSTID_FP]))
    f.tags.add(TXXX(encoding=3, desc="ARTISTS", text=TRACK_ARTISTS))

    f.tags.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_GAIN", text=[RG_TRACK_GAIN]))
    f.tags.add(TXXX(encoding=3, desc="REPLAYGAIN_TRACK_PEAK", text=[RG_TRACK_PEAK]))
    f.tags.add(TXXX(encoding=3, desc="REPLAYGAIN_ALBUM_GAIN", text=[RG_ALBUM_GAIN]))
    f.tags.add(TXXX(encoding=3, desc="REPLAYGAIN_ALBUM_PEAK", text=[RG_ALBUM_PEAK]))

    f.save(v2_version=4)
    print(f"  Tagged: {path}")


def tag_m4a(path):
    """Tag M4A/AAC with MP4 atoms (Picard style: freeform atoms for MBIDs)."""
    f = MP4(path)

    f["\xa9nam"] = [TITLE]
    f["\xa9alb"] = [ALBUM]
    f["\xa9ART"] = [TRACK_ARTIST_DISPLAY]
    f["aART"] = [ALBUM_ARTISTS[0]]
    f["\xa9gen"] = [GENRE]
    f["trkn"] = [(TRACK_NUMBER, 12)]
    f["disk"] = [(DISC_NUMBER, 1)]
    f["\xa9day"] = [RELEASE_DATE]
    f["\xa9wrt"] = [COMPOSER]

    # Freeform atoms for MBIDs (Picard uses "com.apple.iTunes" namespace)
    def free(val):
        if isinstance(val, list):
            return [MP4FreeForm(v.encode("utf-8"), dataformat=MP4FreeForm.FORMAT_TEXT) for v in val]
        return [MP4FreeForm(val.encode("utf-8"), dataformat=MP4FreeForm.FORMAT_TEXT)]

    f["----:com.apple.iTunes:MusicBrainz Artist Id"] = free(MB_ARTIST_IDS)
    f["----:com.apple.iTunes:MusicBrainz Album Artist Id"] = free(MB_ALBUM_ARTIST_IDS)
    f["----:com.apple.iTunes:MusicBrainz Track Id"] = free(MB_RECORDING_ID)
    f["----:com.apple.iTunes:MusicBrainz Release Track Id"] = free(MB_RELEASE_TRACK_ID)
    f["----:com.apple.iTunes:MusicBrainz Album Id"] = free(MB_RELEASE_ID)
    f["----:com.apple.iTunes:MusicBrainz Release Group Id"] = free(MB_RELEASE_GROUP_ID)
    f["----:com.apple.iTunes:Acoustid Fingerprint"] = free(ACOUSTID_FP)
    f["----:com.apple.iTunes:ARTISTS"] = free(TRACK_ARTISTS)

    f["----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN"] = free(RG_TRACK_GAIN)
    f["----:com.apple.iTunes:REPLAYGAIN_TRACK_PEAK"] = free(RG_TRACK_PEAK)
    f["----:com.apple.iTunes:REPLAYGAIN_ALBUM_GAIN"] = free(RG_ALBUM_GAIN)
    f["----:com.apple.iTunes:REPLAYGAIN_ALBUM_PEAK"] = free(RG_ALBUM_PEAK)

    f.save()
    print(f"  Tagged: {path}")


def tag_wma(path):
    """Tag WMA/ASF with ASF attributes (Picard style)."""
    f = ASF(path)

    f["Title"] = [ASFUnicodeAttribute(TITLE)]
    f["WM/AlbumTitle"] = [ASFUnicodeAttribute(ALBUM)]
    f["Author"] = [ASFUnicodeAttribute(TRACK_ARTIST_DISPLAY)]
    f["WM/AlbumArtist"] = [ASFUnicodeAttribute(ALBUM_ARTISTS[0])]
    f["WM/Genre"] = [ASFUnicodeAttribute(GENRE)]
    f["WM/TrackNumber"] = [ASFUnicodeAttribute(str(TRACK_NUMBER))]
    f["WM/PartOfSet"] = [ASFUnicodeAttribute(str(DISC_NUMBER))]
    f["WM/Year"] = [ASFUnicodeAttribute(YEAR)]
    f["WM/Composer"] = [ASFUnicodeAttribute(COMPOSER)]
    f["WM/Conductor"] = [ASFUnicodeAttribute(CONDUCTOR)]

    # MusicBrainz attributes (multiple entries for multi-value)
    f["MusicBrainz/Artist Id"] = [ASFUnicodeAttribute(mid) for mid in MB_ARTIST_IDS]
    f["MusicBrainz/Album Artist Id"] = [ASFUnicodeAttribute(mid) for mid in MB_ALBUM_ARTIST_IDS]
    f["MusicBrainz/Track Id"] = [ASFUnicodeAttribute(MB_RECORDING_ID)]
    f["MusicBrainz/Release Track Id"] = [ASFUnicodeAttribute(MB_RELEASE_TRACK_ID)]
    f["MusicBrainz/Album Id"] = [ASFUnicodeAttribute(MB_RELEASE_ID)]
    f["MusicBrainz/Release Group Id"] = [ASFUnicodeAttribute(MB_RELEASE_GROUP_ID)]
    f["Acoustid/Fingerprint"] = [ASFUnicodeAttribute(ACOUSTID_FP)]
    f["WM/ARTISTS"] = [ASFUnicodeAttribute(a) for a in TRACK_ARTISTS]

    f["REPLAYGAIN_TRACK_GAIN"] = [ASFUnicodeAttribute(RG_TRACK_GAIN)]
    f["REPLAYGAIN_TRACK_PEAK"] = [ASFUnicodeAttribute(RG_TRACK_PEAK)]
    f["REPLAYGAIN_ALBUM_GAIN"] = [ASFUnicodeAttribute(RG_ALBUM_GAIN)]
    f["REPLAYGAIN_ALBUM_PEAK"] = [ASFUnicodeAttribute(RG_ALBUM_PEAK)]

    f.save()
    print(f"  Tagged: {path}")


def main():
    formats = {
        "test.flac": (["-c:a", "flac"], tag_flac),
        "test.mp3": (["-c:a", "libmp3lame", "-q:a", "2"], tag_mp3),
        "test.ogg": (["-c:a", "libvorbis", "-q:a", "2"], tag_ogg),
        "test.m4a": (["-c:a", "aac", "-b:a", "128k"], tag_m4a),
        "test.wma": (["-c:a", "wmav2", "-b:a", "128k"], tag_wma),
    }

    print("Generating test audio files...")
    for filename, (ffmpeg_args, tag_fn) in formats.items():
        path = os.path.join(OUT_DIR, filename)
        print(f"\n{filename}:")

        if not run_ffmpeg(path, ffmpeg_args):
            print(f"  SKIPPED (FFmpeg failed)")
            continue

        tag_fn(path)

    print(f"\nDone. Files in: {OUT_DIR}")
    print("\nFiles created:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
