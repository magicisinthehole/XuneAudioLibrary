/**
 * @file nowplaying_windows.cpp
 * @brief Windows implementation of Now Playing API using SMTC (SystemMediaTransportControls)
 *
 * Uses C++/WinRT to access Windows.Media.SystemMediaTransportControls.
 * Requires the HWND of the main application window passed to xune_nowplaying_init().
 */

#include "xune_audio/xune_nowplaying.h"

#include <windows.h>
#include <systemmediatransportcontrolsinterop.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.Streams.h>

#include <mutex>
#include <atomic>

using namespace winrt;
using namespace Windows::Media;
using namespace Windows::Storage::Streams;

// ── Global State ─────────────────────────────────────────────────────────────

static std::recursive_mutex g_mutex;
static xune_command_callback_t g_callback = nullptr;
static void* g_user_data = nullptr;
static std::atomic<bool> g_initialized{false};

static SystemMediaTransportControls g_smtc{nullptr};
static SystemMediaTransportControlsDisplayUpdater g_updater{nullptr};
static winrt::event_token g_button_token{};

// ── Helpers ──────────────────────────────────────────────────────────────────

static void dispatch_command(xune_media_command_t cmd, int64_t seek_ms = 0) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_callback) {
        g_callback(cmd, seek_ms, g_user_data);
    }
}

static void on_button_pressed(
    SystemMediaTransportControls const&,
    SystemMediaTransportControlsButtonPressedEventArgs const& args) {

    switch (args.Button()) {
        case SystemMediaTransportControlsButton::Play:
            dispatch_command(XUNE_CMD_PLAY);
            break;
        case SystemMediaTransportControlsButton::Pause:
            dispatch_command(XUNE_CMD_PAUSE);
            break;
        case SystemMediaTransportControlsButton::Stop:
            dispatch_command(XUNE_CMD_STOP);
            break;
        case SystemMediaTransportControlsButton::Next:
            dispatch_command(XUNE_CMD_NEXT);
            break;
        case SystemMediaTransportControlsButton::Previous:
            dispatch_command(XUNE_CMD_PREVIOUS);
            break;
        default:
            break;
    }
}

// ── winrt::hstring from UTF-8 ────────────────────────────────────────────────

static winrt::hstring to_hstring(const char* utf8) {
    if (!utf8 || utf8[0] == '\0') return {};

    int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (len <= 0) return {};

    std::wstring wide(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide.data(), len);
    return winrt::hstring(wide);
}

// ── API Implementation ───────────────────────────────────────────────────────

int xune_nowplaying_init(
    void* window_handle,
    xune_command_callback_t callback,
    void* user_data)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (g_initialized.load()) return 0;

    HWND hwnd = static_cast<HWND>(window_handle);
    if (!hwnd) return -2;

    try {
        winrt::init_apartment(winrt::apartment_type::single_threaded);

        // Get SMTC for this window via the interop interface
        auto interop = winrt::get_activation_factory<SystemMediaTransportControls,
            ISystemMediaTransportControlsInterop>();

        winrt::com_ptr<ABI::Windows::Media::ISystemMediaTransportControls> raw_smtc;
        HRESULT hr = interop->GetForWindow(
            hwnd,
            winrt::guid_of<ABI::Windows::Media::ISystemMediaTransportControls>(),
            raw_smtc.put_void());

        if (FAILED(hr)) return -1;

        g_smtc = raw_smtc.as<SystemMediaTransportControls>();
        g_updater = g_smtc.DisplayUpdater();

        // Enable buttons
        g_smtc.IsEnabled(true);
        g_smtc.IsPlayEnabled(true);
        g_smtc.IsPauseEnabled(true);
        g_smtc.IsStopEnabled(true);
        g_smtc.IsNextEnabled(true);
        g_smtc.IsPreviousEnabled(true);

        // Register button handler
        g_callback = callback;
        g_user_data = user_data;
        g_button_token = g_smtc.ButtonPressed(on_button_pressed);

        g_initialized.store(true);
        return 0;
    }
    catch (...) {
        return -1;
    }
}

bool xune_nowplaying_is_available(void) {
    return true;
}

void xune_nowplaying_cleanup(void) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized.load()) return;

    try {
        if (g_smtc) {
            g_smtc.ButtonPressed(g_button_token);
            g_smtc.IsEnabled(false);
            g_updater.ClearAll();
            g_updater.Update();
        }
    } catch (...) {}

    g_smtc = nullptr;
    g_updater = nullptr;
    g_callback = nullptr;
    g_user_data = nullptr;
    g_initialized.store(false);
}

void xune_nowplaying_set_metadata(const xune_track_metadata_t* metadata) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load() || !metadata) return;

    try {
        g_updater.Type(MediaPlaybackType::Music);

        auto props = g_updater.MusicProperties();
        props.Title(to_hstring(metadata->title));
        props.Artist(to_hstring(metadata->artist));
        props.AlbumTitle(to_hstring(metadata->album));
        props.AlbumArtist(to_hstring(metadata->album_artist));

        // Set artwork from raw image data
        if (metadata->artwork_data && metadata->artwork_size > 0) {
            auto stream = InMemoryRandomAccessStream();
            auto writer = DataWriter(stream.GetOutputStreamAt(0));
            writer.WriteBytes(winrt::array_view<const uint8_t>(
                static_cast<const uint8_t*>(metadata->artwork_data),
                static_cast<uint32_t>(metadata->artwork_size)));
            writer.StoreAsync().get();
            writer.DetachStream();

            g_updater.Thumbnail(
                RandomAccessStreamReference::CreateFromStream(stream));
        }

        g_updater.Update();
    } catch (...) {}
}

void xune_nowplaying_clear_metadata(void) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    try {
        g_updater.ClearAll();
        g_updater.Update();
        g_smtc.PlaybackStatus(MediaPlaybackStatus::Closed);
    } catch (...) {}
}

void xune_nowplaying_set_playback_state(xune_playback_state_t state) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    try {
        MediaPlaybackStatus status;
        switch (state) {
            case XUNE_PLAYBACK_PLAYING: status = MediaPlaybackStatus::Playing; break;
            case XUNE_PLAYBACK_PAUSED:  status = MediaPlaybackStatus::Paused; break;
            default:                    status = MediaPlaybackStatus::Stopped; break;
        }
        g_smtc.PlaybackStatus(status);
    } catch (...) {}
}

void xune_nowplaying_set_position(int64_t position_ms, int64_t duration_ms) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    try {
        SystemMediaTransportControlsTimelineProperties timeline;
        timeline.StartTime(std::chrono::seconds(0));
        timeline.EndTime(std::chrono::milliseconds(duration_ms));
        timeline.Position(std::chrono::milliseconds(position_ms));
        timeline.MinSeekTime(std::chrono::seconds(0));
        timeline.MaxSeekTime(std::chrono::milliseconds(duration_ms));
        g_smtc.UpdateTimelineProperties(timeline);
    } catch (...) {}
}

void xune_nowplaying_set_playback_rate(float rate) {
    (void)rate;
    // SMTC doesn't expose a playback rate property
}

void xune_nowplaying_set_command_enabled(xune_media_command_t command, bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    try {
        switch (command) {
            case XUNE_CMD_PLAY:              g_smtc.IsPlayEnabled(enabled); break;
            case XUNE_CMD_PAUSE:             g_smtc.IsPauseEnabled(enabled); break;
            case XUNE_CMD_STOP:              g_smtc.IsStopEnabled(enabled); break;
            case XUNE_CMD_NEXT:              g_smtc.IsNextEnabled(enabled); break;
            case XUNE_CMD_PREVIOUS:          g_smtc.IsPreviousEnabled(enabled); break;
            default: break;
        }
    } catch (...) {}
}
