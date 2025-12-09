/**
 * @file nowplaying_stub.cpp
 * @brief Stub implementation of Now Playing API for unsupported platforms
 *
 * This implementation does nothing but provides the correct API surface
 * for platforms where Now Playing integration is not available.
 */

#include "xune_audio/xune_nowplaying.h"

int xune_nowplaying_init(
    void* window_handle,
    xune_command_callback_t callback,
    void* user_data)
{
    (void)window_handle;
    (void)callback;
    (void)user_data;
    return 0;  // Success - initialization is a no-op
}

bool xune_nowplaying_is_available(void)
{
    return false;  // Not available on this platform
}

void xune_nowplaying_cleanup(void)
{
    // No-op
}

void xune_nowplaying_set_metadata(const xune_track_metadata_t* metadata)
{
    (void)metadata;
    // No-op
}

void xune_nowplaying_clear_metadata(void)
{
    // No-op
}

void xune_nowplaying_set_playback_state(xune_playback_state_t state)
{
    (void)state;
    // No-op
}

void xune_nowplaying_set_position(int64_t position_ms, int64_t duration_ms)
{
    (void)position_ms;
    (void)duration_ms;
    // No-op
}

void xune_nowplaying_set_playback_rate(float rate)
{
    (void)rate;
    // No-op
}

void xune_nowplaying_set_command_enabled(xune_media_command_t command, bool enabled)
{
    (void)command;
    (void)enabled;
    // No-op
}
