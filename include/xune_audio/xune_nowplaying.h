/**
 * @file xune_nowplaying.h
 * @brief Cross-platform Now Playing integration API
 *
 * Provides integration with OS media control surfaces:
 * - macOS: MPNowPlayingInfoCenter + MPRemoteCommandCenter
 * - Windows: SystemMediaTransportControls (SMTC)
 * - Linux: MPRIS D-Bus interface
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "xune_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types & Enums
 * ============================================================================ */

/**
 * @brief Media command types received from OS media controls
 */
typedef enum {
    XUNE_CMD_PLAY = 0,
    XUNE_CMD_PAUSE = 1,
    XUNE_CMD_TOGGLE_PLAY_PAUSE = 2,
    XUNE_CMD_STOP = 3,
    XUNE_CMD_NEXT = 4,
    XUNE_CMD_PREVIOUS = 5,
    XUNE_CMD_SEEK = 6,           /**< seek_position_ms contains target position */
    XUNE_CMD_SKIP_FORWARD = 7,   /**< seek_position_ms contains skip amount */
    XUNE_CMD_SKIP_BACKWARD = 8   /**< seek_position_ms contains skip amount */
} xune_media_command_t;

/**
 * @brief Playback state for Now Playing display
 */
typedef enum {
    XUNE_PLAYBACK_STOPPED = 0,
    XUNE_PLAYBACK_PLAYING = 1,
    XUNE_PLAYBACK_PAUSED = 2
} xune_playback_state_t;

/**
 * @brief Callback function type for receiving media commands from OS
 *
 * @param command The command type received
 * @param seek_position_ms For SEEK commands: target position in milliseconds
 *                         For SKIP commands: amount to skip in milliseconds
 *                         For other commands: 0
 * @param user_data User-provided context pointer from xune_nowplaying_init()
 */
typedef void (*xune_command_callback_t)(
    xune_media_command_t command,
    int64_t seek_position_ms,
    void* user_data
);

/**
 * @brief Track metadata structure for Now Playing display
 *
 * All string fields are UTF-8 encoded and may be NULL if unknown.
 * Strings are not copied - caller must ensure they remain valid until
 * the next call to xune_nowplaying_set_metadata() or xune_nowplaying_clear_metadata().
 */
typedef struct {
    const char* title;          /**< Track title (required) */
    const char* artist;         /**< Artist name (optional, may be NULL) */
    const char* album;          /**< Album title (optional, may be NULL) */
    const char* album_artist;   /**< Album artist (optional, may be NULL) */
    int64_t duration_ms;        /**< Track duration in milliseconds (0 if unknown) */
    const char* artwork_path;   /**< Path to album art file (optional, may be NULL) */
    const void* artwork_data;   /**< Raw image data PNG/JPEG (optional, may be NULL) */
    size_t artwork_size;        /**< Size of artwork_data in bytes (0 if no data) */
} xune_track_metadata_t;

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

/**
 * @brief Initialize the Now Playing subsystem
 *
 * Must be called before any other Now Playing functions.
 * On Windows, requires a valid window handle (HWND).
 * On macOS and Linux, window_handle is ignored (pass NULL).
 *
 * @param window_handle Platform window handle (HWND on Windows, NULL on macOS/Linux)
 * @param callback Function to call when OS sends media commands (may be NULL)
 * @param user_data Context pointer passed to callback (may be NULL)
 * @return 0 on success, negative error code on failure
 *         -1: General initialization failure
 *         -2: Window handle required but not provided (Windows only)
 *         -3: Platform not supported
 */
XUNE_AUDIO_API int xune_nowplaying_init(
    void* window_handle,
    xune_command_callback_t callback,
    void* user_data
);

/**
 * @brief Check if Now Playing integration is available on this platform
 *
 * Can be called before xune_nowplaying_init() to check availability.
 *
 * @return true if Now Playing is supported, false otherwise
 */
XUNE_AUDIO_API bool xune_nowplaying_is_available(void);

/**
 * @brief Clean up and release Now Playing resources
 *
 * Should be called when shutting down. After this call,
 * xune_nowplaying_init() must be called again before using other functions.
 */
XUNE_AUDIO_API void xune_nowplaying_cleanup(void);

/* ============================================================================
 * Metadata Updates
 * ============================================================================ */

/**
 * @brief Update the currently playing track metadata
 *
 * Call this when the current track changes. The metadata will be displayed
 * in the OS media control surfaces (Control Center, lock screen, etc.).
 *
 * @param metadata Pointer to metadata structure. Must not be NULL.
 */
XUNE_AUDIO_API void xune_nowplaying_set_metadata(
    const xune_track_metadata_t* metadata
);

/**
 * @brief Clear the Now Playing metadata
 *
 * Call this when playback stops completely and no track is loaded.
 */
XUNE_AUDIO_API void xune_nowplaying_clear_metadata(void);

/* ============================================================================
 * Playback State Updates
 * ============================================================================ */

/**
 * @brief Update the playback state
 *
 * IMPORTANT: On macOS, this MUST be called on every state change.
 * Unlike iOS, macOS does not have a central media server that can
 * infer playback state automatically.
 *
 * @param state Current playback state
 */
XUNE_AUDIO_API void xune_nowplaying_set_playback_state(
    xune_playback_state_t state
);

/**
 * @brief Update the current playback position
 *
 * Call this periodically during playback (recommended: every 500ms)
 * to keep the OS scrubber/progress display accurate.
 *
 * @param position_ms Current playback position in milliseconds
 * @param duration_ms Total track duration in milliseconds
 */
XUNE_AUDIO_API void xune_nowplaying_set_position(
    int64_t position_ms,
    int64_t duration_ms
);

/**
 * @brief Update the playback rate
 *
 * Optional. Call this if playback speed changes (e.g., for variable speed playback).
 * Defaults to 1.0 (normal speed) if never called.
 *
 * @param rate Playback rate (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
 */
XUNE_AUDIO_API void xune_nowplaying_set_playback_rate(float rate);

/* ============================================================================
 * Command Enablement
 * ============================================================================ */

/**
 * @brief Enable or disable specific media commands
 *
 * Controls which buttons/actions are available in the OS media controls.
 * By default, Play, Pause, TogglePlayPause, Next, and Previous are enabled.
 *
 * @param command The command to enable or disable
 * @param enabled true to enable, false to disable
 */
XUNE_AUDIO_API void xune_nowplaying_set_command_enabled(
    xune_media_command_t command,
    bool enabled
);

#ifdef __cplusplus
}
#endif
