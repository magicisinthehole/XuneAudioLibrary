/**
 * @file nowplaying_macos.mm
 * @brief macOS implementation of Now Playing API using MediaPlayer framework
 *
 * Uses MPNowPlayingInfoCenter for displaying track metadata and
 * MPRemoteCommandCenter for receiving media commands from Control Center,
 * Touch Bar, AirPods, and other external controllers.
 */

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>  // For NSImage
#import <MediaPlayer/MediaPlayer.h>
#include "xune_audio/xune_nowplaying.h"

#include <mutex>

// Mutex protecting all mutable global state. Callbacks fire on an OS-managed
// thread; init/cleanup may be called from any thread. The mutex prevents
// TOCTOU races where a callback reads g_callback while cleanup nulls it.
static std::mutex g_mutex;

// Global state (guarded by g_mutex)
static xune_command_callback_t g_callback = nullptr;
static void* g_user_data = nullptr;
static bool g_initialized = false;

// Command handler tokens for cleanup (guarded by g_mutex)
static id g_playTarget = nil;
static id g_pauseTarget = nil;
static id g_toggleTarget = nil;
static id g_stopTarget = nil;
static id g_nextTarget = nil;
static id g_previousTarget = nil;
static id g_seekTarget = nil;
static id g_skipForwardTarget = nil;
static id g_skipBackwardTarget = nil;

// Helper: invoke the callback under the mutex. Called from each command handler block.
static void dispatch_command(xune_media_command_t cmd, int64_t param) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_callback) {
        g_callback(cmd, param, g_user_data);
    }
}

#pragma mark - Lifecycle

int xune_nowplaying_init(
    void* window_handle,
    xune_command_callback_t callback,
    void* user_data)
{
    (void)window_handle;  // Not needed on macOS

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized) {
        return 0;  // Already initialized
    }

    g_callback = callback;
    g_user_data = user_data;

    @autoreleasepool {
        MPRemoteCommandCenter* commandCenter = [MPRemoteCommandCenter sharedCommandCenter];

        // Play command
        g_playTarget = [commandCenter.playCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_PLAY, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Pause command
        g_pauseTarget = [commandCenter.pauseCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_PAUSE, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Toggle play/pause command
        g_toggleTarget = [commandCenter.togglePlayPauseCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_TOGGLE_PLAY_PAUSE, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Stop command
        g_stopTarget = [commandCenter.stopCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_STOP, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Next track command
        g_nextTarget = [commandCenter.nextTrackCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_NEXT, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Previous track command
        g_previousTarget = [commandCenter.previousTrackCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                (void)event;
                dispatch_command(XUNE_CMD_PREVIOUS, 0);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Seek/change playback position command (scrubbing)
        g_seekTarget = [commandCenter.changePlaybackPositionCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                MPChangePlaybackPositionCommandEvent* positionEvent =
                    (MPChangePlaybackPositionCommandEvent*)event;
                int64_t positionMs = (int64_t)(positionEvent.positionTime * 1000.0);
                dispatch_command(XUNE_CMD_SEEK, positionMs);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Skip forward command
        commandCenter.skipForwardCommand.preferredIntervals = @[@15.0];  // 15 seconds
        g_skipForwardTarget = [commandCenter.skipForwardCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                MPSkipIntervalCommandEvent* skipEvent = (MPSkipIntervalCommandEvent*)event;
                int64_t intervalMs = (int64_t)(skipEvent.interval * 1000.0);
                dispatch_command(XUNE_CMD_SKIP_FORWARD, intervalMs);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Skip backward command
        commandCenter.skipBackwardCommand.preferredIntervals = @[@15.0];  // 15 seconds
        g_skipBackwardTarget = [commandCenter.skipBackwardCommand addTargetWithHandler:
            ^MPRemoteCommandHandlerStatus(MPRemoteCommandEvent* _Nonnull event) {
                MPSkipIntervalCommandEvent* skipEvent = (MPSkipIntervalCommandEvent*)event;
                int64_t intervalMs = (int64_t)(skipEvent.interval * 1000.0);
                dispatch_command(XUNE_CMD_SKIP_BACKWARD, intervalMs);
                return MPRemoteCommandHandlerStatusSuccess;
            }];

        // Enable default commands
        commandCenter.playCommand.enabled = YES;
        commandCenter.pauseCommand.enabled = YES;
        commandCenter.togglePlayPauseCommand.enabled = YES;
        commandCenter.stopCommand.enabled = YES;
        commandCenter.nextTrackCommand.enabled = YES;
        commandCenter.previousTrackCommand.enabled = YES;
        commandCenter.changePlaybackPositionCommand.enabled = YES;
        commandCenter.skipForwardCommand.enabled = YES;
        commandCenter.skipBackwardCommand.enabled = YES;
    }

    g_initialized = true;
    return 0;
}

bool xune_nowplaying_is_available(void)
{
    // MediaPlayer framework is available on macOS 10.12.2+
    if (@available(macOS 10.12.2, *)) {
        return true;
    }
    return false;
}

void xune_nowplaying_cleanup(void)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return;
    }

    // Null callback first so any in-flight dispatch_command that acquires the
    // mutex after us will no-op.
    g_callback = nullptr;
    g_user_data = nullptr;

    @autoreleasepool {
        MPRemoteCommandCenter* commandCenter = [MPRemoteCommandCenter sharedCommandCenter];

        // Remove all command targets
        if (g_playTarget) {
            [commandCenter.playCommand removeTarget:g_playTarget];
            g_playTarget = nil;
        }
        if (g_pauseTarget) {
            [commandCenter.pauseCommand removeTarget:g_pauseTarget];
            g_pauseTarget = nil;
        }
        if (g_toggleTarget) {
            [commandCenter.togglePlayPauseCommand removeTarget:g_toggleTarget];
            g_toggleTarget = nil;
        }
        if (g_stopTarget) {
            [commandCenter.stopCommand removeTarget:g_stopTarget];
            g_stopTarget = nil;
        }
        if (g_nextTarget) {
            [commandCenter.nextTrackCommand removeTarget:g_nextTarget];
            g_nextTarget = nil;
        }
        if (g_previousTarget) {
            [commandCenter.previousTrackCommand removeTarget:g_previousTarget];
            g_previousTarget = nil;
        }
        if (g_seekTarget) {
            [commandCenter.changePlaybackPositionCommand removeTarget:g_seekTarget];
            g_seekTarget = nil;
        }
        if (g_skipForwardTarget) {
            [commandCenter.skipForwardCommand removeTarget:g_skipForwardTarget];
            g_skipForwardTarget = nil;
        }
        if (g_skipBackwardTarget) {
            [commandCenter.skipBackwardCommand removeTarget:g_skipBackwardTarget];
            g_skipBackwardTarget = nil;
        }

        // Clear now playing info
        [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo = nil;
    }

    g_initialized = false;
}

#pragma mark - Metadata

void xune_nowplaying_set_metadata(const xune_track_metadata_t* metadata)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized || !metadata) {
        return;
    }

    @autoreleasepool {
        NSMutableDictionary* nowPlayingInfo = [[NSMutableDictionary alloc] init];

        // Title (required)
        if (metadata->title) {
            nowPlayingInfo[MPMediaItemPropertyTitle] =
                [NSString stringWithUTF8String:metadata->title];
        }

        // Artist
        if (metadata->artist) {
            nowPlayingInfo[MPMediaItemPropertyArtist] =
                [NSString stringWithUTF8String:metadata->artist];
        }

        // Album
        if (metadata->album) {
            nowPlayingInfo[MPMediaItemPropertyAlbumTitle] =
                [NSString stringWithUTF8String:metadata->album];
        }

        // Album artist
        if (metadata->album_artist) {
            nowPlayingInfo[MPMediaItemPropertyAlbumArtist] =
                [NSString stringWithUTF8String:metadata->album_artist];
        }

        // Duration
        if (metadata->duration_ms > 0) {
            nowPlayingInfo[MPMediaItemPropertyPlaybackDuration] =
                @(metadata->duration_ms / 1000.0);
        }

        // Artwork from raw data
        if (metadata->artwork_data && metadata->artwork_size > 0) {
            NSData* imageData = [NSData dataWithBytes:metadata->artwork_data
                                               length:metadata->artwork_size];
            NSImage* image = [[NSImage alloc] initWithData:imageData];
            if (image) {
                MPMediaItemArtwork* artwork = [[MPMediaItemArtwork alloc]
                    initWithBoundsSize:image.size
                    requestHandler:^NSImage* _Nonnull(CGSize size) {
                        (void)size;
                        return image;
                    }];
                nowPlayingInfo[MPMediaItemPropertyArtwork] = artwork;
            }
        }
        // Artwork from file path
        else if (metadata->artwork_path) {
            NSString* path = [NSString stringWithUTF8String:metadata->artwork_path];
            NSImage* image = [[NSImage alloc] initWithContentsOfFile:path];
            if (image) {
                MPMediaItemArtwork* artwork = [[MPMediaItemArtwork alloc]
                    initWithBoundsSize:image.size
                    requestHandler:^NSImage* _Nonnull(CGSize size) {
                        (void)size;
                        return image;
                    }];
                nowPlayingInfo[MPMediaItemPropertyArtwork] = artwork;
            }
        }

        [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo = nowPlayingInfo;
    }
}

void xune_nowplaying_clear_metadata(void)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return;
    }

    @autoreleasepool {
        [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo = nil;
    }
}

#pragma mark - Playback State

void xune_nowplaying_set_playback_state(xune_playback_state_t state)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return;
    }

    @autoreleasepool {
        MPNowPlayingPlaybackState mpState;
        switch (state) {
            case XUNE_PLAYBACK_PLAYING:
                mpState = MPNowPlayingPlaybackStatePlaying;
                break;
            case XUNE_PLAYBACK_PAUSED:
                mpState = MPNowPlayingPlaybackStatePaused;
                break;
            case XUNE_PLAYBACK_STOPPED:
            default:
                mpState = MPNowPlayingPlaybackStateStopped;
                break;
        }

        // CRITICAL: On macOS, playback state MUST be set explicitly
        // (unlike iOS, there's no central media server)
        [MPNowPlayingInfoCenter defaultCenter].playbackState = mpState;
    }
}

void xune_nowplaying_set_position(int64_t position_ms, int64_t duration_ms)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return;
    }

    @autoreleasepool {
        NSDictionary* existingInfo = [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo;
        NSMutableDictionary* nowPlayingInfo = existingInfo
            ? [existingInfo mutableCopy]
            : [[NSMutableDictionary alloc] init];

        nowPlayingInfo[MPNowPlayingInfoPropertyElapsedPlaybackTime] =
            @(position_ms / 1000.0);
        nowPlayingInfo[MPMediaItemPropertyPlaybackDuration] =
            @(duration_ms / 1000.0);

        // Set playback rate to 1.0 to indicate normal playback
        // This helps the system calculate current position between updates
        if (!nowPlayingInfo[MPNowPlayingInfoPropertyPlaybackRate]) {
            nowPlayingInfo[MPNowPlayingInfoPropertyPlaybackRate] = @1.0;
        }

        [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo = nowPlayingInfo;
    }
}

void xune_nowplaying_set_playback_rate(float rate)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return;
    }

    @autoreleasepool {
        NSDictionary* existingInfo = [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo;
        NSMutableDictionary* nowPlayingInfo = existingInfo
            ? [existingInfo mutableCopy]
            : [[NSMutableDictionary alloc] init];

        nowPlayingInfo[MPNowPlayingInfoPropertyPlaybackRate] = @(rate);

        [MPNowPlayingInfoCenter defaultCenter].nowPlayingInfo = nowPlayingInfo;
    }
}

#pragma mark - Command Enablement

void xune_nowplaying_set_command_enabled(xune_media_command_t command, bool enabled)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return;
    }

    @autoreleasepool {
        MPRemoteCommandCenter* commandCenter = [MPRemoteCommandCenter sharedCommandCenter];

        switch (command) {
            case XUNE_CMD_PLAY:
                commandCenter.playCommand.enabled = enabled;
                break;
            case XUNE_CMD_PAUSE:
                commandCenter.pauseCommand.enabled = enabled;
                break;
            case XUNE_CMD_TOGGLE_PLAY_PAUSE:
                commandCenter.togglePlayPauseCommand.enabled = enabled;
                break;
            case XUNE_CMD_STOP:
                commandCenter.stopCommand.enabled = enabled;
                break;
            case XUNE_CMD_NEXT:
                commandCenter.nextTrackCommand.enabled = enabled;
                break;
            case XUNE_CMD_PREVIOUS:
                commandCenter.previousTrackCommand.enabled = enabled;
                break;
            case XUNE_CMD_SEEK:
                commandCenter.changePlaybackPositionCommand.enabled = enabled;
                break;
            case XUNE_CMD_SKIP_FORWARD:
                commandCenter.skipForwardCommand.enabled = enabled;
                break;
            case XUNE_CMD_SKIP_BACKWARD:
                commandCenter.skipBackwardCommand.enabled = enabled;
                break;
        }
    }
}
