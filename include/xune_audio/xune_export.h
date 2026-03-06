/**
 * @file xune_export.h
 * @brief Shared export macro and API version for XuneAudioLibrary
 *
 * All public xune_audio headers include this file for the XUNE_AUDIO_API
 * export macro and XUNE_AUDIO_API_VERSION compile-time constant.
 */

#pragma once

/* ============================================================================
 * API Version
 *
 * Bump when the public API changes. C# P/Invoke callers can verify at
 * runtime via xune_audio_api_version() to detect ABI mismatches.
 * ============================================================================ */

#define XUNE_AUDIO_API_VERSION 3

/* ============================================================================
 * Platform Export Macro
 * ============================================================================ */

#if defined(_WIN32)
    #ifdef XUNE_AUDIO_EXPORTS
        #define XUNE_AUDIO_API __declspec(dllexport)
    #else
        #define XUNE_AUDIO_API __declspec(dllimport)
    #endif
#else
    #define XUNE_AUDIO_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * Runtime Version Query
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the API version compiled into the library. */
XUNE_AUDIO_API int xune_audio_api_version(void);

#ifdef __cplusplus
}
#endif
