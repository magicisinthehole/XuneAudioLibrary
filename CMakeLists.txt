cmake_minimum_required(VERSION 3.10)
project(xune_audio_library VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "XuneAudioLibrary Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")

# Chromaprint library configuration
set(CHROMAPRINT_ROOT "${CMAKE_SOURCE_DIR}/vendor/chromaprint")
set(CHROMAPRINT_BUILD_DIR "${CMAKE_BINARY_DIR}/vendor/chromaprint-build")

# Determine library name based on platform
if(APPLE)
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/libchromaprint.dylib")
elseif(WIN32)
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/chromaprint.dll")
else()
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/libchromaprint.so")
endif()

# Check if Chromaprint is already built
if(NOT EXISTS "${CHROMAPRINT_LIBRARY}")
    message(STATUS "Chromaprint library not found, building from source...")

    # Create build directory
    file(MAKE_DIRECTORY ${CHROMAPRINT_BUILD_DIR})

    # Configure Chromaprint
    message(STATUS "Configuring Chromaprint...")
    # Select FFT backend and platform-specific flags
    if(APPLE)
        set(CHROMAPRINT_FFT_LIB "vdsp")       # macOS Accelerate framework (fastest, zero deps)
        set(CHROMAPRINT_PLATFORM_FLAGS -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15)
    elseif(UNIX)
        set(CHROMAPRINT_FFT_LIB "fftw3")      # FFTW3 on Linux (install with: sudo apt install libfftw3-dev)
        set(CHROMAPRINT_PLATFORM_FLAGS)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -S ${CHROMAPRINT_ROOT}
            -B ${CHROMAPRINT_BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=ON
            -DBUILD_TOOLS=OFF
            -DBUILD_TESTS=OFF
            -DFFT_LIB=${CHROMAPRINT_FFT_LIB}
            ${CHROMAPRINT_PLATFORM_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
        RESULT_VARIABLE CHROMAPRINT_CMAKE_RESULT
        OUTPUT_VARIABLE CHROMAPRINT_CMAKE_OUTPUT
        ERROR_VARIABLE CHROMAPRINT_CMAKE_ERROR
    )

    if(NOT CHROMAPRINT_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure Chromaprint:\n${CHROMAPRINT_CMAKE_ERROR}")
    endif()

    # Build Chromaprint
    message(STATUS "Building Chromaprint...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CHROMAPRINT_BUILD_DIR} --config ${CMAKE_BUILD_TYPE}
        RESULT_VARIABLE CHROMAPRINT_BUILD_RESULT
        OUTPUT_VARIABLE CHROMAPRINT_BUILD_OUTPUT
        ERROR_VARIABLE CHROMAPRINT_BUILD_ERROR
    )

    if(NOT CHROMAPRINT_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Chromaprint:\n${CHROMAPRINT_BUILD_ERROR}")
    endif()

    message(STATUS "✓ Chromaprint library built successfully")
else()
    message(STATUS "✓ Using existing Chromaprint library")
endif()

# macOS specific settings
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    message(STATUS "Found Accelerate framework: ${ACCELERATE_LIBRARY}")
    set(PLATFORM_LIBRARIES ${ACCELERATE_LIBRARY})
else()
    set(PLATFORM_LIBRARIES)
endif()

# Auto-copy to Xune.Audio.Native runtime directories
set(AUDIO_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.Audio.Native")
if(EXISTS "${AUDIO_NATIVE_DIR}")
    message(STATUS "Xune.Audio.Native project found")

    # Determine runtime identifier based on platform and architecture
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(RUNTIME_ID "osx-arm64")
        else()
            set(RUNTIME_ID "osx-x64")
        endif()
    elseif(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
            set(RUNTIME_ID "win-arm64")
        else()
            set(RUNTIME_ID "win-x64")
        endif()
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(RUNTIME_ID "linux-arm64")
        else()
            set(RUNTIME_ID "linux-x64")
        endif()
    endif()

    set(RUNTIME_DIR "${AUDIO_NATIVE_DIR}/runtimes/${RUNTIME_ID}/native")
    message(STATUS "Target runtime directory: ${RUNTIME_DIR}")

    # Custom target to copy Chromaprint library
    add_custom_target(copy_chromaprint ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CHROMAPRINT_LIBRARY}" "${RUNTIME_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E echo "Copied ${CHROMAPRINT_LIBRARY} -> ${RUNTIME_DIR}/"
        COMMENT "Copying Chromaprint library to Xune.Audio.Native runtime directory"
        VERBATIM
    )

    message(STATUS "Auto-copy enabled: library will be copied to .NET runtime folder")
else()
    message(WARNING "Xune.Audio.Native not found - library will not be auto-copied")
endif()

# ============================================================================
# XuneAudio Library (Now Playing integration)
# ============================================================================

message(STATUS "")
message(STATUS "Configuring XuneAudio library...")

# Platform-specific Now Playing sources
if(APPLE)
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_macos.mm)
    find_library(MEDIAPLAYER_FRAMEWORK MediaPlayer REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
    set(NOWPLAYING_LIBS ${MEDIAPLAYER_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${APPKIT_FRAMEWORK})
    message(STATUS "  Platform: macOS (MediaPlayer.framework)")
elseif(WIN32)
    # Windows: stub for now, full SMTC implementation later
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Windows (stub)")
elseif(UNIX)
    # Linux: stub for now, D-Bus/MPRIS implementation later
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Linux (stub)")
else()
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Unknown (stub)")
endif()

# Create xune_audio shared library
add_library(xune_audio SHARED ${NOWPLAYING_SOURCES})

# Set library properties
set_target_properties(xune_audio PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Enable exports macro
target_compile_definitions(xune_audio PRIVATE XUNE_AUDIO_EXPORTS)

# Include directories
target_include_directories(xune_audio
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link platform libraries
if(NOWPLAYING_LIBS)
    target_link_libraries(xune_audio PRIVATE ${NOWPLAYING_LIBS})
endif()

# macOS-specific settings
if(APPLE)
    # Enable Objective-C++ for .mm files
    set_source_files_properties(${NOWPLAYING_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    # Set minimum deployment target
    set_target_properties(xune_audio PROPERTIES
        OSX_DEPLOYMENT_TARGET "10.15"
    )
endif()

# Post-build: copy to .NET runtime directory
if(EXISTS "${AUDIO_NATIVE_DIR}")
    add_custom_command(TARGET xune_audio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xune_audio>" "${RUNTIME_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE_NAME:xune_audio>" "${RUNTIME_DIR}/libxune_audio$<IF:$<PLATFORM_ID:Darwin>,.dylib,.so>"
        COMMENT "Copying xune_audio library to ${RUNTIME_DIR}/"
    )
    message(STATUS "  Auto-copy: enabled -> ${RUNTIME_DIR}/")
endif()

message(STATUS "  Sources: ${NOWPLAYING_SOURCES}")
message(STATUS "  Libraries: ${NOWPLAYING_LIBS}")

# ============================================================================
# Configuration Summary
# ============================================================================

# Print summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "----------------------------------------")
message(STATUS "  Project:              ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  Chromaprint library:  ${CHROMAPRINT_LIBRARY}")
message(STATUS "  XuneAudio library:    $<TARGET_FILE:xune_audio>")
message(STATUS "  Runtime ID:           ${RUNTIME_ID}")
message(STATUS "  Platform libraries:   ${PLATFORM_LIBRARIES}")
message(STATUS "========================================")
message(STATUS "")
